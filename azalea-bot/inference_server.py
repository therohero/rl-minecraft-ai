"""Live-inference HTTP bridge: loads an exported policy and serves actions
to a real Minecraft bot over plain HTTP+JSON, so the bot can be written in
any language (Java/Kotlin Fabric mod, mineflayer/Node.js, etc.) - it just
needs to POST the current game state and read back an action.

This is intentionally NOT a Minecraft mod/plugin itself - actually reading
the world state out of a live Minecraft client/server and turning it into
mouse/keyboard or packet actions is specific to whatever bot framework you
use (mineflayer, Baritone, a Fabric mod, etc.). This server is the stable
seam: your bot integration only needs to speak this tiny HTTP protocol and
never has to know about PyTorch, checkpoints, or training internals.

Run:
    python inference_server.py --model-dir ./model --port 8800

Then from your bot, for each decision tick, POST to /act:
    POST http://127.0.0.1:8800/act
    { "self_hp": 20.0, "self_vel_x": 0.0, ..., "time_left": 30.0 }
  -> { "move_x": 0.3, "move_z": 1.0, "yaw_delta": -0.1, "pitch_delta": 0.0,
       "jump": false, "attack": true, "sprint": true }

See spec.json (written by export_model.py) for the exact observation
fields expected - they match sim/src/protocol.rs's `Observation` struct
field-for-field, since that's what the policy was trained on. Your bot
integration is responsible for computing those fields (positions,
velocities, relative opponent position, HP, etc.) from the real game.
"""

import argparse
import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training", "python"))
import features  # noqa: E402
from device import resolve_device  # noqa: E402
from features import observation_to_vector  # noqa: E402
from logging_setup import get_logger  # noqa: E402

log = get_logger(__name__)


class PolicyHandler(BaseHTTPRequestHandler):
    # Set by `main()` before the server starts; shared read-only across
    # request threads (the TorchScript module is only ever used for
    # inference here, so this is safe without extra locking).
    policy = None
    spec = None
    device = None
    # When True, sample every action head from the trained policy's
    # distribution (tanh(mean + std*temp*noise), Bernoulli on the binary
    # heads, temperature-scaled categorical on the slot head) instead of
    # taking the deterministic mode. A frozen-to-its-mode PvP policy is
    # predictable and easy to read; sampling "uses the training more".
    sample = False
    temperature = 1.0

    def log_message(self, fmt, *args):
        pass  # BaseHTTPRequestHandler's default access log; we use `log` (module logger) instead

    def do_GET(self):
        # The live bot fetches this once at startup so it normalizes its
        # observations with the exact sim constants / hotbar layout this
        # policy trained against, instead of hardcoding them.
        if self.path == "/spec":
            self._send_json(200, self.spec)
            return
        log.warning("404 for unknown endpoint: %s", self.path)
        self.send_error(404, "unknown endpoint, use GET /spec or POST /act")

    def do_POST(self):
        if self.path != "/act":
            log.warning("404 for unknown endpoint: %s", self.path)
            self.send_error(404, "unknown endpoint, use POST /act")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            obs_vec = observation_to_vector(body)

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_vec).unsqueeze(0).to(self.device)
                cont_mean, cont_std, binary_probs, slot_probs = self.policy(obs_tensor)
                cont_mean = cont_mean[0].cpu()
                cont_std = cont_std[0].cpu()
                binary_probs = binary_probs[0].cpu()
                slot_probs = slot_probs[0].cpu()

                names = self.spec.get(
                    "binary_action_order", ["jump", "attack", "sprint", "use_item", "sneak"]
                )
                thr = self.spec["binary_action_threshold"]
                if self.sample:
                    t = max(self.temperature, 1e-3)
                    cont = torch.tanh(cont_mean + cont_std * t * torch.randn_like(cont_mean))
                    binary = torch.bernoulli(binary_probs.clamp(0.0, 1.0))
                    slot_logits = slot_probs.clamp_min(1e-8).log() / t
                    held_slot = int(torch.distributions.Categorical(logits=slot_logits).sample().item())
                    flags = {name: bool(v) for name, v in zip(names, binary.tolist())}
                else:
                    cont = torch.tanh(cont_mean)
                    flags = {name: (p > thr) for name, p in zip(names, binary_probs.tolist())}
                    held_slot = int(slot_probs.argmax().item())

            move_x, move_z, yaw_delta, pitch_delta = cont.tolist()
            scale = self.spec["yaw_pitch_delta_scale_radians"]

            response = {
                "move_x": move_x,
                "move_z": move_z,
                "yaw_delta": yaw_delta * scale,
                "pitch_delta": pitch_delta * scale,
                "jump": flags.get("jump", False),
                "attack": flags.get("attack", False),
                "sprint": flags.get("sprint", False),
                "use_item": flags.get("use_item", False),
                "sneak": flags.get("sneak", False),
                "held_slot": held_slot,
            }
            self._send_json(200, response)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            log.warning("bad request on /act: %s", e)
            self._send_json(400, {"error": f"bad request: {e}"})
        except Exception:
            log.exception("unexpected error handling /act request")
            self._send_json(500, {"error": "internal server error, see server log"})

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default="./model")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="interface to bind (0.0.0.0 for all)")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device for inference (default cpu - a single-observation policy this small is "
        "usually faster on CPU than paying host<->GPU transfer per request). 'auto' / cuda / mps / "
        "xpu / directml also work (see training/python/device.py).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="sample each action from the trained policy's distribution instead of taking the "
        "deterministic mode - a less predictable, harder-to-read live opponent",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="with --sample: >1 widens the sampled distribution, <1 sharpens it toward the mode "
        "(default 1.0 = the exact trained distribution)",
    )
    args = parser.parse_args()

    # This policy trunk is tiny and served one observation at a time; extra
    # intra-op threads only add scheduling overhead per request.
    torch.set_num_threads(1)

    policy_path = os.path.join(args.model_dir, "policy.pt")
    spec_path = os.path.join(args.model_dir, "spec.json")

    if not os.path.isfile(policy_path) or not os.path.isfile(spec_path):
        log.error(
            "missing %s or %s - export a checkpoint first: "
            "`cd training/python && python export_model.py --checkpoint ../checkpoints/latest.pt --out-dir %s`",
            policy_path,
            spec_path,
            args.model_dir,
        )
        raise FileNotFoundError(f"no exported model found in {args.model_dir}")

    PolicyHandler.sample = args.sample
    PolicyHandler.temperature = args.temperature
    PolicyHandler.device = resolve_device(args.device)
    log.info("loading policy: %s (device=%s)", policy_path, PolicyHandler.device)
    # `export_model.py` writes a TorchScript module on purpose (its on-disk
    # format is stable across torch versions, unlike torch.export's .pt2) -
    # the matching deprecation warning on load is expected.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`torch\.jit\..*` is deprecated")
        PolicyHandler.policy = torch.jit.load(policy_path, map_location=PolicyHandler.device)
    PolicyHandler.policy.eval()
    with open(spec_path) as f:
        PolicyHandler.spec = json.load(f)

    # Normalize incoming observations exactly the way the policy was trained
    # to expect - spec.json carries the sim constants that run used.
    features.configure(**PolicyHandler.spec.get("sim_constants", {}))
    log.info("observation normalization: %s", features.active_constants())

    try:
        server = ThreadingHTTPServer((args.host, args.port), PolicyHandler)
    except OSError as e:
        log.error("failed to bind %s:%d: %s (is another server already running on this port?)", args.host, args.port, e)
        raise

    log.info(
        "serving %s on http://%s:%d/act (action selection: %s)",
        policy_path,
        args.host,
        args.port,
        f"sampled, temperature={args.temperature}" if args.sample else "deterministic mode",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("inference server crashed")
        raise
