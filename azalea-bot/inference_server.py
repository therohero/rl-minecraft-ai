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

Live frame stacking / recurrent (--lstm) policies: this server is
otherwise stateless (any bot can call it, no session/identity), so the
*client* carries whatever history a policy needs and sends it along each
request rather than the server keeping per-bot state:
  - `frame_stack` > 1 (spec.json): the request may add a `prev_frames` key -
    a JSON array of up to `frame_stack - 1` older observation dicts,
    newest first. Fewer than that (e.g. right after connecting) is fine -
    missing history is zero-padded, exactly like a fresh training episode.
  - a recurrent (`--lstm`) policy (spec.json's `arch.lstm_hidden` > 0): the
    request may add `lstm_h` / `lstm_c` (flat float arrays of that width,
    the carried hidden state) and the response then includes updated
    `lstm_h` / `lstm_c` for the client to send back next call. Omitted /
    absent on the first call - treated as all-zero, the same as a fresh
    sim episode.
Both are additive - a plain JSON body without these keys behaves exactly
as it always has for a `frame_stack=1`, non-recurrent policy.

Optional binary `/act` body: POST with `Content-Type: application/octet-stream`
and the body is the raw little-endian float32 wire row
(`spec.json`'s `wire_floats_per_slot` floats - the same pre-normalization
layout `sim/src/protocol.rs::Observation::write_wire` emits, decoded here
with the already-tested `features.wire_batch_to_obs`, so this adds no new
normalization logic to keep in sync). The response is then also raw
little-endian float32, 10 floats in wire action order (move_x, move_z,
yaw_delta, pitch_delta, jump, attack, sprint, use_item, sneak, held_slot -
booleans as 0.0/1.0). Only for a `frame_stack=1`, non-recurrent policy -
there's no defined binary layout for frame history / LSTM state yet, so
that combination gets a 400. JSON stays the default for every other case.
"""

import argparse
import json
import os
import signal
import struct
import sys
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training", "python"))
import features  # noqa: E402
from device import resolve_device  # noqa: E402
from features import ACTION_FLOATS_PER_SLOT, observation_to_vector  # noqa: E402
from logging_setup import get_logger  # noqa: E402

log = get_logger(__name__)

# Wire order of the flat binary action response - mirrors env.py / the sim's
# Action struct, matching the JSON response's field order.
_BINARY_ACTION_FIELDS = (
    "move_x", "move_z", "yaw_delta", "pitch_delta",
    "jump", "attack", "sprint", "use_item", "sneak", "held_slot",
)


def _load_policy_and_spec(model_dir: str, device: torch.device):
    """Loads `policy.pt` + `spec.json` from `model_dir`. Raises
    `FileNotFoundError` if either is missing."""
    policy_path = os.path.join(model_dir, "policy.pt")
    spec_path = os.path.join(model_dir, "spec.json")
    if not os.path.isfile(policy_path) or not os.path.isfile(spec_path):
        raise FileNotFoundError(
            f"missing {policy_path} or {spec_path} - export a checkpoint first: "
            f"`cd training/python && python export_model.py --checkpoint ../checkpoints/latest.pt "
            f"--out-dir {model_dir}`"
        )
    # `export_model.py` writes a TorchScript module on purpose (its on-disk
    # format is stable across torch versions) - the matching deprecation
    # warning on load is expected.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`torch\.jit\..*` is deprecated")
        policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()
    with open(spec_path) as f:
        spec = json.load(f)
    return policy, spec, os.path.getmtime(policy_path)


class PolicyHandler(BaseHTTPRequestHandler):
    # Set by `main()` before the server starts. Reassigned wholesale (never
    # mutated in place) under `reload_lock` on a hot-reload, and read without
    # a lock on the request path - a request thread sees either the fully-old
    # or fully-new policy/spec, never a torn mix, since Python attribute
    # assignment is atomic and every read here is a single attribute access.
    policy = None
    spec = None
    device = None
    model_dir = None
    # When True, sample every action head from the trained policy's
    # distribution (tanh(mean + std*temp*noise), Bernoulli on the binary
    # heads, temperature-scaled categorical on the slot head) instead of
    # taking the deterministic mode. A frozen-to-its-mode PvP policy is
    # predictable and easy to read; sampling "uses the training more".
    sample = False
    temperature = 1.0

    reload_lock = threading.Lock()
    # mtime of policy.pt as of the last (re)load, so a request can cheaply
    # notice a newer export without re-reading the file every time.
    _policy_mtime = 0.0
    # Set by the SIGHUP handler; checked (and cleared) once per request so
    # `kill -HUP <pid>` forces a reload even if mtimes haven't ticked (e.g.
    # export_model.py finished within the same filesystem-mtime-granularity
    # second, or --model-dir is a symlink swap).
    _reload_requested = threading.Event()

    def log_message(self, fmt, *args):
        pass  # BaseHTTPRequestHandler's default access log; we use `log` (module logger) instead

    @classmethod
    def maybe_reload(cls):
        """Picks up a newer `policy.pt`/`spec.json` in `model_dir` without a
        restart - checked on every request, so `export_model.py` writing a
        fresh checkpoint (or `kill -HUP` on this process) takes effect on
        the very next `/act` call. A failed reload (a checkpoint export
        that's only half-written) logs and keeps serving the last-good
        policy rather than crashing the server."""
        requested = cls._reload_requested.is_set()
        try:
            mtime = os.path.getmtime(os.path.join(cls.model_dir, "policy.pt"))
        except OSError:
            mtime = cls._policy_mtime  # model dir briefly missing mid-export - try again next request
        if not requested and mtime == cls._policy_mtime:
            return
        with cls.reload_lock:
            cls._reload_requested.clear()
            # Someone else's request thread may have already reloaded while
            # we waited for the lock.
            try:
                current_mtime = os.path.getmtime(os.path.join(cls.model_dir, "policy.pt"))
            except OSError:
                return
            if not requested and current_mtime == cls._policy_mtime:
                return
            try:
                policy, spec, new_mtime = _load_policy_and_spec(cls.model_dir, cls.device)
            except Exception:
                log.exception("hot-reload failed - keeping the currently-served policy")
                return
            features.configure(**spec.get("sim_constants", {}))
            cls.policy, cls.spec, cls._policy_mtime = policy, spec, new_mtime
            log.info("hot-reloaded %s (trained_updates=%s)", cls.model_dir, spec.get("trained_updates"))

    def do_GET(self):
        self.maybe_reload()
        # The live bot fetches this once at startup so it normalizes its
        # observations with the exact sim constants / hotbar layout this
        # policy trained against, instead of hardcoding them.
        if self.path == "/spec":
            self._send_json(200, self.spec)
            return
        log.warning("404 for unknown endpoint: %s", self.path)
        self.send_error(404, "unknown endpoint, use GET /spec or POST /act")

    def do_POST(self):
        self.maybe_reload()
        if self.path != "/act":
            log.warning("404 for unknown endpoint: %s", self.path)
            self.send_error(404, "unknown endpoint, use POST /act")
            return

        content_type = self.headers.get("Content-Type", "")
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if content_type.startswith("application/octet-stream"):
                self._handle_binary_act(body)
            else:
                self._handle_json_act(body)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            log.warning("bad request on /act: %s", e)
            self._send_json(400, {"error": f"bad request: {e}"})
        except Exception:
            log.exception("unexpected error handling /act request")
            self._send_json(500, {"error": "internal server error, see server log"})

    def _handle_json_act(self, raw_body: bytes):
        body = json.loads(raw_body)
        frame_stack = int(self.spec.get("frame_stack", 1))
        lstm_hidden = int((self.spec.get("arch") or {}).get("lstm_hidden", 0))

        obs_vec = observation_to_vector(body)
        if frame_stack > 1:
            base_dim = obs_vec.shape[0]
            frames = [obs_vec]
            for f in body.get("prev_frames", [])[: frame_stack - 1]:
                frames.append(observation_to_vector(f))
            while len(frames) < frame_stack:
                # No history yet (just connected / just respawned) - zero-pad
                # the older frames, exactly like a fresh training episode
                # (training/python/frame_stack.py::FrameStacker.reset).
                frames.append(np.zeros(base_dim, dtype=np.float32))
            obs_vec = np.concatenate(frames)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_vec).unsqueeze(0).to(self.device)
            if lstm_hidden:
                h = self._lstm_tensor(body.get("lstm_h"), lstm_hidden)
                c = self._lstm_tensor(body.get("lstm_c"), lstm_hidden)
                cont_mean, cont_std, binary_probs, slot_probs, new_h, new_c = self.policy(
                    obs_tensor, h, c
                )
                lstm_out = (
                    new_h[0, 0].cpu().tolist(),
                    new_c[0, 0].cpu().tolist(),
                )
            else:
                cont_mean, cont_std, binary_probs, slot_probs = self.policy(obs_tensor)
                lstm_out = None
            cont_mean = cont_mean[0].cpu()
            cont_std = cont_std[0].cpu()
            binary_probs = binary_probs[0].cpu()
            slot_probs = slot_probs[0].cpu()
            cont, flags, held_slot = self._select_action(cont_mean, cont_std, binary_probs, slot_probs)

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
        if lstm_out is not None:
            response["lstm_h"], response["lstm_c"] = lstm_out
        self._send_json(200, response)

    def _handle_binary_act(self, raw_body: bytes):
        frame_stack = int(self.spec.get("frame_stack", 1))
        lstm_hidden = int((self.spec.get("arch") or {}).get("lstm_hidden", 0))
        if frame_stack != 1 or lstm_hidden:
            self._send_json(
                400,
                {
                    "error": "the binary /act body only supports a frame_stack=1, non-recurrent "
                    "policy - use the JSON body (with prev_frames / lstm_h / lstm_c) for this one"
                },
            )
            return

        # `WIRE_FLOATS_PER_SLOT` includes the sim's own trailing per-step
        # output (reward/damage/won/lost/done) - not something a live bot
        # has as input, and `wire_batch_to_obs` never reads that far anyway.
        expected = features.WIRE_FLOATS_PER_SLOT - features.OBS_EVENT_FLOATS
        raw = np.frombuffer(raw_body, dtype="<f4")
        if raw.size != expected:
            raise ValueError(f"binary /act body had {raw.size} floats, expected {expected}")

        obs_row = features.wire_batch_to_obs(raw.reshape(1, -1))[0]
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_row).unsqueeze(0).to(self.device)
            cont_mean, cont_std, binary_probs, slot_probs = self.policy(obs_tensor)
            cont_mean = cont_mean[0].cpu()
            cont_std = cont_std[0].cpu()
            binary_probs = binary_probs[0].cpu()
            slot_probs = slot_probs[0].cpu()
            cont, flags, held_slot = self._select_action(cont_mean, cont_std, binary_probs, slot_probs)

        move_x, move_z, yaw_delta, pitch_delta = (float(x) for x in cont.tolist())
        scale = self.spec["yaw_pitch_delta_scale_radians"]
        values = {
            "move_x": move_x,
            "move_z": move_z,
            "yaw_delta": yaw_delta * scale,
            "pitch_delta": pitch_delta * scale,
            "jump": 1.0 if flags.get("jump", False) else 0.0,
            "attack": 1.0 if flags.get("attack", False) else 0.0,
            "sprint": 1.0 if flags.get("sprint", False) else 0.0,
            "use_item": 1.0 if flags.get("use_item", False) else 0.0,
            "sneak": 1.0 if flags.get("sneak", False) else 0.0,
            "held_slot": float(held_slot),
        }
        assert len(values) == ACTION_FLOATS_PER_SLOT
        body = struct.pack(f"<{ACTION_FLOATS_PER_SLOT}f", *(values[k] for k in _BINARY_ACTION_FIELDS))
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _lstm_tensor(self, values, lstm_hidden: int) -> torch.Tensor:
        """A request's `lstm_h`/`lstm_c` (a flat list, or absent/None on the
        first call) -> the `[1, 1, lstm_hidden]` tensor the traced recurrent
        module expects."""
        if values is None:
            values = [0.0] * lstm_hidden
        elif len(values) != lstm_hidden:
            raise ValueError(f"lstm_h/lstm_c had {len(values)} floats, expected {lstm_hidden}")
        return torch.as_tensor(values, dtype=torch.float32).view(1, 1, lstm_hidden).to(self.device)

    def _select_action(self, cont_mean, cont_std, binary_probs, slot_probs):
        """Deterministic mode, or (with `--sample`) a draw from the trained
        policy's distribution. Returns `(cont[4], flags: dict, held_slot: int)`."""
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
        return cont, flags, held_slot

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    PolicyHandler.sample = args.sample
    PolicyHandler.temperature = args.temperature
    PolicyHandler.device = resolve_device(args.device)
    PolicyHandler.model_dir = args.model_dir
    log.info("loading policy: %s (device=%s)", args.model_dir, PolicyHandler.device)
    try:
        policy, spec, mtime = _load_policy_and_spec(args.model_dir, PolicyHandler.device)
    except FileNotFoundError as e:
        log.error("%s", e)
        raise
    PolicyHandler.policy = policy
    PolicyHandler.spec = spec
    PolicyHandler._policy_mtime = mtime

    # Normalize incoming observations exactly the way the policy was trained
    # to expect - spec.json carries the sim constants that run used.
    features.configure(**PolicyHandler.spec.get("sim_constants", {}))
    log.info("observation normalization: %s", features.active_constants())

    # `kill -HUP <pid>` forces a hot-reload on the next request even without
    # a newer file mtime to notice (see `PolicyHandler.maybe_reload`).
    if hasattr(signal, "SIGHUP"):
        def _on_sighup(_signum, _frame):
            log.info("SIGHUP received - will reload %s on the next request", args.model_dir)
            PolicyHandler._reload_requested.set()

        signal.signal(signal.SIGHUP, _on_sighup)

    try:
        server = ThreadingHTTPServer((args.host, args.port), PolicyHandler)
    except OSError as e:
        log.error("failed to bind %s:%d: %s (is another server already running on this port?)", args.host, args.port, e)
        raise

    log.info(
        "serving %s on http://%s:%d/act (action selection: %s) - watching for a newer "
        "policy.pt on every request%s",
        args.model_dir,
        args.host,
        args.port,
        f"sampled, temperature={args.temperature}" if args.sample else "deterministic mode",
        " (also: kill -HUP %d to force a reload)" % os.getpid() if hasattr(signal, "SIGHUP") else "",
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
