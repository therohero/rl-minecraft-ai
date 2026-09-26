"""End-to-end tests for `azalea-bot/inference_server.py`'s `/act` contract -
in particular the `lstm_state` request/response field an `--lstm` checkpoint
adds (see export_model.py's `InferenceRecurrentPolicy` and the server's
module docstring). Drives the real server as a subprocess, like
test_export_model.py does for export_model.py, since the contract under
test is the actual HTTP+JSON wire shape a live bot integration depends on.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import torch

import features
from ppo_agent import ActorCritic

PYDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(PYDIR))
EXPORT = os.path.join(PYDIR, "export_model.py")
SERVER = os.path.join(REPO_ROOT, "azalea-bot", "inference_server.py")

MIN_OBS = {
    "self_hp": 20.0,
    "self_vel_x": 0.0,
    "self_vel_y": 0.0,
    "self_vel_z": 0.0,
    "self_yaw": 0.0,
    "self_pitch": 0.0,
    "self_on_ground": True,
    "self_attack_cooldown": 1.0,
    "self_dist_from_center": 0.0,
    "time_left": 90.0,
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _export(tmp_path, lstm_hidden: int):
    features.configure()
    model = ActorCritic(features.OBS_DIM, hidden_size=16, num_layers=1, lstm_hidden=lstm_hidden)
    ckpt = str(tmp_path / "latest.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "update": 1,
            "obs_dim": features.OBS_DIM,
            "arch": {
                "hidden_size": 16, "num_layers": 1, "slot_dim": model.slot_dim,
                "lstm_hidden": lstm_hidden,
            },
            "sim_constants": features.active_constants().__dict__,
            "sim_config": {"kit": "sword"},
        },
        ckpt,
    )
    out = tmp_path / "model"
    subprocess.run(
        [sys.executable, EXPORT, "--checkpoint", ckpt, "--out-dir", str(out)],
        check=True, capture_output=True, text=True,
    )
    return str(out)


class _Server:
    """Launches `inference_server.py --model-dir ... --port ...` and tears
    it down; blocks until `/spec` answers so callers never race the load."""

    def __init__(self, model_dir: str):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc = subprocess.Popen(
            [sys.executable, SERVER, "--model-dir", model_dir, "--port", str(self.port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    def __enter__(self):
        deadline = time.time() + 20
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"inference_server.py exited early:\n{out}")
            try:
                urllib.request.urlopen(f"{self.base}/spec", timeout=1)
                return self
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        raise TimeoutError("inference_server.py never answered /spec")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)

    def act(self, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/act", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())


def test_mlp_checkpoint_act_has_no_lstm_state(tmp_path):
    model_dir = _export(tmp_path, lstm_hidden=0)
    with _Server(model_dir) as server:
        resp = server.act(MIN_OBS)
        assert "lstm_state" not in resp
        assert "move_x" in resp and "held_slot" in resp


def test_lstm_checkpoint_act_returns_carryable_state(tmp_path):
    hidden = 8
    model_dir = _export(tmp_path, lstm_hidden=hidden)
    with _Server(model_dir) as server:
        # First call: no lstm_state given -> server starts from zeros.
        r1 = server.act(MIN_OBS)
        state = r1["lstm_state"]
        assert len(state["h"]) == hidden
        assert len(state["c"]) == hidden

        # Carry that state into the next call - the server must accept it
        # and hand back an updated one of the same shape.
        r2 = server.act({**MIN_OBS, "lstm_state": state})
        assert len(r2["lstm_state"]["h"]) == hidden
        assert len(r2["lstm_state"]["c"]) == hidden
        # Two ticks of the same LSTM cell from a non-zero incoming state
        # should not land on exactly the same state again (would indicate
        # the server silently ignored the carried state and re-zeroed it).
        assert r2["lstm_state"]["h"] != state["h"]


def test_lstm_checkpoint_act_tolerates_a_garbled_state(tmp_path):
    """A malformed/mismatched lstm_state degrades to zeroed rather than a
    400 - see the server's module docstring for why."""
    model_dir = _export(tmp_path, lstm_hidden=8)
    with _Server(model_dir) as server:
        resp = server.act({**MIN_OBS, "lstm_state": {"h": [1.0, 2.0], "c": "not a list"}})
        assert len(resp["lstm_state"]["h"]) == 8
