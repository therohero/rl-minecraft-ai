"""`export_model.py` is what freezes a training checkpoint into the
`policy.pt` + `spec.json` pair the live bot loads. These tests drive the
real script end-to-end on a minimal checkpoint and check that the pieces
the `azalea_bot` bridge depends on land in `spec.json` - in particular the
`combat_constants` block it needs to normalise the observation fields it
derives itself."""

import json
import os
import subprocess
import sys

import torch

import features
from ppo_agent import ActorCritic

PYDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT = os.path.join(PYDIR, "export_model.py")


def _write_checkpoint(path, sim_config):
    features.configure()  # defaults
    model = ActorCritic(features.OBS_DIM, hidden_size=16, num_layers=1)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "update": 1,
            "obs_dim": features.OBS_DIM,
            "arch": {"hidden_size": 16, "num_layers": 1, "slot_dim": model.slot_dim},
            "sim_constants": features.active_constants().__dict__,
            "sim_config": sim_config,
        },
        path,
    )


def _export(tmp_path, sim_config):
    ckpt = str(tmp_path / "latest.pt")
    out = tmp_path / "model"
    _write_checkpoint(ckpt, sim_config)
    subprocess.run(
        [sys.executable, EXPORT, "--checkpoint", ckpt, "--out-dir", str(out)],
        check=True,
        capture_output=True,
    )
    with open(out / "spec.json") as f:
        return json.load(f)


def test_combat_constants_default_when_sim_config_has_no_combat(tmp_path):
    spec = _export(tmp_path, {"kit": "sword"})
    cc = spec["combat_constants"]
    assert cc["bow_max_draw_seconds"] == 1.0
    assert cc["swap_lockout_seconds"] == 0.05
    assert cc["axe_shield_disable_seconds"] == 5.0
    assert cc["hurt_invulnerability_seconds"] == 1.0


def test_combat_constants_follow_a_non_default_sim_config(tmp_path):
    spec = _export(
        tmp_path,
        {"kit": "axe", "combat": {"bow_max_draw_seconds": 1.5, "axe_shield_disable_seconds": 8.0}},
    )
    cc = spec["combat_constants"]
    assert cc["bow_max_draw_seconds"] == 1.5
    assert cc["axe_shield_disable_seconds"] == 8.0
    # untouched keys still fall back to the sim default
    assert cc["swap_lockout_seconds"] == 0.05


def test_spec_still_has_the_fields_the_bridge_reads(tmp_path):
    spec = _export(tmp_path, {"kit": "sword"})
    for key in ("sim_constants", "combat_constants", "input_order", "hotbar_layout", "item_ids"):
        assert key in spec
