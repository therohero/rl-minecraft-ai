"""Exports a training checkpoint into a self-contained artifact that can be
loaded outside this repo (e.g. by azalea-bot/inference_server.py) to play
live against the trained bot.

Produces, in --out-dir:
  policy.pt      - TorchScript module, `forward(obs: Tensor[N, OBS_DIM]) ->
                   (cont_mean: Tensor[N, 4], cont_std: Tensor[N, 4],
                    binary_probs: Tensor[N, 5], slot_probs: Tensor[N, S])`.
                   `cont_mean` is the raw pre-tanh continuous action mean and
                   `cont_std` its learned per-dim std - the inference server
                   tanh-squashes the mean for deterministic play, or samples
                   `tanh(mean + std * randn)` when run with `--sample` so a
                   live duel can use the trained policy's full distribution
                   instead of only its mode.
  spec.json      - obs/action field order + constants, so any downstream
                   consumer (even a non-Python bot) knows how to build the
                   observation vector and interpret the action output.

Usage:
    python export_model.py --checkpoint ../checkpoints/latest.pt --out-dir ../../azalea-bot/model
"""

import argparse
import json
import os
import warnings

import torch
import torch.nn as nn

import features
from features import CONTINUOUS_ACTION_DIM, BINARY_ACTION_DIM
from logging_setup import get_logger
from ppo_agent import ActorCritic

log = get_logger(__name__)


class InferencePolicy(nn.Module):
    """Thin wrapper exposing exactly what a live bot needs: the raw
    continuous-action mean and its learned std, the binary-action
    probabilities, and the held-item softmax probabilities. The value head
    and all training-only machinery are dropped. The inference server either
    takes the deterministic mode (tanh(mean), prob > threshold, argmax slot)
    or samples from these distributions - see `azalea-bot/inference_server.py
    --sample`.
    """

    def __init__(self, actor_critic: ActorCritic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs: torch.Tensor):
        mean, log_std, binary_logits, slot_logits, _value = self.actor_critic.forward(obs)
        cont_std = log_std.exp()
        binary_probs = torch.sigmoid(binary_logits)
        slot_probs = torch.softmax(slot_logits, dim=-1)
        return mean, cont_std, binary_probs, slot_probs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="../../azalea-bot/model")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        log.error("checkpoint not found: %s", args.checkpoint)
        raise FileNotFoundError(args.checkpoint)

    os.makedirs(args.out_dir, exist_ok=True)

    log.info("loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    frame_stack = ckpt.get("frame_stack", 1)
    if frame_stack != 1:
        raise SystemExit(
            f"this checkpoint was trained with --frame-stack {frame_stack}. The live inference "
            "bridges (azalea-bot, mod) build a single observation frame and don't stack yet, so "
            "an N>1 checkpoint can't be exported for live play. Train the deployable policy with "
            "--frame-stack 1, or implement live frame stacking first (see TODO.md)."
        )

    # Restore the exact sim-side normalization constants and trunk shape this
    # checkpoint was trained with, so the exported policy + spec.json match.
    sim_constants = ckpt.get("sim_constants", {})
    features.configure(**sim_constants)
    arch = ckpt.get("arch", {})
    obs_dim = ckpt.get("obs_dim", features.OBS_DIM)
    if obs_dim != features.OBS_DIM:
        log.warning(
            "checkpoint obs_dim=%d but current features.py + sim_constants compute %d - "
            "exporting at the checkpoint's size",
            obs_dim,
            features.OBS_DIM,
        )

    model = ActorCritic(
        obs_dim,
        hidden_size=arch.get("hidden_size", 256),
        num_layers=arch.get("num_layers", 2),
        slot_dim=arch.get("slot_dim", features.HOTBAR_ACTION_DIM),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapped = InferencePolicy(model)
    example_input = torch.zeros(1, obs_dim)

    # TorchScript is deprecated in recent torch in favour of torch.export, but
    # its on-disk format is far more stable *across* torch versions - which is
    # exactly what this artifact needs (it's loaded by azalea-bot, possibly on
    # a different machine / torch build, possibly months later). torch.export's
    # `.pt2` serialization makes no such cross-version guarantee yet. Revisit
    # when it does. The policy is a plain feed-forward net (Linear/Tanh/
    # sigmoid/softmax), so a trace is exact.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`torch\.jit\..*` is deprecated")
        scripted = torch.jit.trace(wrapped, example_input)
        policy_path = os.path.join(args.out_dir, "policy.pt")
        scripted.save(policy_path)

    c = features.active_constants()
    sim_config = ckpt.get("sim_config") or {}
    _combat = sim_config.get("combat") or {}
    self_fields = [
        "self_hp (/max_hp)", "self_vel_x", "self_vel_y", "self_vel_z",
        "sin(self_yaw)", "cos(self_yaw)", "sin(self_pitch)", "cos(self_pitch)",
        "self_on_ground (0/1)", "self_attack_cooldown", "self_ping_ms (/max_ping_ms)",
        "self_shield (0..1)", "self_dist_from_center (/arena_radius)",
        "self_ground_height (/terrain_max_amplitude)",
        "self_slope_forward (/terrain_max_amplitude)",
        "self_slope_right (/terrain_max_amplitude)", "self_hurt (0..1)",
        "self_held (/ (ITEM_COUNT-1))", "self_absorption (/max_hp)", "self_eating (0..1)",
        "self_bow_draw (0..1)", "self_burning (0/1)", "self_shield_disabled (0..1)",
        "self_arrows (/16)", "self_slot (/ (HOTBAR_SLOTS-1))", "self_swap_lockout (0..1)",
        "self_food (/20)", "self_sneaking (0/1)", "self_mining (0..1)",
    ]
    other_fields = [
        "present (0/1)", "hp+absorption (/max_hp)", "rel_x", "rel_y", "rel_z",
        "vel_x", "vel_y", "vel_z", "ground_height (/terrain_max_amplitude)",
        "blocking (0..1)", "eating (0/1)", "held_ranged (0/1)", "sneaking (0/1)",
    ]
    inv_items = [
        "sword", "axe", "pickaxe", "bow", "crossbow", "planks", "cobweb",
        "water_bucket", "lava_bucket", "golden_apple", "golden_head",
    ]
    proj_fields = ["present (0/1)", "rel_x", "rel_y", "rel_z", "vel_x", "vel_y", "vel_z"]
    col_fields = ["top_rel (/terrain_max_amplitude)", "water (0/1)", "lava (0/1)", "cobweb (0/1)"]
    # The kit's physical hotbar layout (slot -> item), mirroring
    # sim/src/kit.rs::loadout. The live bot maps a policy slot choice
    # straight onto the real hotbar key.
    hotbar_layouts = {
        "sword": ["sword"] + ["empty"] * 8,
        "axe": ["sword", "axe", "bow", "crossbow"] + ["empty"] * 5,
        "uhc": [
            "sword", "axe", "pickaxe", "bow", "crossbow",
            "planks", "cobweb", "water_bucket", "golden_apple",
        ],
    }
    hotbar_layout = hotbar_layouts.get(sim_config.get("kit") or "sword", hotbar_layouts["sword"])

    obs_field_order = list(self_fields)
    obs_field_order += [f"inventory.{it} (count/16)" for it in inv_items]
    obs_field_order += [f"hotbar.slot{k} (item_id/(ITEM_COUNT-1))" for k in range(features.HOTBAR_SLOTS)]
    for k in range(c.max_observed_enemies):
        obs_field_order += [f"enemy{k}.{f}" for f in other_fields]
    for k in range(c.max_observed_teammates):
        obs_field_order += [f"teammate{k}.{f}" for f in other_fields]
    for k in range(c.max_observed_projectiles):
        obs_field_order += [f"projectile{k}.{f}" for f in proj_fields]
    ncols = c.block_view_size * c.block_view_size
    for k in range(ncols):
        obs_field_order += [f"block_col{k}.{f}" for f in col_fields]
    obs_field_order += ["time_left (/match_time_seconds)", "enemies_alive", "teammates_alive"]

    spec = {
        "obs_dim": obs_dim,
        # 1 for every deployable checkpoint today; the live bridges assume it
        # and export refuses anything else. Present so a future live
        # frame-stacking path has the depth to read.
        "frame_stack": frame_stack,
        "obs_field_order": obs_field_order,
        "kit": sim_config.get("kit"),
        "team_size": sim_config.get("team_size"),
        "attribute_swapping": sim_config.get("attribute_swapping"),
        "max_observed_enemies": c.max_observed_enemies,
        "max_observed_teammates": c.max_observed_teammates,
        "max_observed_projectiles": c.max_observed_projectiles,
        "block_view_size": c.block_view_size,
        "inventory_items": inv_items,
        "continuous_action_dim": CONTINUOUS_ACTION_DIM,
        "continuous_action_order": ["move_x", "move_z", "yaw_delta", "pitch_delta"],
        "continuous_action_range": [-1.0, 1.0],
        "binary_action_dim": BINARY_ACTION_DIM,
        "binary_action_order": ["jump", "attack", "sprint", "use_item", "sneak"],
        "binary_action_threshold": 0.5,
        "policy_outputs": ["cont_mean", "cont_std", "binary_probs", "slot_probs"],
        "input_order": sim_config.get("input_order"),
        "hotbar_slots": features.HOTBAR_SLOTS,
        # held_slot action: `0..hotbar_slots` selects a physical slot (press
        # that number key); `hotbar_slots..held_slot_action_dim` hotkeys
        # `item_ids[held_slot - hotbar_slots]` into the selected slot (9 =
        # empty hand). `hotbar_layout` is the kit's starting slot -> item.
        "held_slot_action_dim": model.slot_dim,
        "hotbar_action_dim": model.slot_dim,
        "held_slot_semantics": "0..hotbar_slots select; hotbar_slots.. hotkey item_ids[a-hotbar_slots]",
        "hotbar_layout": hotbar_layout,
        "item_ids": [
            "empty", "sword", "axe", "pickaxe", "bow", "crossbow",
            "planks", "cobweb", "water_bucket", "lava_bucket", "golden_apple", "golden_head",
        ],
        "yaw_pitch_delta_scale_radians": features.active_constants().max_look_delta,
        # The sim constants this policy trained against - inference_server.py
        # feeds these straight back into features.configure().
        "sim_constants": features.active_constants().__dict__,
        # Combat timing constants the live bot needs to normalise the four
        # observation fields it now derives itself (self_bow_draw,
        # self_swap_lockout, self_hurt, self_shield_disabled) the same way
        # sim/src/observation.rs does. Pulled from the full SimConfig the sim
        # sent in its Hello (stored on the checkpoint as `sim_config`);
        # defaults mirror sim/src/config.rs::CombatConfig::default().
        "combat_constants": {
            "bow_max_draw_seconds": _combat.get("bow_max_draw_seconds", 1.0),
            "swap_lockout_seconds": _combat.get("swap_lockout_seconds", 0.05),
            "axe_shield_disable_seconds": _combat.get("axe_shield_disable_seconds", 5.0),
            "hurt_invulnerability_seconds": _combat.get("hurt_invulnerability_seconds", 1.0),
        },
        "arch": {
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "slot_dim": model.slot_dim,
        },
        "constants": {
            "max_hp": features.active_constants().max_hp,
            "match_time_seconds": features.active_constants().match_time_seconds,
        },
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "trained_updates": ckpt.get("update"),
    }
    spec_path = os.path.join(args.out_dir, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    log.info("wrote %s", policy_path)
    log.info("wrote %s", spec_path)
    log.info("use azalea-bot/inference_server.py to serve this model to a live bot.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("export failed")
        raise
