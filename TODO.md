# TODO

Features and improvements to build. Derived from gaps and `// not implemented`
notes in the code — prune / reprioritise freely. One branch per item (see
`CLAUDE.md`), checked off only after the user has verified it.

## Training

- [ ] **Evaluation harness.** `win_vs_scripted` is the only non-circular
  signal. Add an eval mode that plays the current checkpoint against a ladder
  of past checkpoints + the scripted bot and reports a TrueSkill/ELO number.
- [ ] **Structured metrics.** Console logging only right now — add optional
  TensorBoard / CSV output (returns, win rates, losses, entropy, KL).
- [ ] **Recurrent or frame-stacked policy.** The trunk is a memoryless MLP,
  but the task is a POMDP (latency-delayed view of others, occlusion). Try an
  LSTM head or an N-frame observation stack; keep the MLP as the default.
- [ ] **Automated flat→rough terrain curriculum.** `terrain_flat_only` exists
  as a manual first-pass stage; wire it into a schedule that turns terrain
  amplitude up over the first M updates.
- [ ] **Separate-process rollout pipeline.** Perf note in the training-perf
  memory: after `--update-threads` / auto rollout threads, the next throughput
  lever is decoupling rollout collection from the update loop (or a GPU box).

## Live bridge (`azalea-bot/`)

- [ ] **Inference server hot-reload.** `inference_server.py` only picks up new
  weights on restart. Watch `model/` (or a `SIGHUP`) and reload `policy.pt` +
  `spec.json` in place.
- [ ] **Optional binary `/act` body.** Keep HTTP+JSON as the default, but
  accept a flat-`f32` observation body (and return a flat action) on the same
  endpoint for the case where JSON encode/decode ever shows up in the
  round-trip stats.
- [ ] **Live mining for the `uhc` kit.** `azalea_bot` and the mod hardcode
  `self_mining = 0` and never break placed blocks, so the trained UHC
  block-mining behaviour is dead live. Implement pickaxe/axe mining of placed
  blocks in both bridges.
- [ ] **UHC placement parity.** Verify planks / cobweb / bucket placement via
  `use_item` actually lands where the policy expects (eye raycast ~4.5
  blocks), and expose the golden head (currently unreachable — only 9 hotbar
  slots, no hotkey for it).
- [ ] **ViaProxy online-mode auth.** `run_bot.sh` writes `viaproxy.yml` with
  `auth-method: NONE`; automate the online-mode (Microsoft) path instead of a
  manual edit.

## Sim fidelity (`training/sim/`)

- [ ] **Potions / splash potions.** Biggest missing real-PvP mechanic
  (speed, strength, healing, poison, harming). Currently out of scope.
- [ ] **Enchantment gaps.** Fire Aspect, Flame, Punch, knockback resistance —
  none modelled.
- [ ] **Non-full blocks.** No slabs/stairs, so the 0.6 step-up never fires on
  terrain. Low priority.

## Tooling / infra

- [ ] **CI.** GitHub Actions running `cargo test` (sim + azalea_bot),
  `pytest`, and `./gradlew build` on push / PR.
- [ ] **`torch.export` migration.** `export_model.py` uses TorchScript because
  its on-disk format is stable across torch versions; revisit `.pt2` once it
  makes the same cross-version guarantee.
- [ ] **Reproducible env.** A Dockerfile / devcontainer pinning Rust + Python
  + torch so a fresh machine is one command.

## Mod (`mod/`)

- [ ] **Record match outcome in the JSONL.** `/fight train` currently only
  writes `(obs, action)` per tick and the reward is reconstructed offline;
  logging the real win/loss/HP at episode end would tighten
  `train_from_episodes.py`.
- [ ] **HUD overlay.** Small on-screen readout of mode / detected kit / target
  / inference latency while `/fight` is active.
- [ ] **`/fight reload`.** Re-read `rl-minecraft-ai.properties` without a
  client restart.
