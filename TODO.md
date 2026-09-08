# TODO

Features and improvements to build. Derived from gaps and `// not implemented`
notes in the code — prune / reprioritise freely. One branch per item (see
`CLAUDE.md`), checked off only after the user has verified it.

## Training

- [ ] **LSTM policy head.** Frame stacking (`--frame-stack N`, done) gives
  the MLP a fixed short history; an LSTM head would carry unbounded memory
  for the POMDP. Needs sequence-based PPO (hidden state through the rollout
  buffer, reset on done, truncated BPTT). Keep the MLP the default.
- [ ] **Separate-process rollout pipeline.** Perf note in the training-perf
  memory: after `--update-threads` / auto rollout threads, the next throughput
  lever is decoupling rollout collection from the update loop (or a GPU box).

## Live bridge (`azalea-bot/`)

- [ ] **Inference server hot-reload.** `inference_server.py` only picks up new
  weights on restart. Watch `model/` (or a `SIGHUP`) and reload `policy.pt` +
  `spec.json` in place.
- [ ] **Live frame stacking.** Training supports `--frame-stack N` but the
  bridges build one frame, so `export_model.py` refuses `N>1` checkpoints.
  Keep the last `N` observations per bot in `azalea_bot` + the mod (clearing
  on match start) and feed the concatenation; read `frame_stack` from
  `spec.json`.
- [ ] **Live splash potions.** Both bridges report `self_effects` now, but:
  `azalea_bot` doesn't distinguish a splash potion's contents (maps
  `ItemKind::SplashPotion` → `Empty`) and neither bridge actually *throws*
  one on the `use_item` action. Map the potion registry → the 5 ids and
  make `use_item` with a potion selected throw it.
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

- [ ] **Non-full blocks.** No slabs/stairs, so the 0.6 step-up never fires on
  terrain. Low priority. Add them, but no kit places/carries them by default.

## Tooling / infra

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
