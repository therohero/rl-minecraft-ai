# TODO

Features and improvements to build. Derived from gaps and `// not implemented`
notes in the code — prune / reprioritise freely. One branch per item (see
`CLAUDE.md`), checked off only after the user has verified it.

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

## Mod (`mod/`)

- [ ] **Tune the mod virtual-mouse against a real anticheat.** `ClientGuard`'s
  new mouse model has plausible-but-guessed defaults (`max_yaw_accel_deg`,
  `aim_latency_ticks`, `max_cps`, tremor). Validate / retune them against
  GrimAC (and ideally Vulcan) on a test server; consider a small
  rotation-log dump to eyeball the delta distribution.
- [ ] **`/fight reload`.** Re-read `rl-minecraft-ai.properties` without a
  client restart.
- [ ] **Extend the debug-harness selftest to cover the outcome path.** The
  autorun selftest (`RL_DEBUG_AUTORUN=selftest ./gradlew runClient`) now runs
  end to end - opens the test world, drives `SelfTest` against a mock
  inference server, writes `run/rl-debug/selftest-*.json` with a pass/fail
  exit code. Gaps it can't reach solo: a landed hit / real CPS and a
  death-or-kill episode boundary both need a real second player (client-side
  dummies have no server entity). Add a headless second `azalea` bot (or a
  server-side fake player) so `cps_within_cap` and the win/loss outcome
  assertions become real, then the `mod/` self-merge gate can rely on them.
  A real in-fight episode boundary (kill/death/disengage) also unlocks
  testing an `--lstm` checkpoint's reset-at-episode-boundary half of the
  contract - `lstm_state_carried` in `SelfTest` today only covers that the
  state is round-tripped between ticks, not that it's zeroed at the
  boundary (verified by inspection: every boundary funnels through
  `FightController.resetEpisodeOutcome`, which calls
  `InferenceClient.resetLstmState`).
