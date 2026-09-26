# TODO

Features and improvements to build. Derived from gaps and `// not implemented`
notes in the code — prune / reprioritise freely. One branch per item (see
`CLAUDE.md`), checked off only after the user has verified it.

## Live bridge (`azalea-bot/`)

- [ ] **`azalea_bot` live features need a real-server check.** LSTM inference,
  frame stacking, splash potions, `uhc` mining, and the ViaProxy Microsoft-auth
  setup were all implemented and unit/protocol-tested (`cargo test`, a live
  `inference_server.py` round-trip for JSON/binary/frame-stack/LSTM/hot-reload),
  but none have run against a *real* server / real match yet - see
  `azalea-bot/README.md`'s new sections for what to look for:
  correct behaviour under live latency, mining actually breaking a placed
  plank/cobweb block without misfiring on world terrain, a potion actually
  landing its effect, and the ViaProxy device-code flow actually completing
  end to end (needs a human to open the printed URL - untestable here).
- [ ] **UHC placement / golden head: real-server check.** Reasoned to already
  be correct by construction (`azalea_bot`'s placement and the block-view
  reach both go through azalea's own `block_interaction_range`-driven
  crosshair hit-testing, which is vanilla's 4.5 blocks - matching the sim's
  `place_reach` exactly - see azalea-bot/README.md), but never checked
  against a real server.

- [ ] **Remaining GrimAC flags on the local test server.** A first run of
  `azalea_bot` against `mod/test-server` (Paper 1.21.11 + GrimAC 2.3.74) flagged
  Simulation (hundreds, constant ~0.02 / ~0.10 offsets), NoSlow, BadPacketsJ and
  AntiKB. The guard now holds still and releases the bow while an item is in use
  (targets NoSlow and likely part of Simulation) - re-run and check what's left;
  Simulation / BadPacketsJ / AntiKB are not yet root-caused. A proper fix for the
  use-item slowdown is patching azalea's `local_player_ai_step` (`TODO: using
  items`) so a slowed walk is possible instead of standing still.

## Mod (`mod/`)

- [ ] **Bring the mod's live bridge up to parity with `azalea_bot`.**
  `azalea_bot` now supports live LSTM inference, live frame stacking, live
  splash potions (mapping + throwing), and live `uhc` mining (see
  azalea-bot/README.md for how each works and the wire-protocol additions
  `inference_server.py` now accepts); the mod (`ObservationBuilder` /
  `InferenceClient` / `FightController`'s action application) doesn't have
  any of these yet, so a `--lstm`, `--frame-stack N>1`, splash-potion, or
  `uhc`-mining checkpoint is still dead when played through `/fight`.
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
