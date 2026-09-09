# TODO

Features and improvements to build. Derived from gaps and `// not implemented`
notes in the code — prune / reprioritise freely. One branch per item (see
`CLAUDE.md`), checked off only after the user has verified it.

## Training

- [ ] **Separate-process rollout pipeline.** Perf note in the training-perf
  memory: after `--update-threads` / auto rollout threads, the next throughput
  lever is decoupling rollout collection from the update loop (or a GPU box).

## Live bridge (`azalea-bot/`)

- [ ] **Inference server hot-reload.** `inference_server.py` only picks up new
  weights on restart. Watch `model/` (or a `SIGHUP`) and reload `policy.pt` +
  `spec.json` in place.
- [ ] **Live LSTM inference.** Training supports `--lstm` (recurrent policy
  head) but the bridges keep no state between ticks, so `export_model.py`
  refuses an `--lstm` checkpoint. Keep the last LSTM `(h, c)` per bot in
  `azalea_bot` + the mod (zeroed on match start), thread it through the
  `policy.pt` call, and read `lstm_hidden` from `spec.json`. Pairs with
  live frame stacking below.
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
- [ ] **Safe live-server training (important).** `/fight train` today
  assumes a cooperative target on your own world. To gather real data on an
  actual PvP server without producing garbage episodes or getting banned,
  roughly in priority order:
  - **Stop fighting on death.** On death, hand control straight back to the
    player - don't resume inputs the instant you respawn. A bot that
    re-engages after every death is both useless data and an obvious ban.
    `/fight stopfightingtoggle` opts back into fight-through-respawn for
    auto-round arenas where that's actually wanted.
  - **One episode per fight.** A `/fight train` session currently spans
    everything until `/fight stop`. Cut a new episode file (+ its own
    win/loss/HP outcome record) on each death / kill so
    `train_from_episodes.py` sees clean per-fight returns.
  - **Explicit target lock.** `TargetSelector.nearest` will swing at
    bystanders on a populated server. Add `/fight target <name>` and/or
    "only whoever last hit me" / "only inside this region", and never
    auto-acquire a replacement once the locked target is gone.
  - **Engage only when provoked or told.** Default to not attacking until
    the target hits first (or you `/fight target` them); idle instead of
    chasing when there's no valid target.
  - **Auto-pause on menu / GUI / dimension change / spectator.** Stop
    streaming inputs *and* dataset rows whenever the player has a screen
    open or the server moved them out of the fight.
  - **Mod-side anticheat naturalism.** `ClientGuard` is lighter than
    `azalea_bot`'s guard; a server running Grim/Vulcan needs the same
    reaction-delay / click-timing / rotation treatment. Pull the shared
    logic into one place both bridges use.
  - **Richer episode tags.** Record server address, opponent name, and a
    real-match-vs-practice flag so offline training can filter / downweight
    junk sessions.
- [ ] **`/fight reload`.** Re-read `rl-minecraft-ai.properties` without a
  client restart.
