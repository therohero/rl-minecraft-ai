"""End-to-end smoke test for the training loop - the thing to run to verify
a training-side change actually works, not just that the unit tests pass.

It launches the real Rust sim + `train.py` twice against a throwaway
checkpoint dir, on a deliberately tiny config (few arenas, short rollouts,
a handful of updates) so the whole thing finishes in well under a minute,
then asserts on the run's own log output + the files it left behind:

  1. a fresh run trains N updates, writes `latest.pt` + numbered snapshots
     (honouring --keep-checkpoints), and freezes opponent-league snapshots;
  2. a second run *resumes* from `latest.pt` at the right update and runs
     on to a higher update count;
  3. `evaluate.py` rates the resulting checkpoints against each other + the
     scripted bot and prints an Elo table.

Exit code 0 = the training loop is healthy. Non-zero prints what failed.

Usage:
    python smoke_train.py                 # uses ../sim/target/release/mc_pvp_sim
    python smoke_train.py --sim-binary /path/to/mc_pvp_sim
    python smoke_train.py --keep          # don't delete the temp dir (to inspect)

This is a test harness, not a training entry point - `run.sh` is that.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

from env import DEFAULT_SIM_BINARY

HERE = os.path.dirname(os.path.abspath(__file__))

# Small enough that both runs finish fast; big enough to exercise a real
# rollout, a PPO update, a checkpoint save/prune, and the opponent league.
FRESH_UPDATES = 6
RESUME_UPDATES = 10
COMMON_ARGS = [
    "--num-arenas", "16",
    "--rollout-len", "8",
    "--checkpoint-every", "3",
    "--keep-checkpoints", "2",
    "--opponent-fraction", "0.3",
    "--opponent-pool-size", "3",
    "--opponent-snapshot-every", "2",
    "--seed", "0",
    "--device", "cpu",
]


class SmokeFailure(RuntimeError):
    pass


def _run_training(sim_binary: str, checkpoint_dir: str, total_updates: int, fresh: bool) -> str:
    cmd = [
        sys.executable, os.path.join(HERE, "train.py"),
        "--sim-binary", sim_binary,
        "--checkpoint-dir", checkpoint_dir,
        "--total-updates", str(total_updates),
        *COMMON_ARGS,
    ]
    if fresh:
        cmd.append("--fresh")
    return _run(cmd, "train.py")


def _run_eval(sim_binary: str, checkpoint_dir: str) -> str:
    cmd = [
        sys.executable, os.path.join(HERE, "evaluate.py"),
        "--sim-binary", sim_binary,
        "--checkpoint-dir", checkpoint_dir,
        "--num-arenas", "16",
        "--matches-per-pair", "6",
        "--match-time", "8",
        "--max-ladder", "3",
        "--seed", "0",
    ]
    return _run(cmd, "evaluate.py")


def _run(cmd: list[str], what: str) -> str:
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise SmokeFailure(f"{what} exited {proc.returncode}:\n{_indent(out)}")
    return out


def _indent(text: str, prefix: str = "    | ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _require(cond: bool, msg: str, log: str) -> None:
    if not cond:
        raise SmokeFailure(f"{msg}\n--- run log ---\n{_indent(log)}")


def _last_update_in_log(log: str) -> int:
    updates = [int(m) for m in re.findall(r"\bupdate=(\d+)", log)]
    if not updates:
        raise SmokeFailure(f"no 'update=<n>' lines in the run log:\n{_indent(log)}")
    return max(updates)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-binary", default=os.path.join(HERE, DEFAULT_SIM_BINARY))
    ap.add_argument("--keep", action="store_true", help="keep the temp checkpoint dir for inspection")
    args = ap.parse_args()

    sim_binary = os.path.abspath(args.sim_binary)
    if not os.path.isfile(sim_binary):
        print(f"FAIL: sim binary not found at {sim_binary}\n      build it: (cd ../sim && cargo build --release)")
        return 2

    tmp = tempfile.mkdtemp(prefix="rl-smoke-")
    ckpt = os.path.join(tmp, "checkpoints")
    try:
        print(f"[1/2] fresh run: {FRESH_UPDATES} updates -> {ckpt}")
        log1 = _run_training(sim_binary, ckpt, FRESH_UPDATES, fresh=True)
        _require(os.path.isfile(os.path.join(ckpt, "latest.pt")), "fresh run left no latest.pt", log1)
        numbered = sorted(f for f in os.listdir(ckpt) if re.match(r"policy_update_\d+\.pt$", f))
        _require(bool(numbered), "fresh run left no numbered checkpoint", log1)
        _require(len(numbered) <= 2, f"--keep-checkpoints 2 not honoured: {numbered}", log1)
        _require("added policy snapshot to opponent pool" in log1,
                 "no opponent-league snapshot was frozen during the fresh run", log1)
        league = os.path.join(ckpt, "league")
        persisted = sorted(f for f in os.listdir(league)) if os.path.isdir(league) else []
        _require(bool(persisted), "opponent league was not persisted to checkpoints/league/", log1)
        _require(len(persisted) <= 3, f"--opponent-pool-size 3 not honoured on disk: {persisted}", log1)
        _require(_last_update_in_log(log1) == FRESH_UPDATES,
                 f"fresh run stopped at update {_last_update_in_log(log1)}, expected {FRESH_UPDATES}", log1)
        print(f"      ok: latest.pt + {len(numbered)} numbered snapshot(s) + {len(persisted)} league snapshot(s)")

        print(f"[2/2] resume run: continue to {RESUME_UPDATES} updates")
        log2 = _run_training(sim_binary, ckpt, RESUME_UPDATES, fresh=False)
        m = re.search(r"resumed from .*latest\.pt at update=(\d+)", log2)
        _require(m is not None, "resume run did not log 'resumed from ... at update=<n>'", log2)
        _require(int(m.group(1)) == FRESH_UPDATES + 1,
                 f"resumed at update={m.group(1)}, expected {FRESH_UPDATES + 1}", log2)
        _require(re.search(r"loaded \d+ opponent-league snapshot", log2) is not None,
                 "resume run did not reload the persisted opponent league", log2)
        _require(_last_update_in_log(log2) == RESUME_UPDATES,
                 f"resume run stopped at update {_last_update_in_log(log2)}, expected {RESUME_UPDATES}", log2)
        print("      ok: resumed from latest.pt at the right update, reloaded the league, ran on")

        print("[3/3] eval run: rate the checkpoints against each other + scripted")
        log3 = _run_eval(sim_binary, ckpt)
        _require("evaluating" in log3 and "players" in log3, "evaluate.py did not start a tournament", log3)
        _require(re.search(r"candidate:latest\.pt\b.*<- candidate", log3) is not None,
                 "evaluate.py report is missing the candidate row", log3)
        _require(re.search(r"score vs the field \(\d+ matches\)", log3) is not None,
                 "evaluate.py did not print the candidate summary", log3)
        _require("vs scripted" in log3, "evaluate.py did not play the scripted bot", log3)
        print("      ok: Elo table printed with the candidate rated against the ladder")

        print("\nSMOKE OK - training loop is healthy")
        return 0
    except (SmokeFailure, subprocess.TimeoutExpired) as e:
        print(f"\nSMOKE FAILED: {e}")
        return 1
    finally:
        if args.keep:
            print(f"(kept {tmp})")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
