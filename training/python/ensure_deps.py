"""Make the active Python environment match *this* machine.

`run.sh` / `run.ps1` call this with the repo-local `.venv` interpreter right
after creating it (and on every re-run - it's a fast no-op once things line
up). It exists because the one dependency that matters here, PyTorch, ships
as two very different wheels:

  * the default PyPI `torch` wheel bundles the full CUDA/cuDNN/NCCL stack
    (~3.5 GB) - what you want on an NVIDIA box, pure dead weight without one;
  * `torch ... --index-url https://download.pytorch.org/whl/cpu` is the
    slim (~200 MB) CPU-only build.

So "does this repo need a GPU" should not be a thing the user configures.
This script:

  1. decides the wanted backend from the hardware - `cuda` if an NVIDIA GPU
     is visible (``nvidia-smi`` on PATH or ``/proc/driver/nvidia``), else
     ``cpu``; on macOS always the default wheel (it already carries MPS and
     no CUDA bloat). Override with ``RL_TORCH_BACKEND=cpu|cuda|auto``.
  2. installs torch from the matching index if it's missing, OR reinstalls
     it if the installed build is the wrong kind for this machine (a
     CPU-only wheel on a CUDA box, or a CUDA wheel on a box with no GPU).
     Going to CPU also purges the now-orphaned ``nvidia-*`` / ``triton``
     packages so the disk space actually comes back.
  3. makes sure ``numpy`` and ``pytest`` are present.

Run it directly any time to re-sync: ``.venv/bin/python training/python/ensure_deps.py``
(``--dry-run`` just prints what it would do).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_SPEC = "torch>=2.1"
OTHER_SPECS = ["numpy>=1.24", "pytest>=8.0"]


def _run(cmd: list[str], dry: bool) -> None:
    print("  $", " ".join(cmd))
    if not dry:
        subprocess.check_call(cmd)


def _pip(*args: str) -> list[str]:
    return [sys.executable, "-m", "pip", *args]


def nvidia_gpu_present() -> bool:
    """A usable NVIDIA GPU is visible on this machine."""
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=10,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            pass
    # Driver loaded but nvidia-smi missing (some container images).
    return os.path.exists("/proc/driver/nvidia/version")


def wanted_backend() -> str:
    """`cuda`, `cpu`, or `default` (let pip pick - macOS/other)."""
    override = os.environ.get("RL_TORCH_BACKEND", "auto").strip().lower()
    if override in ("cpu", "cuda"):
        return override
    if override not in ("", "auto"):
        print(f"[ensure_deps] ignoring unknown RL_TORCH_BACKEND={override!r}")

    system = platform.system()
    if system == "Darwin":
        return "default"  # the PyPI wheel already ships MPS, no CUDA payload
    if system not in ("Linux", "Windows"):
        return "default"
    return "cuda" if nvidia_gpu_present() else "cpu"


def installed_torch() -> tuple[str, str | None] | None:
    """`(version, cuda_tag_or_None)` for the installed torch, or None."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch.__version__, getattr(torch.version, "cuda", None)


def _orphan_cuda_packages() -> list[str]:
    out = subprocess.run(
        _pip("list", "--format=freeze"),
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    names = []
    for line in out.splitlines():
        name = line.split("==", 1)[0].strip()
        if name.startswith("nvidia-") or name == "triton":
            names.append(name)
    return names


def sync(dry: bool = False) -> None:
    want = wanted_backend()
    current = installed_torch()
    print(f"[ensure_deps] machine wants torch backend: {want}"
          + ("" if current is None else f"; installed: {current[0]}"
             f" (cuda={current[1] or 'none'})"))

    need_torch = current is None
    if current is not None and want != "default":
        has_cuda = current[1] is not None
        if want == "cuda" and not has_cuda:
            print("[ensure_deps] installed torch is CPU-only but a CUDA build is "
                  "wanted here - reinstalling the CUDA wheel")
            need_torch = True
        elif want == "cpu" and has_cuda:
            print("[ensure_deps] installed torch carries the CUDA stack but this "
                  "machine has no NVIDIA GPU - reinstalling the slim CPU wheel")
            need_torch = True

    if need_torch:
        _run(_pip("install", "--quiet", "--upgrade", "pip"), dry)
        going_cpu = want == "cpu"
        if current is not None:
            # A plain reinstall of the CUDA wheel over a CPU one (or vice
            # versa) is what actually swaps the backend.
            purge = ["torch"]
            if going_cpu:
                purge += _orphan_cuda_packages()
            _run(_pip("uninstall", "--quiet", "-y", *purge), dry)
        cmd = _pip("install", "--quiet", TORCH_SPEC)
        if going_cpu:
            cmd += ["--index-url", CPU_INDEX]
        _run(cmd, dry)
        if going_cpu and not dry:
            subprocess.run(_pip("cache", "purge"), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)

    missing = []
    for spec in OTHER_SPECS:
        mod = spec.split(">=", 1)[0].split("==", 1)[0]
        try:
            __import__(mod)
        except ImportError:
            missing.append(spec)
    if missing:
        _run(_pip("install", "--quiet", *missing), dry)

    if not need_torch and not missing:
        print("[ensure_deps] environment already matches this machine - nothing to do")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="print the pip commands that would run, change nothing")
    args = p.parse_args()
    try:
        sync(dry=args.dry_run)
    except subprocess.CalledProcessError as e:
        print(f"[ensure_deps] a pip command failed ({e}); install manually with "
              f"`{sys.executable} -m pip install -r training/python/requirements.txt`",
              file=sys.stderr)
        sys.exit(1)
