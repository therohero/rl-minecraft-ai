"""Torch device selection with support for non-NVIDIA accelerators.

`torch.cuda` only covers NVIDIA (and, transparently, AMD ROCm builds of
torch - a ROCm wheel still reports its GPUs through `torch.cuda`). This
helper additionally recognizes:

  - Apple Silicon    via `torch.backends.mps`
  - Intel GPUs (XPU) via `torch.xpu` (needs a torch build with XPU support)
  - DirectML         via the `torch-directml` package (any Direct3D 12 GPU
                     on Windows / WSL2 - AMD, Intel, NVIDIA, Qualcomm)

Usage:
    from device import resolve_device
    device = resolve_device(args.device)   # args.device defaults to "auto"
"""

import torch

from logging_setup import get_logger

log = get_logger(__name__)

_DIRECTML_ALIASES = {"dml", "directml", "torch_directml"}


def _directml_device():
    try:
        import torch_directml
    except ImportError as e:
        raise RuntimeError(
            "requested the DirectML device but `torch-directml` is not installed - "
            "`pip install torch-directml` (Windows / WSL2 only; see python/requirements.txt)"
        ) from e
    if not torch_directml.is_available() or torch_directml.device_count() < 1:
        raise RuntimeError("torch-directml is installed but reports no available DirectML device")
    return torch_directml.device()


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def available_devices() -> list[str]:
    """Human-readable list of accelerators this install can actually use,
    for logging / `--help`."""
    found = ["cpu"]
    if torch.cuda.is_available():
        # A ROCm build of torch also answers here; `torch.version.hip`
        # distinguishes it from a real CUDA build.
        found.append("cuda (ROCm)" if getattr(torch.version, "hip", None) else "cuda")
    if _mps_available():
        found.append("mps")
    if _xpu_available():
        found.append("xpu")
    try:
        import torch_directml

        if torch_directml.is_available():
            found.append("directml")
    except ImportError:
        pass
    return found


def resolve_device(requested: str = "auto") -> torch.device:
    """Maps a `--device` string to a concrete `torch.device`.

    "auto" (the default) picks the fastest accelerator present, in order:
    CUDA/ROCm, Apple MPS, Intel XPU, DirectML, then CPU. An explicit value
    ("cuda", "cpu", "mps", "xpu", "directml"/"dml", or a full spec like
    "cuda:1") is honored as-is and raises if it isn't actually available,
    so a typo or a missing driver fails loudly instead of silently
    training on the CPU.
    """
    req = requested.strip().lower()

    if req in _DIRECTML_ALIASES:
        dev = _directml_device()
        log.info("using DirectML device: %s", dev)
        return dev

    if req in ("", "auto"):
        if torch.cuda.is_available():
            kind = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            log.info("auto-selected %s GPU", kind)
            return torch.device("cuda")
        if _mps_available():
            log.info("auto-selected Apple MPS GPU")
            return torch.device("mps")
        if _xpu_available():
            log.info("auto-selected Intel XPU GPU")
            return torch.device("xpu")
        try:
            dev = _directml_device()
            log.info("auto-selected DirectML device: %s", dev)
            return dev
        except RuntimeError:
            pass
        log.info("no GPU detected - using CPU (available: %s)", ", ".join(available_devices()))
        return torch.device("cpu")

    device = torch.device(req)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch reports no CUDA/ROCm GPU")
    if device.type == "mps" and not _mps_available():
        raise RuntimeError("--device mps requested but torch reports no available MPS backend")
    if device.type == "xpu" and not _xpu_available():
        raise RuntimeError("--device xpu requested but torch reports no available XPU backend")
    log.info("using requested device: %s", device)
    return device
