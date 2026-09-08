"""Tests for `device.py`: `--device` string resolution. CPU is always
present; every other backend must fail loudly when absent rather than
silently fall back."""

import pytest
import torch

from device import available_devices, resolve_device


def test_cpu_always_resolves_and_is_listed():
    assert resolve_device("cpu") == torch.device("cpu")
    assert "cpu" in available_devices()


def test_auto_returns_a_concrete_device():
    dev = resolve_device("auto")
    assert isinstance(dev, torch.device) or hasattr(dev, "type")


def test_blank_is_treated_as_auto():
    assert resolve_device("  ").type == resolve_device("auto").type


@pytest.mark.parametrize("name", ["cuda", "mps", "xpu"])
def test_unavailable_backend_raises(name):
    avail = {
        "cuda": torch.cuda.is_available(),
        "mps": getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available(),
        "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
    }
    if avail[name]:
        pytest.skip(f"{name} is actually available on this box")
    with pytest.raises(RuntimeError):
        resolve_device(name)


def test_directml_without_package_raises():
    try:
        import torch_directml  # noqa: F401

        pytest.skip("torch-directml is installed")
    except ImportError:
        pass
    with pytest.raises(RuntimeError):
        resolve_device("directml")
