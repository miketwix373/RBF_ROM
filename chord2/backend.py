"""Array-backend shim: numpy on CPU, cupy on GPU.

CLAUDE.md mandates that any GPU acceleration in CHORD2 go through a single
`get_backend(device)` shim and that CuPy be the only GPU dependency. This
module provides that shim plus a small set of helpers for moving arrays
between host and device transparently.

Usage
-----
    from chord2.backend import get_backend, asnumpy

    xp, on_gpu = get_backend("auto")       # or "cpu" / "gpu"
    a = xp.asarray(numpy_array)            # uploads to device if on GPU
    s = xp.linalg.svd(a, full_matrices=False)
    s_host = asnumpy(s[1])                 # back to numpy for plotting

The shim returns `(xp, on_gpu)` where `xp` is the array module to use
(`numpy` or `cupy`) and `on_gpu` is a bool flag callers can branch on
when they need to (e.g. to pick a CPU-only library path for an algorithm
not implemented in cupy).
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np


_VALID_DEVICES = ("auto", "cpu", "gpu")


def get_backend(device: str = "auto") -> Tuple[object, bool]:
    """Return `(xp, on_gpu)` for the requested device.

    Parameters
    ----------
    device
        - `"cpu"`  : force numpy.
        - `"gpu"`  : force cupy; raise if cupy or a CUDA device is missing.
        - `"auto"` : use cupy if importable and `cupy.cuda.runtime.getDeviceCount()`
                     reports at least one device, else fall back to numpy.

    Returns
    -------
    (xp, on_gpu)
        `xp` is the array module (`numpy` or `cupy`); `on_gpu` is True iff
        the cupy path was taken.
    """
    if device not in _VALID_DEVICES:
        raise ValueError(f"device must be one of {_VALID_DEVICES}, got {device!r}")

    if device == "cpu":
        return np, False

    try:
        import cupy as cp  # type: ignore
    except ImportError as e:
        if device == "gpu":
            raise RuntimeError(
                "device='gpu' requested but cupy is not importable: " f"{e}"
            ) from e
        return np, False

    try:
        n_dev = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        if device == "gpu":
            raise RuntimeError(
                f"device='gpu' requested but no CUDA device is visible: {e}"
            ) from e
        print(f"chord2.backend: cupy import OK but no CUDA device ({e}); "
              f"falling back to numpy", file=sys.stderr)
        return np, False

    if n_dev < 1:
        if device == "gpu":
            raise RuntimeError("device='gpu' requested but getDeviceCount()==0")
        return np, False

    return cp, True


def asnumpy(x):
    """Return a numpy view/copy of `x`, regardless of whether it lives on the GPU.

    - `cupy.ndarray` -> host copy via `.get()`.
    - `numpy.ndarray` (or anything `np.asarray` can handle) -> unchanged.
    """
    get = getattr(x, "get", None)
    if callable(get) and type(x).__module__.startswith("cupy"):
        return get()
    return np.asarray(x)


def to_device(x, xp):
    """Move array `x` onto the backend `xp`.

    Calling `xp.asarray(x)` does the right thing for both directions
    (host->device when `xp` is cupy, no-op when `xp` is numpy and `x`
    is already a numpy array). This thin wrapper exists so the caller's
    intent is obvious in the code.
    """
    return xp.asarray(x)


def describe_device(xp, on_gpu: bool) -> str:
    """One-line human description of the active backend, for logging."""
    if not on_gpu:
        return f"numpy {np.__version__} (CPU)"
    try:
        import cupy as cp  # type: ignore
        dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"]
        mem_gb = dev["totalGlobalMem"] / 1e9
        return f"cupy {cp.__version__} on {name} ({mem_gb:.1f} GB)"
    except Exception as e:
        return f"cupy (device query failed: {e})"
