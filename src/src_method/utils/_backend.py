"""Array-module backend selection for CPU (numpy) and GPU (cupy).

Kept intentionally minimal: a single resolver returns the appropriate
array module, a PRNG factory, and a host-transfer helper.  All hot-loop
code paths receive an ``xp`` module and call ``xp.linalg.*`` /
``xp.asarray`` directly, so backend selection adds zero per-op overhead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from numpy.typing import DTypeLike, NDArray


def get_xp(device: str) -> ModuleType:
    """Return the array module for the requested device.

    Args:
        device: ``"cpu"`` for numpy or ``"gpu"`` for cupy.

    Returns:
        The numpy or cupy module.

    Raises:
        ValueError: If ``device`` is not recognised.
        ImportError: If ``device="gpu"`` but cupy is not installed.
    """
    if device == "cpu":
        return np
    if device == "gpu":
        import cupy  # noqa: PLC0415  (lazy: optional dependency)

        return cupy
    msg = f"Unknown device {device!r}; expected 'cpu' or 'gpu'."
    raise ValueError(msg)


def default_rng(
    seed: int | None,
) -> np.random.Generator:
    """Return a seeded NumPy ``Generator``.

    Always uses NumPy so that the same seed produces identical draws
    regardless of the device, and avoids CuPy ``Generator`` API
    differences (e.g. missing ``.normal()``).
    """
    return np.random.default_rng(seed)


def to_numpy(arr: NDArray) -> np.ndarray:
    """Bring an array onto the host as a numpy array (no-op for numpy)."""
    if isinstance(arr, np.ndarray):
        return arr
    # cupy.ndarray exposes .get(); fall back to np.asarray for other dispatchers.
    get = getattr(arr, "get", None)
    return get() if callable(get) else np.asarray(arr)


def sketch_dtype(dtype: DTypeLike | None, *inputs: Sequence[NDArray]) -> np.dtype:
    """Resolve the sketch dtype.

    Args:
        dtype: Explicit sketch dtype, or None to follow the inputs.
        *inputs: MPS or MPO tensors.

    Returns:
        The explicit dtype, or the real floating counterpart of the inputs dtype.
    """
    if dtype is not None:
        return np.dtype(dtype)
    input_dtype = np.result_type(*(arr.dtype for inpt in inputs for arr in inpt))
    if input_dtype.kind in "fc":
        return np.finfo(input_dtype).dtype
    return np.dtype(np.float64)
