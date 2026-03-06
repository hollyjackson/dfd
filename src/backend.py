"""Backend abstraction for CPU/GPU toggle.

Provides a single point of control for switching between NumPy (CPU) and
CuPy (GPU) backends.  All computational modules import from here rather
than directly from numpy/scipy, making the GPU mode a drop-in replacement.

Usage::

    import backend
    backend.set_backend(use_gpu=True)

    xp = backend.xp()          # cupy or numpy
    sp = backend.sparse_module()  # cupyx.scipy.sparse or scipy.sparse
    ndi = backend.ndimage_module()  # cupyx.scipy.ndimage or scipy.ndimage
"""

import numpy as np
import scipy.sparse
import scipy.ndimage

_use_gpu = False


def set_backend(use_gpu: bool):
    """Set the global backend.  Call once at startup before any computation."""
    global _use_gpu
    if use_gpu:
        try:
            import cupy  # noqa: F401
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU mode but is not installed. "
                "Install it with: pip install cupy-cuda12x  (or cupy-cuda11x)"
            )
    _use_gpu = use_gpu


def get_backend() -> bool:
    """Return True if GPU mode is active."""
    return _use_gpu


def xp():
    """Return the array module: cupy (GPU) or numpy (CPU)."""
    if _use_gpu:
        import cupy
        return cupy
    return np


def sparse_module():
    """Return the sparse module: cupyx.scipy.sparse (GPU) or scipy.sparse (CPU)."""
    if _use_gpu:
        import cupyx.scipy.sparse
        return cupyx.scipy.sparse
    return scipy.sparse


def ndimage_module():
    """Return the ndimage module: cupyx.scipy.ndimage (GPU) or scipy.ndimage (CPU)."""
    if _use_gpu:
        import cupyx.scipy.ndimage
        return cupyx.scipy.ndimage
    return scipy.ndimage


def sparse_linalg_module():
    """Return the sparse linalg module: cupyx.scipy.sparse.linalg or scipy.sparse.linalg."""
    if _use_gpu:
        import cupyx.scipy.sparse.linalg
        return cupyx.scipy.sparse.linalg
    return scipy.sparse.linalg


def to_device(arr):
    """Move array to the current device (GPU if active, otherwise no-op)."""
    if _use_gpu:
        import cupy
        if isinstance(arr, cupy.ndarray):
            return arr
        return cupy.asarray(arr)
    return np.asarray(arr)


def to_cpu(arr):
    """Pull array to CPU (numpy).  No-op if already a numpy array."""
    if _use_gpu:
        import cupy
        if isinstance(arr, cupy.ndarray):
            return arr.get()
    return np.asarray(arr)
