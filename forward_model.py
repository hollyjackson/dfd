"""Forward model for depth-from-defocus.

Each focal-plane image is modelled as a sparse matrix-vector product::

    b_i = A_i @ x

where ``x`` is the vectorised all-in-focus (AIF) image and ``A_i`` is the
per-pixel Gaussian blur operator for focal plane *i*.  The blur radius at
each pixel is determined by the circle-of-confusion (CoC) formula given the
camera parameters in ``globals``.

Public API
----------
precompute_indices  -- build pixel-shift index tables (call once per image size)
buildA              -- assemble the sparse blur operator stack for a depth map
forward             -- apply the full forward model: AIF -> defocus stack
"""
import numpy as np
import scipy

import globals


# ---------------------------------------------------------------------------
# Kernel geometry helpers
# ---------------------------------------------------------------------------

def compute_u_v():
    """Return meshgrid coordinate arrays for the blur kernel.

    Returns
    -------
    u, v : ndarray, shape (1, 1, 1, K, K)
        Row and column offset coordinates over the K×K kernel window,
        broadcast-ready for per-pixel kernel evaluation.
    """
    max_kernel_size = globals.MAX_KERNEL_SIZE
    lim = max_kernel_size // 2

    us = np.linspace(-lim, lim, max_kernel_size, dtype=np.float32)
    vs = np.linspace(-lim, lim, max_kernel_size, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(us, vs, indexing='ij')
    u = grid_u[None, None, None, ...]
    v = grid_v[None, None, None, ...]

    return u, v


def compute_shifted_indices(width, height):
    """Return per-pixel neighbour index arrays for the K×K kernel window.

    For each pixel (i, j) and each kernel offset (di, dj) within the
    K×K window, stores the absolute row and column index of the
    corresponding neighbour.

    Parameters
    ----------
    width, height : int
        Image dimensions.

    Returns
    -------
    row_indices, col_indices : ndarray, shape (width, height, K, K)
        Absolute row / column indices for each pixel's kernel neighbours.
    """
    max_kernel_size = globals.MAX_KERNEL_SIZE
    lim = max_kernel_size // 2

    row_indices = np.zeros((width, height, max_kernel_size, max_kernel_size), dtype=np.intp)
    col_indices  = np.zeros((width, height, max_kernel_size, max_kernel_size), dtype=np.intp)

    grid = np.meshgrid(np.arange(width, dtype=np.intp), np.arange(height, dtype=np.intp), indexing='ij')
    indices = np.stack(grid, axis=-1)

    for i in range(-lim, lim+1):
        for j in range(-lim, lim+1):
            row = indices[:, :, 0] + i
            col = indices[:, :, 1] + j
            row_indices[:, :, i+lim, j+lim] = row
            col_indices[:, :, i+lim, j+lim] = col

    return row_indices, col_indices


def generate_mask(row_indices, col_indices, width, height):
    """Return a boolean mask that removes out-of-bounds kernel entries.

    Parameters
    ----------
    row_indices, col_indices : ndarray, shape (width, height, K, K)
        Absolute neighbour indices from ``compute_shifted_indices``.
    width, height : int
        Image dimensions.

    Returns
    -------
    mask : ndarray of bool, shape (width * height * K * K,)
        True where the neighbour index falls within the image boundary.
    """
    condition1 = row_indices.flatten() < 0
    condition2 = row_indices.flatten() >= width
    condition3 = col_indices.flatten() < 0
    condition4 = col_indices.flatten() >= height
    indices_to_delete = np.where(condition1 | condition2 | condition3 | condition4)
    mask = np.ones(col_indices.flatten().shape[0], dtype=bool)
    mask[indices_to_delete] = False

    return mask


def compute_mask_flattened_indices(row_indices, col_indices, mask, width, height):
    """Convert 2-D neighbour indices to flat CSR (row, col) index arrays.

    Parameters
    ----------
    row_indices, col_indices : ndarray, shape (width, height, K, K)
    mask : ndarray of bool, shape (width * height * K * K,)
        Boundary mask from ``generate_mask``.
    width, height : int

    Returns
    -------
    row, col : ndarray of int
        Flattened source-pixel and neighbour indices for CSR construction,
        with out-of-bounds entries removed.
    """
    max_kernel_size = globals.MAX_KERNEL_SIZE

    flattened_indices = row_indices * height + col_indices

    row = np.arange(width * height, dtype=np.intp)
    row = np.expand_dims(row, 1)
    row = np.tile(row, (1, max_kernel_size * max_kernel_size))
    row = row.flatten()
    row = row[mask].astype(np.intp)

    col = flattened_indices.flatten()
    col = col[mask].astype(np.int32)

    return row, col


def precompute_indices(width, height):
    """Build all index structures needed to assemble blur matrices.

    Combines ``compute_u_v``, ``compute_shifted_indices``, ``generate_mask``,
    and ``compute_mask_flattened_indices`` into a single call.  The result can
    be cached and reused for any depth map of the same spatial dimensions.

    Parameters
    ----------
    width, height : int
        Image dimensions.

    Returns
    -------
    u, v : ndarray
        Kernel coordinate grids (see ``compute_u_v``).
    row, col : ndarray of int
        Flat CSR index arrays (see ``compute_mask_flattened_indices``).
    mask : ndarray of bool
        Boundary mask (see ``generate_mask``).
    """
    u, v = compute_u_v()
    row_indices, col_indices = compute_shifted_indices(width, height)
    mask = generate_mask(row_indices, col_indices, width, height)
    row, col = compute_mask_flattened_indices(row_indices, col_indices, mask, width, height)

    return u, v, row, col, mask


# ---------------------------------------------------------------------------
# CoC and kernel computation
# ---------------------------------------------------------------------------

def computer(dpt, Df):
    """Compute the per-pixel blur radius (in pixels) for each focal plane.

    Uses the thin-lens circle-of-confusion formula::

        CoC = D * |z - Df| / z * f / (Df - f)
        r   = CoC / 2 / ps

    where ``D``, ``f``, ``ps`` are camera parameters from ``globals``.
    Radii below ``globals.thresh`` are clamped to that threshold.

    Parameters
    ----------
    dpt : ndarray, shape (width, height)
        Depth map in metres.
    Df : array-like, shape (fs,)
        Focus distances in metres for each focal plane.

    Returns
    -------
    r : ndarray, shape (width, height, fs)
        Blur radius in pixels for every pixel and focal plane.
    """
    # format focus setting
    if not isinstance(Df, np.ndarray):
        Df = Df.numpy().astype(np.float32)
    Df_expanded = Df.reshape(1, 1, -1)
    # compute CoC
    CoC = ((globals.D)
        * (np.abs(dpt[..., None] - Df_expanded) / (dpt[..., None] + 1e-8))
        * (globals.f / (Df_expanded - globals.f)))
    r = CoC / 2. / globals.ps

    # threshold
    r[np.where(r < globals.thresh)] = globals.thresh

    return r


def computeG(r, u, v, eps=1e-8):
    """Compute normalised 2-D Gaussian blur kernels for each pixel and focal plane.

    Parameters
    ----------
    r : ndarray, shape (width, height, fs, 1, 1)
        Per-pixel, per-focal-plane blur radius (from ``computer``).
    u, v : ndarray, shape (1, 1, 1, K, K)
        Kernel coordinate grids (from ``compute_u_v``).

    Returns
    -------
    G : ndarray, shape (width, height, fs, K, K)
        Normalised Gaussian kernels.
    norm : ndarray
        Per-kernel normalisation factors (sum before division).
    """
    # compute Gaussian kernels
    G = np.exp(-(u**2 + v**2) / (2 * (r+eps)**2))
    # G = ((u**2 + v**2) <= r**2).astype(np.float32) # disc

    # normalize gaussian kernels
    norm = np.sum(G, axis=(-2, -1), keepdims=True, dtype=np.float32)
    G /= (norm + eps)

    return G, norm

# def computeG_faster(r, u, v, eps=1e-8):
#     # exploit Gaussian separability for speed up
#     u2 = (u[..., :, :1] ** 2).astype(np.float32)
#     v2 = (v[..., :1, :] ** 2).astype(np.float32)
#     inv2sigma2 = 1 / (2 * (r+eps)**2)
#
#     # 1D Gaussians (broadcasted)
#     Gu = np.exp(-u2 * inv2sigma2)
#     Gv = np.exp(-v2 * inv2sigma2)
#
#     # assemble full 2D kernel
#     G = Gu * Gv
#
#     # normalize gaussian kernels
#     norm = np.sum(G, axis=(-2,-1), keepdims=True, dtype=np.float32)
#     G /= (norm+1e-8)
#
#     return G, norm


# ---------------------------------------------------------------------------
# Sparse matrix construction
# ---------------------------------------------------------------------------

def build_fixed_pattern_csr(width, height, fs, row, col, data, dtype=np.float32):
    """Build a stack of CSR matrices sharing the same sparsity pattern.

    Sorting ``(row, col)`` pairs once and reusing the ``indptr`` / ``indices``
    arrays is significantly faster than constructing each matrix from COO
    triplets when only the ``data`` values change (see ``buildA`` with a
    ``template_A_stack``).

    Parameters
    ----------
    width, height : int
        Image dimensions; the matrix shape is ``(width*height, width*height)``.
    fs : int
        Number of focal planes (stack depth).
    row, col : ndarray of int
        COO row and column indices.
    data : ndarray
        Non-zero values (shared initial fill for all ``fs`` matrices).
    dtype : dtype, optional
        Storage dtype for the sparse matrices.

    Returns
    -------
    A_stack : list of scipy.sparse.csr_matrix, length fs
    order : ndarray of int
        Permutation that sorts ``(row, col)`` into CSR order; needed to
        update ``A.data`` in place when reusing the template.
    """
    row = np.asarray(row, dtype=np.int32)
    col = np.asarray(col, dtype=np.int32)
    data = np.asarray(data, dtype=dtype)

    order = np.lexsort((col, row))
    row_sorted, col_sorted = row[order], col[order]

    # csr structure ensures it stays in original row, col order
    indptr = np.zeros(width * height + 1, dtype=np.int32)
    np.add.at(indptr, row_sorted + 1, 1)
    np.cumsum(indptr, out=indptr)
    indices = col_sorted.astype(np.int32, copy=False)

    # build matrices
    A_stack = []
    for idx in range(fs):
        A = scipy.sparse.csr_matrix((data, indices, indptr),
                shape=(width*height, width*height), dtype=data.dtype, copy=False)
        A_stack.append(A)

    return A_stack, order


def buildA(dpt, u, v, row, col, mask, template_A_stack=None):
    """Assemble the stack of per-focal-plane sparse blur operators.

    Parameters
    ----------
    dpt : ndarray, shape (width, height)
        Depth map in metres.
    u, v : ndarray
        Kernel coordinate grids from ``precompute_indices``.
    row, col : ndarray of int
        Flat CSR index arrays from ``precompute_indices``.
    mask : ndarray of bool
        Boundary mask from ``precompute_indices``.
    template_A_stack : tuple (A_stack_cache, order), optional
        Pre-built CSR structure from a previous ``buildA`` call.  When
        provided, only the ``.data`` arrays are updated in place, which is
        significantly faster than constructing new matrices from scratch.

    Returns
    -------
    A_stack : list of scipy.sparse.csr_matrix, length fs
        Sparse blur operator for each focal plane.
    """
    width, height = dpt.shape
    r = computer(dpt, globals.Df)
    _, _, fs = r.shape
    r = r[..., None, None]

    G, _ = computeG(r, u, v)

    A_stack = []
    if template_A_stack is not None:
        A_stack_cache, order = template_A_stack

    for idx in range(fs):
        data = G[:, :, idx, :, :]
        data = data.flatten()
        data = data[mask]

        if template_A_stack is None:
            # warning -- this is > 3x slower
            A = scipy.sparse.csr_matrix((data, (row, col)),
                shape=(width*height, width*height), dtype=data.dtype)
            A_stack.append(A)
        else:
            A = A_stack_cache[idx].copy()
            A.data[:] = data[order]
            A_stack.append(A)

    return A_stack


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward(dpt, aif, indices=None, template_A_stack=None):
    """Apply the forward model: AIF image -> defocus stack.

    Computes ``b_i = A_i @ x`` for each focal plane *i*, where ``x`` is the
    vectorised AIF and ``A_i`` is the blur operator from ``buildA``.

    Parameters
    ----------
    dpt : ndarray, shape (width, height)
        Depth map in metres.
    aif : ndarray, shape (width, height, 3)
        All-in-focus RGB image.
    indices : tuple, optional
        Precomputed index structures from ``precompute_indices``.  Pass this
        to avoid recomputing when calling ``forward`` repeatedly.
    template_A_stack : tuple, optional
        Cached CSR structure for fast matrix updates (see ``buildA``).

    Returns
    -------
    defocus_stack : ndarray, shape (fs, width, height, 3)
        Simulated defocus stack for each focal plane.
    """
    width, height = dpt.shape

    if indices is None:
        u, v, row, col, mask = precompute_indices(width, height)
    else:
        u, v, row, col, mask = indices

    A_stack = buildA(dpt, u, v, row, col, mask, template_A_stack=template_A_stack)

    defocus_stack = []

    aif_red = aif[:, :, 0].flatten()
    aif_green = aif[:, :, 1].flatten()
    aif_blue = aif[:, :, 2].flatten()

    for idx in range(len(A_stack)):
        A = A_stack[idx]

        b_red = A @ aif_red
        b_green = A @ aif_green
        b_blue = A @ aif_blue
        b = np.column_stack((b_red, b_green, b_blue))

        b = b.reshape((width, height, 3))

        defocus_stack.append(b)

    return np.stack(defocus_stack, 0)