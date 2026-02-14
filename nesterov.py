"""
Bounded FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for AIF estimation.

Given the current depth map estimate, solves the box-constrained least-squares
problem:

    min  0.5 * ||A(Z) x - b||²      s.t.  0 ≤ x ≤ IMAGE_RANGE

where A(Z) is the depth-dependent sparse blur operator (built in forward_model),
x is the vectorised all-in-focus image, and b is the observed focal stack.

Nesterov momentum (FISTA sequence) is used for acceleration.  The step size is
set to 1/L where L ≈ ||A||² is estimated via power iteration.
"""

import numpy as np
import scipy

from PIL import Image
import tqdm

import forward_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def buildb(defocus_stack):
    """Flatten a focal stack into per-channel observation vectors.

    Parameters
    ----------
    defocus_stack : list of ndarray, each shape (H, W, 3)
        Observed defocused images for each focal setting.

    Returns
    -------
    b_red_stack, b_green_stack, b_blue_stack : lists of 1-D ndarray
        Flattened pixel vectors for each focal image and each colour channel.
    """
    b_red_stack = []
    b_green_stack = []
    b_blue_stack = []
    for idx in range(len(defocus_stack)):        
        b_red = defocus_stack[idx][:,:,0].flatten()
        b_green = defocus_stack[idx][:,:,1].flatten()        
        b_blue = defocus_stack[idx][:,:,2].flatten()

        b_red_stack.append(b_red)
        b_green_stack.append(b_green)
        b_blue_stack.append(b_blue)
    return b_red_stack, b_green_stack, b_blue_stack     

def compute_Lipschitz_constant(A):
    """Compute ||A||² exactly via a sparse matrix 2-norm (slower than power iteration).

    Prefer ``approx_Lipschitz_constant`` for large systems; use this when an
    exact step size is needed.
    """
    norm = scipy.sparse.linalg.norm(A, ord=2)
    return norm**2

def approx_Lipschitz_constant(A, A_T, iters=15):
    """Estimate ||A||² via power iteration (used as the FISTA step-size denominator).

    Parameters
    ----------
    A : sparse matrix, shape (m, n)
        The stacked blur operator.
    A_T : sparse matrix, shape (n, m)
        Transpose of A (precomputed for efficiency).
    iters : int
        Number of power iterations; 15 is typically sufficient.

    Returns
    -------
    float32
        Approximate largest eigenvalue of A^T A, i.e. ||A||².
    """
    n = A.shape[1]
    x = np.random.standard_normal(n).astype(np.float32, copy=False)
    x /= (np.linalg.norm(x) + 1e-8)

    for _ in range(iters):
        y = A.dot(x)
        z = A_T.dot(y)
        nz = np.linalg.norm(z)
        if nz == 0:
            return 1.0
        x = z / nz

    y = A.dot(x)
    return np.float32(np.dot(y, y))  # Rayleigh quotient: x^T A^T A x ≈ ||A||²


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def bounded_fista_3d(dpt, defocus_stack, dataset_params, max_kernel_size,
                     indices=None, template_A_stack=None,
                     tol=1e-6, maxiter=1000, verbose=True):
    """Solve the bounded least-squares AIF sub-problem via FISTA.

    Parameters
    ----------
    dpt : ndarray, shape (H, W)
        Current depth map estimate (held fixed during this call).
    defocus_stack : list of ndarray, each shape (H, W, 3)
        Observed focal stack.
    dataset_params : DatasetParams
        Camera/scene parameters (passed to ``forward_model``).
    max_kernel_size : int
        Side length of the square kernel window (must be odd).
    indices : tuple or None
        Pre-computed sparse index arrays from ``forward_model.precompute_indices``.
        Computed from *dpt* if not provided.
    template_A_stack : list or None
        CSR template for ``forward_model.buildA``; avoids re-allocating the
        sparsity pattern each call.
    tol : float
        Convergence criterion on ||x_new - x_old||.
    maxiter : int
        Maximum number of FISTA iterations.
    verbose : bool
        Print progress and final residual norms.

    Returns
    -------
    aif : ndarray, shape (H, W, 3)
        Reconstructed all-in-focus image, clipped to [0, IMAGE_RANGE].
    """
    if verbose:
        print('Bounded FISTA...')

    width, height = dpt.shape
    if indices is None:
        u, v, row, col, mask = forward_model.precompute_indices(width, height, max_kernel_size)
    else:
        u, v, row, col, mask = indices

    A_stack = forward_model.buildA(dpt, u, v, row, col, mask, dataset_params,
                                   template_A_stack=template_A_stack)
    b_red_stack, b_green_stack, b_blue_stack = buildb(defocus_stack)

    IMAGE_RANGE = 255.  # default assumes [0-255] range
    if defocus_stack.max() <= 1.5:  # check if images are normalized to [0-1]
        IMAGE_RANGE = 1.

    # Stack all focal images into one tall sparse system A x = b
    A = scipy.sparse.vstack(A_stack).tocsr(copy=False)
    A.sort_indices()  # sorting indices speeds up repeated sparse mat-vec products
    assert A.dtype == np.float32, "float64 promotion would seriously slow down sparse mat-vec throughput"

    A_T = A.T

    b_red = np.concatenate(b_red_stack)
    b_green = np.concatenate(b_green_stack)
    b_blue = np.concatenate(b_blue_stack)
    b = np.stack([b_red, b_green, b_blue], axis=1).astype(np.float32, copy=False)

    # Step size = 1/L where L = ||A||² (Lipschitz constant of the gradient).
    # approx_Lipschitz_constant uses power iteration and is much faster for large A;
    # swap in compute_Lipschitz_constant for an exact value when debugging convergence.
    # L = compute_Lipschitz_constant(A)
    L = approx_Lipschitz_constant(A, A_T)
    eta = 1.0 / L

    # Initialise primal variable x and momentum auxiliary variable y
    aif = np.zeros((width * height, 3), dtype=np.float32)
    aif_guess = aif.copy()

    # Pre-compute A * y for the first iteration (updated at end of each loop)
    Ay0 = A.dot(aif_guess[:, 0])
    Ay1 = A.dot(aif_guess[:, 1])
    Ay2 = A.dot(aif_guess[:, 2])
    Ay = np.column_stack((Ay0, Ay1, Ay2))

    t = 1.0  # FISTA momentum sequence parameter

    progress = tqdm.trange(maxiter, desc="Optimizing", leave=True, disable=(not verbose))
    for i in progress:
        # Gradient of 0.5 ||Ay - b||² w.r.t. y
        r = Ay - b
        g0 = A_T.dot(r[:, 0])
        g1 = A_T.dot(r[:, 1])
        g2 = A_T.dot(r[:, 2])
        grad = np.column_stack((g0, g1, g2))

        # Projected gradient step: gradient descent followed by projection onto [0, IMAGE_RANGE]
        aif_new = np.clip(aif_guess - eta * grad, 0, IMAGE_RANGE)

        # FISTA momentum update: t_{k+1} = (1 + sqrt(1 + 4 t_k²)) / 2
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        aif_guess = aif_new + ((t - 1) / t_new) * (aif_new - aif)

        # Convergence check on the primal update
        if np.linalg.norm(aif_new - aif) < tol:
            if verbose:
                print('Achieved tolerance')
            break

        aif = aif_new
        t = t_new

        # Update A * y for next iteration
        Ay0 = A.dot(aif_guess[:, 0])
        Ay1 = A.dot(aif_guess[:, 1])
        Ay2 = A.dot(aif_guess[:, 2])
        Ay = np.column_stack((Ay0, Ay1, Ay2))

    if verbose:
        print('r1norm', np.linalg.norm(r), 'norm(x)', np.linalg.norm(aif))

    return aif.reshape((width, height, 3))