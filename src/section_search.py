"""Objective functions and depth-search routines for depth-from-defocus.

Provides two search strategies:

Grid search
    Evaluates the objective at a dense grid of depth values and returns
    the per-pixel minimizer.  Intended as a coarse initialization.

Golden-section search
    Refines an initial depth estimate using bracketed golden-section search,
    exploiting the objective's approximately unimodal structure per pixel.
"""
import math

import numpy as np
import scipy
import tqdm

import forward_model

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi


# ---------------------------------------------------------------------------
# Windowed-MSE helpers
# ---------------------------------------------------------------------------

def windowed_mse_gss(depth_map, gt_aif, defocus_stack, dataset_params, max_kernel_size,
                     window_size, indices=None, template_A_stack=None):
    """Windowed MSE for arbitrary (non-constant) depth maps.

    For each spatial offset (i, j) within a square window of side
    *window_size*, shifts the depth map by (i, j), runs the full
    forward model on the shifted map, and accumulates the per-pixel MSE.
    The final loss at each pixel is the average over all valid offsets,
    smoothing the loss landscape across spatially-neighbouring depth values.

    Requires one full forward-model pass per offset — O(window_size²) total.
    Not used in practice; prefer windowed_mse_grid for the grid-search path,
    where the forward pass is already precomputed.
    """
    rad = window_size // 2
    _, width, height, _ = defocus_stack.shape

    losses = np.zeros((width, height), dtype=np.float32)
    denom = np.zeros((width, height), dtype=np.float32)
    for i in range(-rad, rad+1):
        x_shifted = np.roll(depth_map, shift=i, axis=0)
        for j in range(-rad, rad+1):
            shifted = np.roll(x_shifted, shift=j, axis=1)
            # compute mse
            pred = forward_model.forward(shifted, gt_aif, dataset_params, max_kernel_size,
                                         indices=indices, template_A_stack=template_A_stack)
            mse = np.mean((defocus_stack - pred)**2, axis=(0, -1))
            i_start = -i if i < 0 else 0
            i_end = width-i if i > 0 else width
            j_start = -j if j < 0 else 0
            j_end = height-j if j > 0 else height
            losses[i_start:i_end, j_start:j_end] += mse[i_start+i:i_end+i, j_start+j:j_end+j]
            denom[i_start:i_end, j_start:j_end] += 1
            
    return losses / denom

def windowed_mse_grid(defocus_stack, pred, window_size):
    """Windowed MSE for constant-depth (grid-search) evaluations.

    Given a precomputed forward-model prediction, spatially averages the
    per-pixel MSE within a square window of side *window_size*,
    smoothing the loss landscape across neighbouring pixels.
    """
    rad = window_size // 2
    _, width, height, _ = defocus_stack.shape
    mse = np.mean((defocus_stack - pred)**2, axis=(0, -1))
    losses = np.zeros((width, height), dtype=np.float32)
    denom = np.zeros((width, height), dtype=np.float32)
    row = np.arange(width)
    col = np.arange(height)
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            i_start = -i if i < 0 else 0
            i_end = width-i if i > 0 else width
            j_start = -j if j < 0 else 0
            j_end = height-j if j > 0 else height
            losses[i_start:i_end, j_start:j_end] += mse[i_start+i:i_end+i, j_start+j:j_end+j]
            denom[i_start:i_end, j_start:j_end] += 1
    return losses / denom

def windowed_mse_grid_fast(defocus_stack, pred, window_size):
    """Windowed MSE using scipy uniform_filter (faster alternative to windowed_mse_grid).

    Replaces the explicit accumulation loop with scipy.ndimage.uniform_filter,
    which applies a box-filter average over *window_size* pixels.
    Intended to be equivalent to windowed_mse_grid but has not yet been
    validated against it — use with caution.
    """
    # TODO: verify output matches windowed_mse_grid before using in production
    mse = np.mean((defocus_stack - pred)**2, axis=(0, -1))
    win_mean = scipy.ndimage.uniform_filter(mse, size=window_size, mode='nearest')
    return win_mean

# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective_full(dpt, aif, defocus_stack, dataset_params, max_kernel_size,
                   window_size=None, indices=None, template_A_stack=None,
                   pred=None, windowed=False):
    """Per-pixel reconstruction loss between the observed and predicted focal stacks.

    Runs the forward model for the given depth map and computes the mean
    squared error against defocus_stack.  When windowed=True, replaces plain
    MSE with a spatially-averaged windowed loss.

    If dpt is constant (all pixels share the same depth value), the cheaper
    windowed_mse_grid path is taken; otherwise windowed_mse_gss is used,
    which re-runs the forward model for each spatial offset.

    Parameters
    ----------
    dpt : ndarray, shape (W, H)
        Candidate depth map in metres.
    aif : ndarray, shape (W, H, C)
        All-in-focus reference image.
    defocus_stack : ndarray, shape (fs, W, H, C)
        Observed focal stack.
    dataset_params : DatasetParams
        Camera/scene parameters (passed to ``forward_model``).
    max_kernel_size : int
        Side length of the square kernel window (must be odd).
    window_size : int or None
        Side length of the spatial averaging window (must be odd).
        Required when *windowed* is True.
    indices : tuple, optional
        Precomputed (u, v, row, col, mask) from forward_model.precompute_indices.
    template_A_stack : tuple, optional
        Precomputed CSR template from forward_model.build_fixed_pattern_csr.
    pred : ndarray, shape (fs, W, H, C), optional
        If provided, skips the forward-model call and uses this prediction directly.
    windowed : bool
        If True, apply spatial windowing to the MSE.

    Returns
    -------
    ndarray, shape (W, H)
        Per-pixel reconstruction loss.
    """
    grid_search = np.all(np.isclose(dpt, dpt[0][0]))

    if windowed:
        if grid_search:
            if pred is None:
                pred = forward_model.forward(dpt, aif, dataset_params, max_kernel_size,
                                             indices=indices, template_A_stack=template_A_stack)
            loss = windowed_mse_grid(defocus_stack, pred, window_size)
        else:
            loss = windowed_mse_gss(dpt, aif, defocus_stack, dataset_params, max_kernel_size,
                                    window_size, indices=indices, template_A_stack=template_A_stack)
    else:
        if pred is None:
            pred = forward_model.forward(dpt, aif, dataset_params, max_kernel_size,
                                         indices=indices, template_A_stack=template_A_stack)
        loss = np.mean((defocus_stack - pred)**2, axis=(0, -1))
    return loss


# ---------------------------------------------------------------------------
# Depth search
# ---------------------------------------------------------------------------

def grid_search(gt_aif, defocus_stack, dataset_params, max_kernel_size,
                window_size=None, indices=None, min_Z=0.1, max_Z=10,
                num_Z=100, verbose=True, windowed=False):
    """Coarse per-pixel depth search over a uniform depth grid.

    Evaluates the reconstruction objective at num_Z candidate depths
    linearly spaced in [min_Z, max_Z] and returns the per-pixel depth with
    minimum loss, along with the full loss surface for downstream refinement
    (e.g. golden_section_search).

    Parameters
    ----------
    gt_aif : ndarray, shape (W, H, C)
        All-in-focus reference image.
    defocus_stack : ndarray, shape (fs, W, H, C)
        Observed focal stack.
    dataset_params : DatasetParams
        Camera/scene parameters.
    max_kernel_size : int
        Side length of the square kernel window (must be odd).
    window_size : int or None
        Side length of the spatial averaging window (must be odd).
        Required when *windowed* is True.
    indices : tuple, optional
        Precomputed (u, v, row, col, mask) from forward_model.precompute_indices.
    min_Z, max_Z : float
        Depth range to search.
    num_Z : int
        Number of candidate depths.
    verbose : bool
        Show a tqdm progress bar.
    windowed : bool
        Use spatially-windowed MSE instead of plain MSE.

    Returns
    -------
    depth_maps : ndarray, shape (W, H)
        Per-pixel depth at the grid minimum.
    Z : ndarray, shape (num_Z,)
        The candidate depth values.
    min_indices : ndarray, shape (W, H), int
        Index into Z of the per-pixel minimum.
    all_losses : ndarray, shape (W, H, num_Z)
        Full per-pixel loss surface over the depth grid.
    """
    Z = np.linspace(min_Z, max_Z, num_Z, dtype=np.float32)

    width, height, num_channels = gt_aif.shape
    if indices is None:
        u, v = forward_model.compute_u_v(max_kernel_size)
    else:
        u, v, _, _, _ = indices

    all_losses = np.zeros((width, height, num_Z), dtype=np.float32)
    for i in tqdm.tqdm(range(num_Z), desc="Grid search".ljust(20), ncols=80, disable=(not verbose)):
        r = forward_model.computer(np.array([[Z[i]]], dtype=np.float32), dataset_params)[...,None,None]
        G, _ = forward_model.computeG(r, u, v)
        G = G.squeeze()
        defocus_stack_pred = np.zeros((G.shape[0], width, height, num_channels), dtype=np.float32)
        for j in range(G.shape[0]): # each focal setting
            kernel = G[j]
            for c in range(num_channels):
                defocus_stack_pred[j,:,:,c] = scipy.ndimage.convolve(gt_aif[:,:,c], kernel, mode='constant')

        dpt = np.ones((width, height), dtype=np.float32) * Z[i]
        all_losses[:,:,i] = objective_full(dpt, gt_aif, defocus_stack, dataset_params, max_kernel_size,
                                           window_size=window_size, indices=indices,
                                           pred=defocus_stack_pred, windowed=windowed)

    sorted_indices = np.argsort(all_losses, axis=2)
    min_indices = sorted_indices[:,:,0]
    depth_maps = Z[min_indices]

    return depth_maps, Z, min_indices, all_losses


def golden_section_search(Z, argmin_indices, gt_aif, defocus_stack,
                          dataset_params, max_kernel_size,
                          window_size=None, indices=None, template_A_stack=None,
                          window=1, tolerance=1e-5, convergence_error=0,
                          max_iter=100, last_dpt=None, a_b_init=None,
                          verbose=True, windowed=False):
    """Per-pixel depth refinement via bracketed golden-section search.

    Starting from a bracket [a, b] centred on the grid-search minimum
    (argmin_indices ± window steps in Z), narrows each pixel's bracket using
    the golden-section rule until the interval width falls below tolerance or
    max_iter iterations are reached.

    Supports partial convergence: convergence_error is the fraction of pixels
    allowed to remain unconverged (e.g. 0.01 → 99 % of pixels must converge).

    If last_dpt is provided, the returned depth map takes the per-pixel minimum
    of the GSS result and last_dpt, keeping whichever achieves lower loss.

    Parameters
    ----------
    Z : ndarray, shape (num_Z,)
        Depth grid from grid_search.
    argmin_indices : ndarray, shape (W, H), int
        Per-pixel index of the grid minimum, used to initialise the bracket.
    gt_aif : ndarray, shape (W, H, C)
        All-in-focus reference image.
    defocus_stack : ndarray, shape (fs, W, H, C)
        Observed focal stack.
    dataset_params : DatasetParams
        Camera/scene parameters.
    max_kernel_size : int
        Side length of the square kernel window (must be odd).
    window_size : int or None
        Side length of the spatial averaging window (must be odd).
        Required when *windowed* is True.
    indices : tuple, optional
        Precomputed (u, v, row, col, mask) from forward_model.precompute_indices.
    template_A_stack : tuple, optional
        Precomputed CSR template from forward_model.build_fixed_pattern_csr.
    window : int
        Half-width (in grid steps) of the initial search bracket.
    tolerance : float
        Stop when |b - a| < tolerance for all (or convergence_error fraction) pixels.
    convergence_error : float in [0, 1)
        Fraction of pixels allowed to remain unconverged.  0 requires all pixels
        to converge.
    max_iter : int
        Maximum number of GSS iterations.
    last_dpt : ndarray, shape (W, H), optional
        Previous depth estimate.  If provided, the output takes the per-pixel
        minimum of the GSS result and last_dpt by objective value.
    a_b_init : tuple of (ndarray, ndarray), optional
        Explicit (a, b) bracket arrays, overriding the grid-based initialisation.
    verbose : bool
        Print progress and convergence messages.
    windowed : bool
        Use spatially-windowed MSE instead of plain MSE.

    Returns
    -------
    ndarray, shape (W, H)
        Refined per-pixel depth map.
    """
    _obj = lambda d: objective_full(
        d, gt_aif, defocus_stack, dataset_params, max_kernel_size,
        window_size=window_size, indices=indices,
        template_A_stack=template_A_stack, windowed=windowed)

    assert convergence_error >= 0 and convergence_error < 1
    if verbose:
        print("\nGolden-section search...")

    # build a grid around each min
    if a_b_init is None:
        num_Z = len(Z)
        a = Z[np.maximum(argmin_indices-window,0)]
        b = Z[np.minimum(argmin_indices+window,num_Z-1)]
    else:
        a, b = a_b_init

    if verbose:
        print('...searching for',(1 - convergence_error)*100,'% convergence')

    c = b - (b - a) * invphi
    d = a + (b - a) * invphi

    f_c = _obj(c)
    f_d = _obj(d)

    i = 0
    while (((convergence_error == 0 and np.any(b - a > tolerance))
            or (convergence_error != 0 and np.sum((b - a) > tolerance) / a.size > (1 - convergence_error)))
            and (i < max_iter)):
        # tie-safe implementation
        active = (b - a) > tolerance
        go_left = (f_c <= f_d) & active
        go_right = (~go_left) & active

        if np.any(go_left):
            b[go_left] = d[go_left]
            d[go_left] = c[go_left]
            f_d[go_left] = f_c[go_left]
            c[go_left] = b[go_left] - (b[go_left] - a[go_left]) * invphi

        if np.any(go_right):
            a[go_right] = c[go_right]
            c[go_right] = d[go_right]
            f_c[go_right] = f_d[go_right]
            d[go_right] = a[go_right] + (b[go_right] - a[go_right]) * invphi

        if np.any(go_left):
            f_c = _obj(c)

        if np.any(go_right):
            f_d = _obj(d)

        i += 1

    if (i >= max_iter) and verbose:
        print('Failed to converge after',i,'iterations')
        print(np.sum((b - a) <= tolerance) / a.size * 100, '% convergence achieved')

    if verbose:
        print("...done")

    dpt = (b + a) / 2

    if last_dpt is not None:
        mse = _obj(dpt)
        last_mse = _obj(last_dpt)
        dpt = np.where(mse <= last_mse, dpt, last_dpt)

    return dpt
