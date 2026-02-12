"""
Outlier removal for depth maps estimated via depth-from-defocus.

Two detection strategies are supported:
  - 'tv'       : flags pixels in high total-variation regions of the depth map
                 (edges / sharp depth discontinuities where DfD is unreliable)
  - 'constant' : flags pixels in near-constant-colour patches of the all-in-focus
                 image (textureless regions where DfD has no signal)

Detected outlier pixels are replaced by the mean depth of their non-outlier
neighbours within a window of radius MAX_KERNEL_SIZE // 2.
"""

import numpy as np
import matplotlib.pyplot as plt

import globals


# ---------------------------------------------------------------------------
# TV-based detection
# ---------------------------------------------------------------------------

def total_variation(image):
    """Return the scalar total variation of *image* (sum of absolute gradients)."""
    tv_x = np.abs(image[:, 1:] - image[:, :-1])
    tv_y = np.abs(image[1:, :] - image[:-1, :])
    return np.sum(tv_x) + np.sum(tv_y)


def compute_tv_map(image, patch_size=None):
    """Compute a per-pixel total-variation map using local patches.

    Parameters
    ----------
    image : ndarray, shape (H, W) or (H, W, C)
        Input image (e.g. a depth map).
    patch_size : int or None
        Side length of the square patch (must be odd).
        Defaults to ``globals.MAX_KERNEL_SIZE``.

    Returns
    -------
    tv_map : ndarray, shape (H, W)
        TV value normalised by patch area at each pixel.
        Border pixels within *pad* of the edge are left as zero.
    """
    if patch_size is not None:
        assert patch_size % 2 != 0, "patch_size must be odd"
    if patch_size is None:
        patch_size = globals.MAX_KERNEL_SIZE

    pad = patch_size // 2
    width, height = image.shape[:2]

    tv_map = np.zeros((width, height))
    for i in range(pad, width - pad):
        for j in range(pad, height - pad):
            window = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            tv_map[i, j] = total_variation(window) / patch_size**2

    return tv_map


def find_high_tv_patches(dpt, tv_thresh=0.15, patch_size=None):
    """Return pixels whose local TV exceeds *tv_thresh* in the depth map *dpt*.

    Parameters
    ----------
    dpt : ndarray, shape (H, W)
        Depth map to analyse.
    tv_thresh : float
        Normalised TV threshold above which a pixel is flagged.
    patch_size : int or None
        Passed through to :func:`compute_tv_map`.

    Returns
    -------
    problem_pixels : ndarray, shape (N, 2)
        (row, col) indices of flagged pixels.
    tv_map : ndarray, shape (H, W)
        Full TV map (useful for visualisation / threshold tuning).
    """
    tv_map = compute_tv_map(dpt, patch_size=patch_size)
    problem_pixels = np.argwhere(tv_map > tv_thresh)
    return problem_pixels, tv_map


# ---------------------------------------------------------------------------
# Constant-patch detection
# ---------------------------------------------------------------------------

def find_constant_patches(aif, diff_thresh=2, patch_size=None):
    """Return pixels that lie inside near-constant-colour patches of *aif*.

    Textureless regions of the all-in-focus image provide no defocus cue, so
    depth estimates there are unreliable.

    Parameters
    ----------
    aif : ndarray, shape (H, W, 3)
        All-in-focus RGB image.
    diff_thresh : int
        Maximum per-channel (max - min) range within a patch for it to be
        considered constant.  Units are raw pixel values [0, 255].
    patch_size : int or None
        Side length of the square patch (must be odd).
        Defaults to ``globals.MAX_KERNEL_SIZE``.

    Returns
    -------
    problem_pixels : ndarray, shape (N, 2)
        (row, col) indices of flagged pixels.
    """
    if patch_size is not None:
        assert patch_size % 2 != 0, "patch_size must be odd"
    if patch_size is None:
        patch_size = globals.MAX_KERNEL_SIZE
    rad = patch_size // 2

    padded_aif = np.pad(aif, ((rad, rad), (rad, rad), (0, 0)), mode='edge')
    patches = np.lib.stride_tricks.sliding_window_view(padded_aif, (patch_size, patch_size, 3))
    patches = np.squeeze(patches)

    max_vals = patches.max(axis=(2, 3))
    min_vals = patches.min(axis=(2, 3))
    color_diffs = max_vals - min_vals

    mask = np.all(color_diffs <= diff_thresh, axis=2)
    problem_pixels = np.argwhere(mask)

    return problem_pixels


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def remove_outliers(depth_map, gt_aif, patch_type='tv', diff_thresh=2, tv_thresh=0.15, to_plot=True):
    """Detect and replace outlier pixels in *depth_map*.

    Outliers are identified by one of two strategies (see module docstring) and
    then replaced by the mean depth of their non-outlier neighbours within a
    neighbourhood of radius ``MAX_KERNEL_SIZE // 2``.

    Parameters
    ----------
    depth_map : ndarray, shape (H, W) or (H, W, C)
        Depth map to clean.  Modified in-place.
    gt_aif : ndarray, shape (H, W, 3)
        All-in-focus image used by the ``'constant'`` strategy and for plotting.
    patch_type : {'tv', 'constant'}
        Which detection strategy to use.
    diff_thresh : int
        Per-channel range threshold for the ``'constant'`` strategy.
    tv_thresh : float
        TV threshold for the ``'tv'`` strategy.
    to_plot : bool
        If True, display the outlier map before removal.

    Returns
    -------
    depth_map : ndarray
        Cleaned depth map (same array, modified in-place).
    outlier_fraction : float
        Fraction of pixels flagged as outliers (for logging / diagnostics).
    """
    assert patch_type in ['tv', 'constant']
    print("Removing outliers...")

    if patch_type == 'constant':
        problem_pixels = find_constant_patches(gt_aif, diff_thresh=diff_thresh)
    else:
        problem_pixels, tv_map = find_high_tv_patches(depth_map, tv_thresh=tv_thresh)

    problem_pixel_set = set(map(tuple, problem_pixels))
    print('found', len(problem_pixel_set), 'outliers')

    if to_plot:
        if patch_type == 'constant':
            plt.imshow(gt_aif / 255.)
        else:
            plt.imshow(tv_map)
            plt.colorbar()
        plt.scatter([y for x, y in problem_pixels], [x for x, y in problem_pixels],
            color='red', marker='x', s=50)
        plt.title("Outliers (" + patch_type + ')')
        plt.show()

    removed = 0
    # Radius of the replacement neighbourhood, derived from the maximum kernel size
    neighborhood_rad = int((float(globals.MAX_KERNEL_SIZE) - 1) / 2.)
    for i, j in problem_pixels:
        patch = []
        for dx in range(-neighborhood_rad, neighborhood_rad + 1):
            for dy in range(-neighborhood_rad, neighborhood_rad + 1):
                if i + dx < 0 or i + dx >= gt_aif.shape[0]:
                    continue
                if j + dy < 0 or j + dy >= gt_aif.shape[1]:
                    continue
                if (i + dx, j + dy) in problem_pixel_set:
                    continue
                if dx == 0 and dy == 0:
                    continue
                patch.append(depth_map[i + dx, j + dy])
        if len(patch) != 0:
            removed += 1
            avg_depth = np.array(patch).mean(axis=0, keepdims=True)
            depth_map[i, j] = avg_depth
            problem_pixel_set.remove((i, j))  # remove so neighbour lookups can use it

    print(removed, '/', len(problem_pixels), 'outliers removed')

    return depth_map, (len(problem_pixels) / (depth_map.shape[0] * depth_map.shape[1]))