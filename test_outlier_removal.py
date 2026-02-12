import numpy as np
import globals
import outlier_removal

# Set the one global this module needs
globals.MAX_KERNEL_SIZE = 7


# ---------------------------------------------------------------------------
# total_variation
# ---------------------------------------------------------------------------

def test_total_variation_constant():
    img = np.ones((10, 10)) * 5.0
    assert outlier_removal.total_variation(img) == 0.0


def test_total_variation_known():
    # Horizontal ramp: each row is [0, 1, 2], no vertical differences
    img = np.tile(np.array([0.0, 1.0, 2.0]), (3, 1))
    # tv_x: 3 rows * 2 steps of 1 = 6; tv_y: 0
    assert outlier_removal.total_variation(img) == 6.0


# ---------------------------------------------------------------------------
# compute_tv_map
# ---------------------------------------------------------------------------

def test_tv_map_constant_image():
    img = np.full((20, 20), 3.0)
    tv_map = outlier_removal.compute_tv_map(img, patch_size=3)
    assert tv_map.shape == (20, 20)
    assert np.all(tv_map == 0.0)


def test_tv_map_odd_patch_required():
    img = np.ones((10, 10))
    try:
        outlier_removal.compute_tv_map(img, patch_size=4)
        assert False, "Should have raised AssertionError for even patch_size"
    except AssertionError:
        pass


def test_tv_map_shape():
    img = np.random.default_rng(1).random((17, 23))
    tv_map = outlier_removal.compute_tv_map(img, patch_size=3)
    assert tv_map.shape == img.shape


# ---------------------------------------------------------------------------
# find_high_tv_patches
# ---------------------------------------------------------------------------

def test_find_high_tv_no_outliers():
    dpt = np.ones((20, 20)) * 2.0   # constant depth → zero TV everywhere
    pixels, tv_map = outlier_removal.find_high_tv_patches(dpt, tv_thresh=0.01, patch_size=3)
    assert len(pixels) == 0
    assert tv_map.shape == dpt.shape


def test_find_high_tv_known_spike():
    dpt = np.ones((15, 15)) * 2.0
    dpt[7, 7] = 100.0               # spike raises TV in surrounding patches
    pixels, _ = outlier_removal.find_high_tv_patches(dpt, tv_thresh=0.5, patch_size=3)
    flagged = set(map(tuple, pixels))
    assert (7, 7) in flagged


def test_find_high_tv_returns_2d_indices():
    dpt = np.random.default_rng(2).random((15, 15))
    pixels, _ = outlier_removal.find_high_tv_patches(dpt, tv_thresh=0.01, patch_size=3)
    if len(pixels) > 0:
        assert pixels.shape[1] == 2


# ---------------------------------------------------------------------------
# find_constant_patches
# ---------------------------------------------------------------------------

def test_constant_patches_all_flat():
    aif = np.full((15, 15, 3), 128, dtype=np.uint8)
    pixels = outlier_removal.find_constant_patches(aif, diff_thresh=2, patch_size=3)
    assert len(pixels) == 15 * 15   # every pixel sits inside a constant patch


def test_constant_patches_high_variance():
    rng = np.random.default_rng(3)
    # Force large per-channel range by placing 0 and 255 in every patch
    aif = rng.integers(50, 200, (20, 20, 3), dtype=np.uint8)
    aif[0, :, :] = 0
    aif[-1, :, :] = 255
    aif[:, 0, :] = 0
    aif[:, -1, :] = 255
    pixels = outlier_removal.find_constant_patches(aif, diff_thresh=2, patch_size=3)
    assert len(pixels) == 0


def test_constant_patches_odd_required():
    aif = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        outlier_removal.find_constant_patches(aif, patch_size=4)
        assert False, "Should have raised AssertionError for even patch_size"
    except AssertionError:
        pass


def test_constant_patches_returns_2d_indices():
    aif = np.full((10, 10, 3), 50, dtype=np.uint8)
    pixels = outlier_removal.find_constant_patches(aif, diff_thresh=2, patch_size=3)
    assert pixels.shape[1] == 2


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------

def test_remove_outliers_no_change_when_clean():
    rng = np.random.default_rng(4)
    depth = rng.uniform(1, 5, (20, 20)).astype(np.float32)
    # Checkerboard ensures every patch (any size) contains both 0 and 255,
    # so color_diffs = 255 > diff_thresh everywhere and nothing is flagged.
    aif = np.zeros((20, 20, 3), dtype=np.uint8)
    aif[::2, ::2, :] = 255
    depth_copy = depth.copy()
    result, frac = outlier_removal.remove_outliers(
        depth, aif, patch_type='constant', diff_thresh=2, to_plot=False
    )
    assert np.allclose(result, depth_copy)
    assert frac == 0.0


def test_remove_outliers_replaces_spike():
    depth = np.ones((15, 15), dtype=np.float32) * 2.0
    depth[7, 7] = 50.0              # isolated spike that TV will flag
    aif = np.zeros((15, 15, 3), dtype=np.uint8)
    result, frac = outlier_removal.remove_outliers(
        depth.copy(), aif, patch_type='tv', tv_thresh=0.5, to_plot=False
    )
    assert result[7, 7] < 50.0     # replaced with neighbour average (~2.0)
    assert frac > 0.0


def test_remove_outliers_fraction_range():
    aif = np.full((10, 10, 3), 128, dtype=np.uint8)   # all-constant → all flagged
    depth = np.ones((10, 10), dtype=np.float32)
    _, frac = outlier_removal.remove_outliers(
        depth.copy(), aif, patch_type='constant', diff_thresh=2, to_plot=False
    )
    assert 0.0 <= frac <= 1.0


def test_remove_outliers_invalid_patch_type():
    try:
        outlier_removal.remove_outliers(
            np.ones((5, 5)), np.zeros((5, 5, 3), dtype=np.uint8),
            patch_type='unknown', to_plot=False
        )
        assert False, "Should have raised AssertionError for unknown patch_type"
    except AssertionError:
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        test_total_variation_constant,
        test_total_variation_known,
        test_tv_map_constant_image,
        test_tv_map_odd_patch_required,
        test_tv_map_shape,
        test_find_high_tv_no_outliers,
        test_find_high_tv_known_spike,
        test_find_high_tv_returns_2d_indices,
        test_constant_patches_all_flat,
        test_constant_patches_high_variance,
        test_constant_patches_odd_required,
        test_constant_patches_returns_2d_indices,
        test_remove_outliers_no_change_when_clean,
        test_remove_outliers_replaces_spike,
        test_remove_outliers_fraction_range,
        test_remove_outliers_invalid_patch_type,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
