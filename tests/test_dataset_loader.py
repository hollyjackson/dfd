import os
import shutil
import tempfile

import numpy as np
import skimage.io
from PIL import Image
from scipy.io import savemat

import utils
import dataset_loader
from dataset_params import DatasetParams


# ---------------------------------------------------------------------------
# Synthetic file helpers
# (each helper writes a file into a caller-supplied directory)
# ---------------------------------------------------------------------------

def _write_depth_tiff(tmpdir, width, height, value=1e4, name='depth.tiff'):
    """Write a float32 depth TIFF (raw, pre-division scale) into tmpdir.

    load_NYUv2_dpt reads the raw float32 and divides by 1e4, so pass
    ``value=1e4`` to expect 1.0 m after loading.
    """
    path = os.path.join(tmpdir, name)
    skimage.io.imsave(path, np.full((width, height), value, dtype=np.float32))
    return path


def _write_rgb_png(tmpdir, width, height, value=128, name='rgb.png'):
    """Write a uint8 RGB PNG into tmpdir."""
    path = os.path.join(tmpdir, name)
    skimage.io.imsave(path, np.full((width, height, 3), value, dtype=np.uint8))
    return path


def _write_mat_depth(tmpdir, height, width, depth_value=5.0, name='depth.mat'):
    """Write a synthetic Make3D .mat depth file into tmpdir."""
    grid = np.zeros((height, width, 4), dtype=np.float32)
    grid[:, :, 3] = depth_value
    path = os.path.join(tmpdir, name)
    savemat(path, {'Position3DGrid': grid})
    return path


def _write_focal_stack(tmpdir, n_frames=3, width=8, height=10):
    """Write n_frames synthetic JPEG focal-stack images into tmpdir."""
    for i in range(n_frames):
        arr = np.full((width, height, 3), i * 20, dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(tmpdir, f'a{i:02d}.jpg'))


def _write_calib(tmpdir, example_name, focal_depths, aperture, focal_length):
    """Write a synthetic MobileDepth calibration tree into tmpdir; return calib path."""
    calib_dir = os.path.join(
        tmpdir, 'photos-calibration-results', 'calibration', example_name
    )
    os.makedirs(calib_dir)
    lines = [f"{d} {aperture}" for d in focal_depths] + [str(focal_length)]
    with open(os.path.join(calib_dir, 'calibrated.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return calib_dir


# ---------------------------------------------------------------------------
# load_NYUv2_dpt
# ---------------------------------------------------------------------------

def test_load_NYUv2_dpt_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_depth_tiff(tmpdir, 20, 30)
        dpt = dataset_loader.load_NYUv2_dpt(path, resize_frac=1)
        assert dpt.dtype == np.float32


def test_load_NYUv2_dpt_scale():
    # Raw value 1e4 -> 1e4 / 1e4 = 1.0 m
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_depth_tiff(tmpdir, 20, 30, value=1e4)
        dpt = dataset_loader.load_NYUv2_dpt(path, resize_frac=1)
        assert np.allclose(dpt, 1.0)


def test_load_NYUv2_dpt_clipped_low():
    # Raw value 0.0 -> 0.0 m -> clipped to 0.1
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_depth_tiff(tmpdir, 20, 30, value=0.0)
        dpt = dataset_loader.load_NYUv2_dpt(path, resize_frac=1)
        assert np.allclose(dpt, 0.1)


def test_load_NYUv2_dpt_clipped_high():
    # Raw value 2e5 -> 20.0 m -> clipped to 10.0
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_depth_tiff(tmpdir, 20, 30, value=2e5)
        dpt = dataset_loader.load_NYUv2_dpt(path, resize_frac=1)
        assert np.allclose(dpt, 10.0)


def test_load_NYUv2_dpt_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_depth_tiff(tmpdir, 20, 30, value=5e3)
        dpt = dataset_loader.load_NYUv2_dpt(path, resize_frac=1)
        assert dpt.min() >= 0.1
        assert dpt.max() <= 10.0


def test_load_NYUv2_dpt_shape_full():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_depth_tiff(tmpdir, 20, 30)
        dpt = dataset_loader.load_NYUv2_dpt(path, resize_frac=1)
        assert dpt.shape == (20, 30)


def test_load_NYUv2_dpt_shape_half():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_depth_tiff(tmpdir, 20, 30)
        dpt = dataset_loader.load_NYUv2_dpt(path, resize_frac=2)
        assert dpt.shape == (10, 15)


def test_load_save_dpt():
    # Round-trip: save_dpt (scales by 1e4, saves float32 TIFF) ->
    # load_NYUv2_dpt (reads float32, divides by 1e4).
    # Values in [0.1, 10.0] survive the clip unchanged.
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = "test_float32"
        dpt_orig = np.random.uniform(0.1, 10.0, size=(5, 6)).astype(np.float32)
        utils.save_dpt(tmpdir, fn, dpt_orig)
        dpt_loaded = dataset_loader.load_NYUv2_dpt(
            os.path.join(tmpdir, fn + '.tiff'), resize_frac=1
        )
        assert np.allclose(dpt_orig, dpt_loaded, rtol=1e-4, atol=0)


# ---------------------------------------------------------------------------
# load_NYUv2_aif
# ---------------------------------------------------------------------------

def test_load_NYUv2_aif_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_rgb_png(tmpdir, 20, 30)
        aif = dataset_loader.load_NYUv2_aif(path, resize_frac=1)
        assert aif.dtype == np.float32


def test_load_NYUv2_aif_scale():
    # Pixel value 255 -> 255
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_rgb_png(tmpdir, 20, 30, value=255)
        aif = dataset_loader.load_NYUv2_aif(path, resize_frac=1)
        assert np.allclose(aif, 255.0)


def test_load_NYUv2_aif_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_rgb_png(tmpdir, 20, 30, value=128)
        aif = dataset_loader.load_NYUv2_aif(path, resize_frac=1)
        assert aif.min() >= 0.0
        assert aif.max() <= 255.0


def test_load_NYUv2_aif_shape_full():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_rgb_png(tmpdir, 20, 30)
        aif = dataset_loader.load_NYUv2_aif(path, resize_frac=1)
        assert aif.shape == (20, 30, 3)


def test_load_NYUv2_aif_shape_half():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_rgb_png(tmpdir, 20, 30)
        aif = dataset_loader.load_NYUv2_aif(path, resize_frac=2)
        assert aif.shape == (10, 15, 3)


# ---------------------------------------------------------------------------
# _load_make3d_depth
# ---------------------------------------------------------------------------

def test_load_make3d_depth_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_mat_depth(tmpdir, 15, 20, depth_value=3.0)
        dpt = dataset_loader._load_make3d_depth(path)
        assert dpt.shape == (15, 20)


def test_load_make3d_depth_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_mat_depth(tmpdir, 15, 20, depth_value=3.0)
        dpt = dataset_loader._load_make3d_depth(path)
        assert np.allclose(dpt, 3.0)


def test_load_make3d_depth_dtype():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_mat_depth(tmpdir, 15, 20)
        dpt = dataset_loader._load_make3d_depth(path)
        assert dpt.dtype == np.float32


# ---------------------------------------------------------------------------
# _find_mobile_depth_example_dir
# ---------------------------------------------------------------------------

def test_find_mobile_depth_example_dir_found():
    with tempfile.TemporaryDirectory() as root:
        target = os.path.join(root, 'subfolder', 'keyboard')
        os.makedirs(target)
        result = dataset_loader._find_mobile_depth_example_dir(root, 'keyboard')
        assert result == target


def test_find_mobile_depth_example_dir_not_found():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, 'subfolder', 'bottles'))
        try:
            dataset_loader._find_mobile_depth_example_dir(root, 'keyboard')
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


def test_find_mobile_depth_example_dir_multiple_subfolders():
    # Only the second subfolder contains the example
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, 'group1', 'bottles'))
        target = os.path.join(root, 'group2', 'keyboard')
        os.makedirs(target)
        result = dataset_loader._find_mobile_depth_example_dir(root, 'keyboard')
        assert result == target


# ---------------------------------------------------------------------------
# _load_mobile_depth_focal_stack
# ---------------------------------------------------------------------------

def test_load_mobile_depth_focal_stack_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_focal_stack(tmpdir, n_frames=4, width=8, height=10)
        stack = dataset_loader._load_mobile_depth_focal_stack(tmpdir, resize_frac=1)
        assert stack.shape[0] == 4


def test_load_mobile_depth_focal_stack_shape():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_focal_stack(tmpdir, n_frames=3, width=8, height=10)
        stack = dataset_loader._load_mobile_depth_focal_stack(tmpdir, resize_frac=1)
        assert stack.shape == (3, 8, 10, 3)


def test_load_mobile_depth_focal_stack_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_focal_stack(tmpdir, n_frames=3, width=8, height=10)
        stack = dataset_loader._load_mobile_depth_focal_stack(tmpdir, resize_frac=1)
        assert stack.min() >= 0.0
        assert stack.max() <= 255.0


def test_load_mobile_depth_focal_stack_half_res():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_focal_stack(tmpdir, n_frames=2, width=8, height=10)
        stack = dataset_loader._load_mobile_depth_focal_stack(tmpdir, resize_frac=2)
        assert stack.shape == (2, 4, 5, 3)


# ---------------------------------------------------------------------------
# _load_mobile_depth_calibration
# ---------------------------------------------------------------------------

def test_load_mobile_depth_calibration_values():
    focal_depths = [1.0, 1.5, 2.0]
    with tempfile.TemporaryDirectory() as root:
        _write_calib(root, 'keyboard', focal_depths, aperture=0.125, focal_length=0.005)
        calib_dir, Df, f, D = dataset_loader._load_mobile_depth_calibration('keyboard', root)
        assert np.allclose(Df, focal_depths)
        assert f == 0.005
        assert D == 0.125


def test_load_mobile_depth_calibration_returns_calib_dir():
    with tempfile.TemporaryDirectory() as root:
        _write_calib(root, 'keyboard', [1.0, 2.0], aperture=0.1, focal_length=0.004)
        calib_dir, _, _, _ = dataset_loader._load_mobile_depth_calibration('keyboard', root)
        expected = os.path.join(root, 'photos-calibration-results', 'calibration', 'keyboard')
        assert calib_dir == expected


def test_load_mobile_depth_calibration_name_remapping():
    # 'metals' should resolve to the 'metal' calibration folder
    with tempfile.TemporaryDirectory() as root:
        _write_calib(root, 'metal', [1.0], aperture=0.1, focal_length=0.004)
        _, Df, _, _ = dataset_loader._load_mobile_depth_calibration('metals', root)
        assert np.allclose(Df, [1.0])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        test_load_NYUv2_dpt_dtype,
        test_load_NYUv2_dpt_scale,
        test_load_NYUv2_dpt_clipped_low,
        test_load_NYUv2_dpt_clipped_high,
        test_load_NYUv2_dpt_range,
        test_load_NYUv2_dpt_shape_full,
        test_load_NYUv2_dpt_shape_half,
        test_load_save_dpt,
        test_load_NYUv2_aif_dtype,
        test_load_NYUv2_aif_scale,
        test_load_NYUv2_aif_range,
        test_load_NYUv2_aif_shape_full,
        test_load_NYUv2_aif_shape_half,
        test_load_make3d_depth_shape,
        test_load_make3d_depth_values,
        test_load_make3d_depth_dtype,
        test_find_mobile_depth_example_dir_found,
        test_find_mobile_depth_example_dir_not_found,
        test_find_mobile_depth_example_dir_multiple_subfolders,
        test_load_mobile_depth_focal_stack_count,
        test_load_mobile_depth_focal_stack_shape,
        test_load_mobile_depth_focal_stack_range,
        test_load_mobile_depth_focal_stack_half_res,
        test_load_mobile_depth_calibration_values,
        test_load_mobile_depth_calibration_returns_calib_dir,
        test_load_mobile_depth_calibration_name_remapping,
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
