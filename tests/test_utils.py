import os
import tempfile
import shutil
import struct
import sys

import numpy as np
import utils


# ============================================================================
# Mathematical/Statistical Functions
# ============================================================================

# ---------------------------------------------------------------------------
# total_variation
# ---------------------------------------------------------------------------

def test_total_variation_constant():
    """Total variation of constant image should be zero."""
    img = np.ones((10, 10)) * 5.0
    assert utils.total_variation(img) == 0.0


def test_total_variation_horizontal_gradient():
    """Test TV with horizontal gradient."""
    # Horizontal ramp: each row is [0, 1, 2], no vertical differences
    img = np.tile(np.array([0.0, 1.0, 2.0]), (3, 1))
    # tv_x: 3 rows * 2 steps of 1 = 6; tv_y: 0
    assert utils.total_variation(img) == 6.0


def test_total_variation_vertical_gradient():
    """Test TV with vertical gradient."""
    # Vertical ramp: each column is [0, 1, 2]
    img = np.tile(np.array([[0.0], [1.0], [2.0]]), (1, 3))
    # tv_x: 0; tv_y: 3 columns * 2 steps of 1 = 6
    assert utils.total_variation(img) == 6.0


def test_total_variation_both_gradients():
    """Test TV with both horizontal and vertical gradients."""
    img = np.array([[0.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0]])
    # Horizontal: 3 rows * 2 steps of 1 = 6
    # Vertical: 3 cols * 2 steps of 1 = 6
    # Total: 12
    assert utils.total_variation(img) == 12.0


# ---------------------------------------------------------------------------
# compute_RMS
# ---------------------------------------------------------------------------

def test_compute_RMS_identical():
    """RMS error between identical arrays should be zero."""
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert utils.compute_RMS(pred, gt) == 0.0


def test_compute_RMS_known_value():
    """Test RMS with known error."""
    pred = np.array([[2.0, 3.0], [4.0, 5.0]])
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Differences: [1, 1, 1, 1]
    # Mean of squares: 1
    # RMS: 1
    assert utils.compute_RMS(pred, gt) == 1.0


def test_compute_RMS_larger_error():
    """Test RMS with larger known error."""
    pred = np.array([[3.0, 6.0]])
    gt = np.array([[0.0, 2.0]])
    # Differences: [3, 4]
    # Squares: [9, 16]
    # Mean: 12.5
    # RMS: 3.5355...
    expected = np.sqrt(12.5)
    assert np.isclose(utils.compute_RMS(pred, gt), expected)


# ---------------------------------------------------------------------------
# compute_AbsRel
# ---------------------------------------------------------------------------

def test_compute_AbsRel_identical():
    """AbsRel between identical positive arrays should be zero."""
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    gt = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert utils.compute_AbsRel(pred, gt) == 0.0


def test_compute_AbsRel_known_value():
    """Test AbsRel with known relative error."""
    pred = np.array([[2.0, 4.0]])
    gt = np.array([[1.0, 2.0]])
    # Relative errors: [|2-1|/1, |4-2|/2] = [1, 1]
    # Mean: 1
    assert np.isclose(utils.compute_AbsRel(pred, gt), 1.0)


def test_compute_AbsRel_handles_zero_gt():
    """AbsRel should handle near-zero ground truth with epsilon."""
    pred = np.array([[1.0, 2.0]])
    gt = np.array([[0.0, 1e-10]])
    # Should not crash and should use epsilon for stability
    result = utils.compute_AbsRel(pred, gt)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# compute_accuracy_metrics
# ---------------------------------------------------------------------------

def test_accuracy_metrics_identical():
    """All delta metrics should be 1.0 for identical predictions."""
    pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    gt = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metrics = utils.compute_accuracy_metrics(pred, gt)

    assert metrics["delta1"] == 1.0
    assert metrics["delta2"] == 1.0
    assert metrics["delta3"] == 1.0


def test_accuracy_metrics_threshold():
    """Test accuracy metrics at threshold boundaries."""
    # Create predictions at exact threshold boundaries
    pred = np.array([[1.0, 1.24, 1.26, 2.0]])
    gt = np.array([[1.0, 1.0, 1.0, 1.0]])
    # Ratios: [1.0, 1.24, 1.26, 2.0]
    # delta1: ratio < 1.25 → [1.0, 1.24] → 2/4 = 0.5
    # delta2: ratio < 1.5625 → [1.0, 1.24, 1.26] → 3/4 = 0.75
    # delta3: ratio < 1.953125 → [1.0, 1.24, 1.26] → 3/4 = 0.75

    metrics = utils.compute_accuracy_metrics(pred, gt)
    assert metrics["delta1"] == 0.5
    assert metrics["delta2"] == 0.75
    assert metrics["delta3"] == 0.75


def test_accuracy_metrics_handles_zero_gt():
    """Accuracy metrics should handle near-zero ground truth with epsilon."""
    pred = np.array([[1.0, 2.0]])
    gt = np.array([[0.0, 1e-10]])
    metrics = utils.compute_accuracy_metrics(pred, gt)

    # Should not crash
    assert "delta1" in metrics
    assert "delta2" in metrics
    assert "delta3" in metrics
    assert all(np.isfinite(v) for v in metrics.values())


# ============================================================================
# Image Processing Functions
# ============================================================================

# ---------------------------------------------------------------------------
# to_uint8
# ---------------------------------------------------------------------------

def test_to_uint8_valid_range():
    """Test conversion of valid uint8 range."""
    img = np.array([[0, 127, 255], [50, 100, 200]])
    result = utils.to_uint8(img)
    assert np.array_equal(result, img)
    assert result.dtype == np.int_


def test_to_uint8_clips_negative():
    """Test that negative values are clipped to 0."""
    img = np.array([[-10, -5, 0], [100, 200, 255]])
    result = utils.to_uint8(img)
    expected = np.array([[0, 0, 0], [100, 200, 255]])
    assert np.array_equal(result, expected)


def test_to_uint8_clips_high():
    """Test that values > 255 are clipped to 255."""
    img = np.array([[100, 255, 300], [400, 500, 1000]])
    result = utils.to_uint8(img)
    expected = np.array([[100, 255, 255], [255, 255, 255]])
    assert np.array_equal(result, expected)


def test_to_uint8_float_input():
    """Test conversion of float input."""
    img = np.array([[0.5, 127.8, 255.3], [50.1, 100.9, 200.0]])
    result = utils.to_uint8(img)
    # Conversion to int truncates, then clips
    expected = np.array([[0, 127, 255], [50, 100, 200]])
    assert np.array_equal(result, expected)


def test_to_uint8_rejects_invalid_type():
    """Test that non-numpy arrays raise TypeError."""
    try:
        utils.to_uint8([1, 2, 3])
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "NumPy array" in str(e)


# ============================================================================
# File I/O Functions
# ============================================================================

# ---------------------------------------------------------------------------
# read_bin_file
# ---------------------------------------------------------------------------

def test_read_bin_file_float32():
    """Test reading binary file with float32 data."""
    # Create a temporary binary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
        temp_path = f.name

        # Write header: type code (5=float32), height=2, width=3
        f.write(struct.pack('B', 5))  # float32 type code
        f.write(struct.pack('i', 2))  # height
        f.write(struct.pack('i', 3))  # width

        # Write data
        data = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)
        f.write(data.tobytes())

    try:
        # Read and verify
        result = utils.read_bin_file(temp_path)
        expected = data
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        assert np.allclose(result, expected)
    finally:
        os.unlink(temp_path)


def test_read_bin_file_uint8():
    """Test reading binary file with uint8 data."""
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.bin') as f:
        temp_path = f.name

        # Write header: type code (0=uint8), height=3, width=2
        f.write(struct.pack('B', 0))  # uint8 type code
        f.write(struct.pack('i', 3))  # height
        f.write(struct.pack('i', 2))  # width

        # Write data
        data = np.array([[100, 200], [50, 150], [0, 255]], dtype=np.uint8)
        f.write(data.tobytes())

    try:
        result = utils.read_bin_file(temp_path)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32  # Always returns float32
        assert np.allclose(result, data.astype(np.float32))
    finally:
        os.unlink(temp_path)


# ---------------------------------------------------------------------------
# exif_to_float
# ---------------------------------------------------------------------------

def test_exif_to_float_none():
    """Test that None input returns None."""
    assert utils.exif_to_float(None) is None


def test_exif_to_float_with_rational():
    """Test conversion of EXIF rational value."""
    # Mock an EXIF tag with rational value
    class MockRational:
        def __init__(self, num, den):
            self.num = num
            self.den = den

    class MockTag:
        def __init__(self, val):
            self.values = [val]

    tag = MockTag(MockRational(5, 2))
    result = utils.exif_to_float(tag)
    assert result == 2.5


def test_exif_to_float_with_simple_value():
    """Test conversion of EXIF simple numeric value."""
    class MockTag:
        def __init__(self, val):
            self.values = [val]

    tag = MockTag(3.14)
    result = utils.exif_to_float(tag)
    assert result == 3.14


# ---------------------------------------------------------------------------
# save_dpt_npy and load_dpt_npy
# ---------------------------------------------------------------------------

def test_save_load_dpt_npy_roundtrip():
    """Test saving and loading depth map in npy format."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create test depth map
        dpt = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)

        # Save
        utils.save_dpt_npy(temp_dir, 'test_depth', dpt)

        # Verify file exists
        expected_path = os.path.join(temp_dir, 'test_depth.npy')
        assert os.path.exists(expected_path)

        # Load and verify
        loaded = utils.load_dpt_npy(temp_dir, 'test_depth')
        assert loaded.shape == dpt.shape
        assert np.allclose(loaded, dpt)
    finally:
        shutil.rmtree(temp_dir)


def test_load_dpt_npy_preserves_dtype():
    """Test that loading preserves data type information."""
    temp_dir = tempfile.mkdtemp()

    try:
        dpt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        utils.save_dpt_npy(temp_dir, 'test', dpt)
        loaded = utils.load_dpt_npy(temp_dir, 'test')

        # Should preserve float64
        assert loaded.dtype == np.float64
    finally:
        shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# save_dpt
# ---------------------------------------------------------------------------

def test_save_dpt_scales_correctly():
    """Test that save_dpt scales depth by 1e4."""
    temp_dir = tempfile.mkdtemp()

    try:
        dpt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        utils.save_dpt(temp_dir, 'test_depth', dpt)

        # Load using skimage and verify scaling
        import skimage.io
        expected_path = os.path.join(temp_dir, 'test_depth.tiff')
        loaded = skimage.io.imread(expected_path)

        # Should be scaled by 1e4
        expected = dpt * 1e4
        assert np.allclose(loaded, expected)
    finally:
        shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# save_aif and load_aif
# ---------------------------------------------------------------------------

def test_save_load_aif_roundtrip():
    """Test saving and loading all-in-focus image."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Create test AIF image
        aif = np.array([[100.5, 200.7], [150.3, 250.9]], dtype=np.float32)

        # Save
        utils.save_aif(temp_dir, 'test_aif', aif)

        # Verify file exists
        expected_path = os.path.join(temp_dir, 'test_aif.tiff')
        assert os.path.exists(expected_path)

        # Load and verify
        loaded = utils.load_aif(temp_dir, 'test_aif')
        assert loaded.shape == aif.shape
        assert loaded.dtype == np.float32
        assert np.allclose(loaded, aif)
    finally:
        shutil.rmtree(temp_dir)


def test_save_aif_handles_3d():
    """Test that save_aif squeezes 3D images correctly."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Create 3D image with singleton dimension
        aif = np.array([[[100.0, 200.0], [150.0, 250.0]]], dtype=np.float32)

        # Save
        utils.save_aif(temp_dir, 'test_3d', aif)

        # Load and verify it was squeezed
        loaded = utils.load_aif(temp_dir, 'test_3d')
        assert loaded.shape == (2, 2)
    finally:
        shutil.rmtree(temp_dir)


# ============================================================================
# General Utility Functions
# ============================================================================

# ---------------------------------------------------------------------------
# create_experiment_folder
# ---------------------------------------------------------------------------

def test_create_experiment_folder_creates_directory():
    """Test that experiment folder is created."""
    base_dir = tempfile.mkdtemp()

    try:
        folder = utils.create_experiment_folder('test_exp', base_folder=base_dir)

        # Should exist
        assert os.path.exists(folder)
        assert os.path.isdir(folder)

        # Should contain experiment name
        assert 'test_exp' in folder

        # Should contain timestamp in format YYYY-MM-DD_HH-MM-SS
        folder_name = os.path.basename(folder)
        parts = folder_name.split('_')
        assert len(parts) >= 4  # test_exp_YYYY-MM-DD_HH-MM-SS

    finally:
        shutil.rmtree(base_dir)


def test_create_experiment_folder_naming_pattern():
    """Test that experiment folder follows naming pattern."""
    base_dir = tempfile.mkdtemp()

    try:
        folder = utils.create_experiment_folder('my_test', base_folder=base_dir)

        # Folder name should start with experiment name
        folder_name = os.path.basename(folder)
        assert folder_name.startswith('my_test_')

        # Should contain date-time pattern (at least has underscores and hyphens)
        assert '_' in folder_name
        assert '-' in folder_name

    finally:
        shutil.rmtree(base_dir)


# ---------------------------------------------------------------------------
# format_number
# ---------------------------------------------------------------------------

def test_format_number_small():
    """Test formatting of very small numbers uses scientific notation."""
    result = utils.format_number(0.0001)
    assert 'e' in result
    assert '1.000000e-04' == result


def test_format_number_very_small():
    """Test formatting of very small numbers."""
    result = utils.format_number(1e-6)
    assert 'e' in result
    assert '1.000000e-06' == result


def test_format_number_normal():
    """Test formatting of normal-sized numbers uses fixed notation."""
    result = utils.format_number(1.23456789)
    assert 'e' not in result
    assert '1.234568' == result


def test_format_number_large():
    """Test formatting of large numbers uses fixed notation."""
    result = utils.format_number(1234.56789)
    assert 'e' not in result
    assert '1234.567890' == result


def test_format_number_boundary():
    """Test formatting at the boundary (0.001)."""
    # Just above threshold - should use fixed
    result = utils.format_number(0.001)
    assert 'e' not in result

    # Just below threshold - should use scientific
    result = utils.format_number(0.0009)
    assert 'e' in result


def test_format_number_negative():
    """Test formatting of negative numbers."""
    result = utils.format_number(-0.0001)
    assert 'e' in result

    result = utils.format_number(-1.234)
    assert 'e' not in result


# ---------------------------------------------------------------------------
# kernel_size_heuristic
# ---------------------------------------------------------------------------

def test_kernel_size_heuristic_small_image():
    """Test heuristic for small image returns minimum of 7."""
    size = utils.kernel_size_heuristic(50, 50)
    assert size >= 7
    assert size % 2 == 1  # Should be odd


def test_kernel_size_heuristic_medium_image():
    """Test heuristic for medium image."""
    # For 640x480: avg = 560, 0.039 * 560 ≈ 21.84 → rounds to 22 → 23 (make odd)
    size = utils.kernel_size_heuristic(640, 480)
    expected = round(0.039 * 560)
    if expected % 2 == 0:
        expected += 1
    assert size == max(7, expected)
    assert size % 2 == 1


def test_kernel_size_heuristic_large_image():
    """Test heuristic for large image."""
    # For 1920x1080: avg = 1500, 0.039 * 1500 = 58.5 → rounds to 58 → 59 (make odd)
    size = utils.kernel_size_heuristic(1920, 1080)
    expected = round(0.039 * 1500)
    if expected % 2 == 0:
        expected += 1
    assert size == expected
    assert size % 2 == 1


def test_kernel_size_heuristic_always_odd():
    """Test that heuristic always returns odd values."""
    for width in [100, 200, 300, 400, 500, 640, 800, 1024, 1920]:
        for height in [100, 200, 300, 400, 500, 480, 600, 768, 1080]:
            size = utils.kernel_size_heuristic(width, height)
            assert size % 2 == 1, f"kernel_size_heuristic({width}, {height}) = {size} is even"
            assert size >= 7, f"kernel_size_heuristic({width}, {height}) = {size} < 7"


def test_kernel_size_heuristic_rectangular():
    """Test heuristic with rectangular images."""
    # Very wide
    size1 = utils.kernel_size_heuristic(1000, 100)
    # Very tall
    size2 = utils.kernel_size_heuristic(100, 1000)

    # Should be the same (uses average)
    assert size1 == size2
    assert size1 % 2 == 1


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == '__main__':
    print("Running tests for utils.py...")

    # Get all test functions from current module
    current_module = sys.modules[__name__]
    test_functions = [getattr(current_module, name) for name in dir(current_module)
                     if name.startswith('test_') and callable(getattr(current_module, name))]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests")

    if failed == 0:
        print("All tests passed!")
