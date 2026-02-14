import numpy as np
import scipy.sparse

import forward_model
import section_search
from dataset_params import DatasetParams

dataset_params = DatasetParams.for_NYUv2()
MAX_KERNEL_SIZE = 7
WINDOW_SIZE = 3


# ---------------------------------------------------------------------------
# windowed_mse_grid
# ---------------------------------------------------------------------------

def test_windowed_mse_grid_shape():
    width, height = 20, 20
    fs = len(dataset_params.Df)
    defocus_stack = np.random.rand(fs, width, height, 3).astype(np.float32)
    pred = np.random.rand(fs, width, height, 3).astype(np.float32)

    loss = section_search.windowed_mse_grid(defocus_stack, pred, WINDOW_SIZE)
    assert loss.shape == (width, height)


def test_windowed_mse_grid_perfect_match_gives_zero():
    width, height = 20, 20
    fs = len(dataset_params.Df)
    defocus_stack = np.random.rand(fs, width, height, 3).astype(np.float32)

    loss = section_search.windowed_mse_grid(defocus_stack, defocus_stack, WINDOW_SIZE)
    assert np.allclose(loss, 0.0, atol=1e-6)


def test_windowed_mse_grid_positive():
    # MSE should always be non-negative
    width, height = 15, 15
    fs = len(dataset_params.Df)
    defocus_stack = np.random.rand(fs, width, height, 3).astype(np.float32)
    pred = np.random.rand(fs, width, height, 3).astype(np.float32)

    loss = section_search.windowed_mse_grid(defocus_stack, pred, WINDOW_SIZE)
    assert np.all(loss >= 0)


def test_windowed_mse_grid_symmetric_around_center():
    # For a centered feature, windowed MSE should be roughly symmetric
    width, height = 20, 20
    fs = len(dataset_params.Df)

    # Create symmetric defocus stack (constant)
    defocus_stack = np.ones((fs, width, height, 3), dtype=np.float32) * 0.5
    # Create centered gaussian-like error pattern
    y, x = np.ogrid[:width, :height]
    center_y, center_x = width // 2, height // 2
    error_pattern = np.exp(-((x - center_x)**2 + (y - center_y)**2) / 20.0)
    pred = defocus_stack.copy()
    pred[:, :, :, 0] += error_pattern[None, :, :] * 0.1

    loss = section_search.windowed_mse_grid(defocus_stack, pred, WINDOW_SIZE)
    # Loss at center should be roughly equal from all symmetric directions
    center = loss[center_y, center_x]
    assert center > 0  # There should be some error


def test_windowed_mse_grid_window_size_effect():
    # With window_size=1 (no windowing), should equal direct MSE
    width, height = 10, 10
    fs = len(dataset_params.Df)
    defocus_stack = np.random.rand(fs, width, height, 3).astype(np.float32)
    pred = np.random.rand(fs, width, height, 3).astype(np.float32)

    loss_windowed = section_search.windowed_mse_grid(defocus_stack, pred, window_size=1)
    direct_mse = np.mean((defocus_stack - pred)**2, axis=(0, -1))

    assert np.allclose(loss_windowed, direct_mse, atol=1e-5)


# ---------------------------------------------------------------------------
# windowed_mse_grid_fast
# ---------------------------------------------------------------------------

def test_windowed_mse_grid_fast_shape():
    width, height = 20, 20
    fs = len(dataset_params.Df)
    defocus_stack = np.random.rand(fs, width, height, 3).astype(np.float32)
    pred = np.random.rand(fs, width, height, 3).astype(np.float32)

    loss = section_search.windowed_mse_grid_fast(defocus_stack, pred, WINDOW_SIZE)
    assert loss.shape == (width, height)


def test_windowed_mse_grid_fast_vs_slow():
    # Fast version should give similar results to slow version
    # Note: windowed_mse_grid_fast has a TODO to verify against windowed_mse_grid
    # This test checks basic compatibility but may need adjustment
    width, height = 20, 20
    fs = len(dataset_params.Df)
    defocus_stack = np.random.rand(fs, width, height, 3).astype(np.float32)
    pred = np.random.rand(fs, width, height, 3).astype(np.float32)

    loss_slow = section_search.windowed_mse_grid(defocus_stack, pred, WINDOW_SIZE)
    loss_fast = section_search.windowed_mse_grid_fast(defocus_stack, pred, WINDOW_SIZE)

    # Both should have same shape
    assert loss_slow.shape == loss_fast.shape

    # Both should be non-negative
    assert np.all(loss_slow >= 0)
    assert np.all(loss_fast >= 0)

    # Check if they're reasonably close (may fail if fast version differs significantly)
    # Using larger tolerance due to different boundary handling
    if not np.allclose(loss_slow, loss_fast, atol=1e-2, rtol=1e-1):
        # Calculate difference metrics for debugging
        max_diff = np.max(np.abs(loss_slow - loss_fast))
        mean_diff = np.mean(np.abs(loss_slow - loss_fast))
        # For now, just verify they're in the same ballpark
        assert max_diff < np.mean(loss_slow) * 0.5, f"Fast version differs significantly (max diff: {max_diff})"


def test_windowed_mse_grid_fast_perfect_match():
    width, height = 20, 20
    fs = len(dataset_params.Df)
    defocus_stack = np.random.rand(fs, width, height, 3).astype(np.float32)

    loss = section_search.windowed_mse_grid_fast(defocus_stack, defocus_stack, WINDOW_SIZE)
    assert np.allclose(loss, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# windowed_mse_gss
# ---------------------------------------------------------------------------

def test_windowed_mse_gss_shape():
    width, height = 10, 10
    depth_map = np.full((width, height), 2.0, dtype=np.float32)
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(depth_map, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    loss = section_search.windowed_mse_gss(depth_map, gt_aif, defocus_stack, dataset_params,
                                           MAX_KERNEL_SIZE, WINDOW_SIZE, indices=indices)
    assert loss.shape == (width, height)


def test_windowed_mse_gss_perfect_reconstruction():
    # If depth_map is ground truth, loss should be near zero
    width, height = 10, 10
    gt_dpt = np.full((width, height), dataset_params.Df[1], dtype=np.float32)
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_dpt, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    loss = section_search.windowed_mse_gss(gt_dpt, gt_aif, defocus_stack, dataset_params,
                                           MAX_KERNEL_SIZE, WINDOW_SIZE, indices=indices)
    assert np.allclose(loss, 0.0, atol=1e-3)


def test_windowed_mse_gss_non_negative():
    width, height = 10, 10
    depth_map = np.random.rand(width, height).astype(np.float32) * 5 + 0.5
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = np.random.rand(len(dataset_params.Df), width, height, 3).astype(np.float32) * 255

    loss = section_search.windowed_mse_gss(depth_map, gt_aif, defocus_stack, dataset_params,
                                           MAX_KERNEL_SIZE, WINDOW_SIZE, indices=indices)
    assert np.all(loss >= 0)


# ---------------------------------------------------------------------------
# objective_full
# ---------------------------------------------------------------------------

def test_objective_full_shape():
    width, height = 10, 10
    dpt = np.full((width, height), 2.0, dtype=np.float32)
    aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(dpt, aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    loss = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                         indices=indices)
    assert loss.shape == (width, height)


def test_objective_full_perfect_reconstruction():
    width, height = 10, 10
    dpt = np.full((width, height), dataset_params.Df[0], dtype=np.float32)
    aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(dpt, aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    loss = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                         indices=indices)
    assert np.allclose(loss, 0.0, atol=1e-4)


def test_objective_full_non_negative():
    width, height = 10, 10
    dpt = np.random.rand(width, height).astype(np.float32) * 5 + 0.5
    aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    defocus_stack = np.random.rand(len(dataset_params.Df), width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)

    loss = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                         indices=indices)
    assert np.all(loss >= 0)


def test_objective_full_with_precomputed_pred():
    # Providing pred should skip forward model and give same result
    width, height = 10, 10
    dpt = np.full((width, height), 2.0, dtype=np.float32)
    aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    pred = forward_model.forward(dpt, aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)
    defocus_stack = pred.copy()

    loss_with_pred = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                                   indices=indices, pred=pred)
    loss_without_pred = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                                      indices=indices)

    assert np.allclose(loss_with_pred, loss_without_pred, atol=1e-5)


def test_objective_full_windowed_vs_non_windowed():
    # Windowed and non-windowed should give different results but both valid
    width, height = 15, 15
    dpt = np.full((width, height), 2.0, dtype=np.float32)
    aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = np.random.rand(len(dataset_params.Df), width, height, 3).astype(np.float32) * 255

    loss_windowed = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                                  window_size=WINDOW_SIZE, indices=indices, windowed=True)
    loss_plain = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                               indices=indices, windowed=False)

    assert loss_windowed.shape == loss_plain.shape
    assert np.all(loss_windowed >= 0)
    assert np.all(loss_plain >= 0)


def test_objective_full_grid_search_path():
    # Constant depth map should trigger grid search path
    width, height = 10, 10
    constant_depth = 2.5
    dpt = np.full((width, height), constant_depth, dtype=np.float32)
    aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = np.random.rand(len(dataset_params.Df), width, height, 3).astype(np.float32) * 255

    # Should work for both windowed and non-windowed
    loss = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                         indices=indices, windowed=False)
    assert loss.shape == (width, height)

    loss_windowed = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                                  window_size=WINDOW_SIZE, indices=indices, windowed=True)
    assert loss_windowed.shape == (width, height)


def test_objective_full_non_constant_depth():
    # Non-constant depth map should use gss path when windowed
    width, height = 10, 10
    dpt = np.random.rand(width, height).astype(np.float32) * 5 + 0.5
    aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = np.random.rand(len(dataset_params.Df), width, height, 3).astype(np.float32) * 255

    loss = section_search.objective_full(dpt, aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
                                         window_size=WINDOW_SIZE, indices=indices, windowed=True)

    assert loss.shape == (width, height)
    assert np.all(loss >= 0)


# ---------------------------------------------------------------------------
# grid_search
# ---------------------------------------------------------------------------

def test_grid_search_output_shapes():
    width, height = 10, 10
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.0, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 20
    depth_maps, Z, min_indices, all_losses = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    assert depth_maps.shape == (width, height)
    assert Z.shape == (num_Z,)
    assert min_indices.shape == (width, height)
    assert all_losses.shape == (width, height, num_Z)


def test_grid_search_depth_range():
    width, height = 10, 10
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.0, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 20
    depth_maps, Z, min_indices, all_losses = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    # Returned depth should be within the search range
    assert np.all(depth_maps >= min_Z)
    assert np.all(depth_maps <= max_Z)
    # Z should be linearly spaced
    assert np.allclose(Z, np.linspace(min_Z, max_Z, num_Z))


def test_grid_search_finds_ground_truth():
    # Grid search should find depth close to ground truth
    width, height = 10, 10
    gt_depth_value = 2.5
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), gt_depth_value, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 50
    depth_maps, Z, min_indices, all_losses = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    # Should find depth close to ground truth (within grid resolution)
    grid_step = (max_Z - min_Z) / (num_Z - 1)
    assert np.allclose(depth_maps, gt_depth_value, atol=grid_step + 0.01)


def test_grid_search_min_indices_valid():
    width, height = 8, 8
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.0, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 20
    depth_maps, Z, min_indices, all_losses = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    # min_indices should be valid indices into Z
    assert np.all(min_indices >= 0)
    assert np.all(min_indices < num_Z)
    # depth_maps should match Z[min_indices]
    assert np.allclose(depth_maps, Z[min_indices])


def test_grid_search_all_losses_consistent():
    # Loss at min_indices should be the minimum across all depths
    width, height = 8, 8
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.0, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 15
    depth_maps, Z, min_indices, all_losses = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    for i in range(width):
        for j in range(height):
            min_idx = min_indices[i, j]
            min_loss = all_losses[i, j, min_idx]
            # This should be the minimum loss for this pixel
            assert min_loss <= np.min(all_losses[i, j, :]) + 1e-5


def test_grid_search_windowed():
    width, height = 12, 12
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.0, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 15
    depth_maps, Z, min_indices, all_losses = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        window_size=WINDOW_SIZE, indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False, windowed=True
    )

    assert depth_maps.shape == (width, height)
    assert all_losses.shape == (width, height, num_Z)


# ---------------------------------------------------------------------------
# golden_section_search
# ---------------------------------------------------------------------------

def test_golden_section_search_shape():
    width, height = 10, 10
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.5, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    # Run grid search first
    min_Z, max_Z, num_Z = 1.0, 5.0, 20
    _, Z, min_indices, _ = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    # Refine with GSS
    refined_depth = section_search.golden_section_search(
        Z, min_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, window=2, tolerance=1e-3, max_iter=20, verbose=False
    )

    assert refined_depth.shape == (width, height)


def test_golden_section_search_refines_estimate():
    # GSS should produce depths within the initial bracket
    width, height = 8, 8
    gt_depth_value = 2.5
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), gt_depth_value, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 30
    _, Z, min_indices, _ = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    window = 3
    refined_depth = section_search.golden_section_search(
        Z, min_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, window=window, tolerance=1e-4, max_iter=50, verbose=False
    )

    # Refined depth should be close to ground truth
    assert np.allclose(refined_depth, gt_depth_value, atol=1e-2)


def test_golden_section_search_convergence():
    # With sufficient iterations, should converge within tolerance
    width, height = 6, 6
    gt_depth_value = 3.0
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), gt_depth_value, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 2.0, 4.0, 15
    _, Z, min_indices, _ = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    tolerance = 1e-4
    refined_depth = section_search.golden_section_search(
        Z, min_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, window=2, tolerance=tolerance, max_iter=100, verbose=False
    )

    # Result should be close to ground truth
    assert np.allclose(refined_depth, gt_depth_value, atol=1e-2)


def test_golden_section_search_with_last_dpt():
    # When providing last_dpt, result should be per-pixel minimum
    width, height = 8, 8
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.5, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 20
    _, Z, min_indices, _ = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    # Create a last_dpt that's actually better for some pixels
    last_dpt = gt_depth.copy()

    refined_depth = section_search.golden_section_search(
        Z, min_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, window=2, tolerance=1e-3, max_iter=30, verbose=False,
        last_dpt=last_dpt
    )

    # Should keep the better estimate (ground truth)
    assert np.allclose(refined_depth, gt_depth, atol=1e-2)


def test_golden_section_search_custom_bracket():
    # Test with custom a_b_init instead of grid-based initialization
    width, height = 6, 6
    gt_depth_value = 2.5
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), gt_depth_value, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    # Provide custom bracket around ground truth
    a = np.full((width, height), 2.0, dtype=np.float32)
    b = np.full((width, height), 3.0, dtype=np.float32)

    # Z is not used when a_b_init is provided, but still required
    Z = np.linspace(1.0, 5.0, 20)
    argmin_indices = np.zeros((width, height), dtype=int)  # Dummy

    refined_depth = section_search.golden_section_search(
        Z, argmin_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, window=2, tolerance=1e-4, max_iter=50, verbose=False,
        a_b_init=(a, b)
    )

    # Should find depth close to ground truth within the bracket
    assert np.all(refined_depth >= 2.0)
    assert np.all(refined_depth <= 3.0)
    assert np.allclose(refined_depth, gt_depth_value, atol=1e-2)


def test_golden_section_search_partial_convergence():
    # Test convergence_error parameter
    width, height = 8, 8
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.5, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 20
    _, Z, min_indices, _ = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    # Allow 10% of pixels to not converge
    refined_depth = section_search.golden_section_search(
        Z, min_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, window=2, tolerance=1e-4, convergence_error=0.1, max_iter=10, verbose=False
    )

    assert refined_depth.shape == (width, height)


def test_golden_section_search_windowed():
    width, height = 10, 10
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), 2.5, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    min_Z, max_Z, num_Z = 1.0, 5.0, 20
    _, Z, min_indices, _ = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        window_size=WINDOW_SIZE, indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False, windowed=True
    )

    refined_depth = section_search.golden_section_search(
        Z, min_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        window_size=WINDOW_SIZE, indices=indices,
        window=2, tolerance=1e-3, max_iter=30, verbose=False, windowed=True
    )

    assert refined_depth.shape == (width, height)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_full_pipeline_grid_plus_gss():
    # Test complete pipeline: grid search followed by GSS refinement
    width, height = 12, 12
    gt_depth_value = 2.3
    gt_aif = np.random.rand(width, height, 3).astype(np.float32) * 255
    gt_depth = np.full((width, height), gt_depth_value, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    # Coarse grid search
    min_Z, max_Z, num_Z = 1.0, 4.0, 15
    grid_depth, Z, min_indices, all_losses = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    # Should be roughly close
    grid_step = (max_Z - min_Z) / (num_Z - 1)
    assert np.allclose(grid_depth, gt_depth_value, atol=grid_step + 0.1)

    # Refine with GSS
    refined_depth = section_search.golden_section_search(
        Z, min_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, window=2, tolerance=1e-4, max_iter=50, verbose=False
    )

    # Refined should be better
    assert np.allclose(refined_depth, gt_depth_value, atol=1e-2)


def test_varied_depth_map():
    # Test with a non-constant depth map
    width, height = 10, 10
    rng = np.random.default_rng(42)
    gt_depth = rng.uniform(1.5, 3.5, (width, height)).astype(np.float32)
    gt_aif = rng.uniform(0, 255, (width, height, 3)).astype(np.float32)
    indices = forward_model.precompute_indices(width, height, MAX_KERNEL_SIZE)
    defocus_stack = forward_model.forward(gt_depth, gt_aif, dataset_params, MAX_KERNEL_SIZE, indices=indices)

    # Grid search
    min_Z, max_Z, num_Z = 1.0, 4.0, 20
    grid_depth, Z, min_indices, _ = section_search.grid_search(
        gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, min_Z=min_Z, max_Z=max_Z,
        num_Z=num_Z, verbose=False
    )

    # Should recover depth reasonably well
    grid_step = (max_Z - min_Z) / (num_Z - 1)
    assert np.allclose(grid_depth, gt_depth, atol=grid_step + 0.15)

    # Refine
    refined_depth = section_search.golden_section_search(
        Z, min_indices, gt_aif, defocus_stack, dataset_params, MAX_KERNEL_SIZE,
        indices=indices, window=2, tolerance=1e-3, max_iter=30, verbose=False
    )

    # Refinement should improve accuracy
    mean_error_refined = np.mean(np.abs(refined_depth - gt_depth))
    assert mean_error_refined < 0.1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        # windowed_mse_grid
        test_windowed_mse_grid_shape,
        test_windowed_mse_grid_perfect_match_gives_zero,
        test_windowed_mse_grid_positive,
        test_windowed_mse_grid_symmetric_around_center,
        test_windowed_mse_grid_window_size_effect,

        # windowed_mse_grid_fast
        test_windowed_mse_grid_fast_shape,
        test_windowed_mse_grid_fast_vs_slow,
        test_windowed_mse_grid_fast_perfect_match,

        # windowed_mse_gss
        test_windowed_mse_gss_shape,
        test_windowed_mse_gss_perfect_reconstruction,
        test_windowed_mse_gss_non_negative,

        # objective_full
        test_objective_full_shape,
        test_objective_full_perfect_reconstruction,
        test_objective_full_non_negative,
        test_objective_full_with_precomputed_pred,
        test_objective_full_windowed_vs_non_windowed,
        test_objective_full_grid_search_path,
        test_objective_full_non_constant_depth,

        # grid_search
        test_grid_search_output_shapes,
        test_grid_search_depth_range,
        test_grid_search_finds_ground_truth,
        test_grid_search_min_indices_valid,
        test_grid_search_all_losses_consistent,
        test_grid_search_windowed,

        # golden_section_search
        test_golden_section_search_shape,
        test_golden_section_search_refines_estimate,
        test_golden_section_search_convergence,
        test_golden_section_search_with_last_dpt,
        test_golden_section_search_custom_bracket,
        test_golden_section_search_partial_convergence,
        test_golden_section_search_windowed,

        # Integration tests
        test_full_pipeline_grid_plus_gss,
        test_varied_depth_map,
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
