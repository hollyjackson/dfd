import numpy as np
import scipy.sparse

import globals
import forward_model
import nesterov
import dataset_loader

# Set up camera/sensor globals (provides f, D, Df, ps, thresh, MAX_KERNEL_SIZE)
globals.init_NYUv2()
globals.MAX_KERNEL_SIZE = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_sparse(n):
    """Return an (n x n) float32 sparse identity matrix."""
    return scipy.sparse.eye(n, format='csr', dtype=np.float32)


def _make_tiny_case(width=8, height=8, seed=0):
    """Build a small synthetic (dpt, defocus_stack, indices) tuple via the forward model."""
    rng = np.random.default_rng(seed)
    # Constant depth at the first focus distance
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    aif = rng.uniform(0, 255, (width, height, 3)).astype(np.float32)
    indices = forward_model.precompute_indices(width, height)
    defocus_stack = forward_model.forward(dpt, aif, indices=indices)
    return dpt, aif, defocus_stack, indices


# ---------------------------------------------------------------------------
# buildb
# ---------------------------------------------------------------------------

def test_buildb_output_lengths():
    fs, width, height = 5, 6, 7
    stack = [np.random.default_rng(i).random((width, height, 3)) for i in range(fs)]
    r, g, b = nesterov.buildb(stack)
    assert len(r) == fs
    assert len(g) == fs
    assert len(b) == fs


def test_buildb_vector_shape():
    width, height = 6, 7
    stack = [np.zeros((width, height, 3))]
    r, g, b = nesterov.buildb(stack)
    assert r[0].shape == (width * height,)
    assert g[0].shape == (width * height,)
    assert b[0].shape == (width * height,)


def test_buildb_channel_separation():
    width, height = 4, 4
    img = np.zeros((width, height, 3), dtype=np.float32)
    img[:, :, 0] = 10.0
    img[:, :, 1] = 20.0
    img[:, :, 2] = 30.0
    r, g, b = nesterov.buildb([img])
    assert np.all(r[0] == 10.0)
    assert np.all(g[0] == 20.0)
    assert np.all(b[0] == 30.0)


# ---------------------------------------------------------------------------
# compute_Lipschitz_constant  (exact)
# ---------------------------------------------------------------------------

def test_compute_lipschitz_identity():
    A = _identity_sparse(50)
    L = nesterov.compute_Lipschitz_constant(A)
    assert abs(L - 1.0) < 1e-4


def test_compute_lipschitz_scaled():
    # ||3I||_2 = 3, so ||3I||² = 9
    A = 3.0 * _identity_sparse(50)
    L = nesterov.compute_Lipschitz_constant(A)
    assert abs(L - 9.0) < 1e-4


# ---------------------------------------------------------------------------
# approx_Lipschitz_constant  (power iteration)
# ---------------------------------------------------------------------------

def test_approx_lipschitz_identity():
    A = _identity_sparse(100)
    L = nesterov.approx_Lipschitz_constant(A, A.T)
    assert abs(L - 1.0) < 0.01


def test_approx_lipschitz_scaled():
    A = 3.0 * _identity_sparse(100)
    L = nesterov.approx_Lipschitz_constant(A, A.T)
    assert abs(L - 9.0) < 0.1


def test_approx_lipschitz_positive():
    rng = np.random.default_rng(7)
    B = rng.standard_normal((30, 15)).astype(np.float32)
    A = scipy.sparse.csr_matrix(B)
    L = nesterov.approx_Lipschitz_constant(A, A.T)
    assert L > 0.0


def test_approx_matches_exact_small_matrix():
    # With enough iterations, power method should be within 5% of exact value
    rng = np.random.default_rng(42)
    B = rng.standard_normal((20, 10)).astype(np.float32)
    A = scipy.sparse.csr_matrix(B)
    exact = nesterov.compute_Lipschitz_constant(A)
    approx = nesterov.approx_Lipschitz_constant(A, A.T, iters=50)
    assert abs(approx - exact) / exact < 0.05


# ---------------------------------------------------------------------------
# bounded_fista_3d
# ---------------------------------------------------------------------------

def test_bounded_fista_output_shape():
    dpt, _, defocus_stack, indices = _make_tiny_case(width=8, height=8)
    result = nesterov.bounded_fista_3d(
        dpt, defocus_stack, IMAGE_RANGE=255.0, indices=indices, verbose=False
    )
    assert result.shape == (8, 8, 3)


def test_bounded_fista_output_in_range():
    dpt, _, defocus_stack, indices = _make_tiny_case(width=8, height=8)
    result = nesterov.bounded_fista_3d(
        dpt, defocus_stack, IMAGE_RANGE=255.0, indices=indices, verbose=False
    )
    assert result.min() >= 0.0
    assert result.max() <= 255.0


def test_bounded_fista_unit_image_range():
    # IMAGE_RANGE=1.0 (normalised images) should clip to [0, 1]
    dpt, _, defocus_stack, indices = _make_tiny_case(width=8, height=8)
    defocus_stack_norm = defocus_stack / 255.0
    result = nesterov.bounded_fista_3d(
        dpt, defocus_stack_norm, IMAGE_RANGE=1.0, indices=indices, verbose=False
    )
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_bounded_fista_ground_truth_depth():
    # Given ground-truth depth, FISTA should recover the AIF on interior pixels.
    # Border pixels are excluded: the forward model truncates the blur kernel at
    # image boundaries, making those pixels poorly conditioned in the inverse problem.
    gt_aif, gt_dpt, _ = dataset_loader.load_example_image(fs=5, res='half')
    defocus_stack = forward_model.forward(gt_dpt, gt_aif)
    result = nesterov.bounded_fista_3d(
        gt_dpt, defocus_stack, IMAGE_RANGE=255.0,
        tol=1e-8, maxiter=500, verbose=True
    )
    pad = globals.MAX_KERNEL_SIZE // 2
    mse = np.mean((result[pad:-pad, pad:-pad] - gt_aif[pad:-pad, pad:-pad]) ** 2)
    print(f"MSE: {mse}")
    print(f"Result range: [{result.min()}, {result.max()}]")
    print(f"GT AIF range: [{gt_aif.min()}, {gt_aif.max()}]")
    assert mse < 0.5


def test_bounded_fista_zero_observations():
    # b=0 => min ||Ax||² s.t. 0 <= x <= 255 has the unique solution x=0
    # (x=0 is feasible and achieves loss=0, which is the global minimum)
    width, height = 6, 6
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    num_focal = len(globals.Df)
    defocus_stack = [np.zeros((width, height, 3), dtype=np.float32)] * num_focal
    indices = forward_model.precompute_indices(width, height)
    result = nesterov.bounded_fista_3d(
        dpt, defocus_stack, IMAGE_RANGE=255.0, indices=indices,
        tol=1e-8, maxiter=500, verbose=False
    )
    assert np.allclose(result, 0.0, atol=1e-3)


def test_bounded_fista_verbose_false():
    # Should complete without error when verbose=False
    dpt, _, defocus_stack, indices = _make_tiny_case(width=6, height=6)
    result = nesterov.bounded_fista_3d(
        dpt, defocus_stack, IMAGE_RANGE=255.0, indices=indices,
        verbose=False, maxiter=5
    )
    assert result.shape == (6, 6, 3)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        test_buildb_output_lengths,
        test_buildb_vector_shape,
        test_buildb_channel_separation,
        test_compute_lipschitz_identity,
        test_compute_lipschitz_scaled,
        test_approx_lipschitz_identity,
        test_approx_lipschitz_scaled,
        test_approx_lipschitz_positive,
        test_approx_matches_exact_small_matrix,
        test_bounded_fista_output_shape,
        test_bounded_fista_output_in_range,
        test_bounded_fista_unit_image_range,
        test_bounded_fista_ground_truth_depth,
        test_bounded_fista_zero_observations,
        test_bounded_fista_verbose_false,
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