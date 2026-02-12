import numpy as np
import scipy.sparse

import globals
import forward_model
import section_search

globals.init_NYUv2()


# ---------------------------------------------------------------------------
# compute_u_v
# ---------------------------------------------------------------------------

def test_compute_u_v_shape():
    K = globals.MAX_KERNEL_SIZE
    u, v = forward_model.compute_u_v()
    assert u.shape == (1, 1, 1, K, K)
    assert v.shape == (1, 1, 1, K, K)


def test_compute_u_v_center_is_zero():
    K = globals.MAX_KERNEL_SIZE
    u, v = forward_model.compute_u_v()
    assert u[0, 0, 0, K // 2, K // 2] == 0.0
    assert v[0, 0, 0, K // 2, K // 2] == 0.0


def test_compute_u_v_range():
    K = globals.MAX_KERNEL_SIZE
    lim = K // 2
    u, v = forward_model.compute_u_v()
    assert u.min() == -lim
    assert u.max() == lim
    assert v.min() == -lim
    assert v.max() == lim


# ---------------------------------------------------------------------------
# compute_shifted_indices
# ---------------------------------------------------------------------------

def test_compute_shifted_indices_shape():
    width, height = 10, 12
    K = globals.MAX_KERNEL_SIZE
    row_indices, col_indices = forward_model.compute_shifted_indices(width, height)
    assert row_indices.shape == (width, height, K, K)
    assert col_indices.shape == (width, height, K, K)


def test_compute_shifted_indices_center_is_identity():
    # The center kernel entry (K//2, K//2) should point back to the pixel itself
    width, height = 10, 12
    K = globals.MAX_KERNEL_SIZE
    lim = K // 2
    row_indices, col_indices = forward_model.compute_shifted_indices(width, height)
    grid_i, grid_j = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    assert np.all(row_indices[:, :, lim, lim] == grid_i)
    assert np.all(col_indices[:, :, lim, lim] == grid_j)


# ---------------------------------------------------------------------------
# generate_mask
# ---------------------------------------------------------------------------

def test_generate_mask_total_length():
    width, height = 8, 10
    K = globals.MAX_KERNEL_SIZE
    row_indices, col_indices = forward_model.compute_shifted_indices(width, height)
    mask = forward_model.generate_mask(row_indices, col_indices, width, height)
    assert mask.shape == (width * height * K * K,)


def test_generate_mask_interior_pixel_all_valid():
    # An interior pixel (far from border) should have all K*K kernel entries valid
    width, height = 20, 20
    K = globals.MAX_KERNEL_SIZE
    lim = K // 2
    row_indices, col_indices = forward_model.compute_shifted_indices(width, height)
    mask = forward_model.generate_mask(row_indices, col_indices, width, height)
    mask_4d = mask.reshape(width, height, K, K)
    assert np.all(mask_4d[lim, lim, :, :])


def test_generate_mask_corner_pixel_has_invalid():
    # Corner pixel (0, 0) has out-of-bounds neighbours
    width, height = 20, 20
    K = globals.MAX_KERNEL_SIZE
    row_indices, col_indices = forward_model.compute_shifted_indices(width, height)
    mask = forward_model.generate_mask(row_indices, col_indices, width, height)
    mask_4d = mask.reshape(width, height, K, K)
    assert not np.all(mask_4d[0, 0, :, :])


# ---------------------------------------------------------------------------
# computer (CoC radius)
# ---------------------------------------------------------------------------

def test_computer_output_shape():
    width, height = 10, 12
    dpt = np.full((width, height), 2.0, dtype=np.float32)
    r = forward_model.computer(dpt, globals.Df)
    assert r.shape == (width, height, len(globals.Df))


def test_computer_all_values_above_thresh():
    width, height = 10, 12
    dpt = np.full((width, height), 2.0, dtype=np.float32)
    r = forward_model.computer(dpt, globals.Df)
    assert np.all(r >= globals.thresh)


def test_computer_at_focus_clamped_to_thresh():
    # When depth equals a focal plane distance, CoC = 0, so r is clamped to thresh
    width, height = 4, 4
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    r = forward_model.computer(dpt, globals.Df)
    assert np.allclose(r[:, :, 0], globals.thresh)


# ---------------------------------------------------------------------------
# computeG
# ---------------------------------------------------------------------------

def test_computeG_shape():
    width, height, fs = 6, 8, 5
    K = globals.MAX_KERNEL_SIZE
    u, v = forward_model.compute_u_v()
    r = np.full((width, height, fs, 1, 1), 2.0, dtype=np.float32)
    G, norm = forward_model.computeG(r, u, v)
    assert G.shape == (width, height, fs, K, K)


def test_computeG_normalized():
    # Each kernel should sum to approximately 1
    width, height, fs = 6, 8, 5
    u, v = forward_model.compute_u_v()
    r = np.full((width, height, fs, 1, 1), 2.0, dtype=np.float32)
    G, _ = forward_model.computeG(r, u, v)
    kernel_sums = G.sum(axis=(-2, -1))
    assert np.allclose(kernel_sums, 1.0, atol=1e-5)


def test_computeG_symmetric():
    # For uniform r, the kernel should be symmetric: G[..., i, j] == G[..., j, i]
    width, height, fs = 4, 4, 5
    u, v = forward_model.compute_u_v()
    r = np.full((width, height, fs, 1, 1), 1.5, dtype=np.float32)
    G, _ = forward_model.computeG(r, u, v)
    assert np.allclose(G, np.swapaxes(G, -1, -2), atol=1e-6)


# ---------------------------------------------------------------------------
# buildA
# ---------------------------------------------------------------------------

def test_buildA_returns_correct_number_of_matrices():
    width, height = 10, 10
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    A_stack = forward_model.buildA(dpt, u, v, row, col, mask)
    assert len(A_stack) == len(globals.Df)


def test_buildA_matrix_shape():
    width, height = 10, 10
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    A_stack = forward_model.buildA(dpt, u, v, row, col, mask)
    for A in A_stack:
        assert A.shape == (width * height, width * height)


def test_buildA_interior_row_sums_near_one():
    # Interior pixel rows should sum to ~1 (normalised Gaussian, full kernel present)
    width, height = 20, 20
    K = globals.MAX_KERNEL_SIZE
    lim = K // 2
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    u, v, row, col, mask = forward_model.precompute_indices(width, height)
    A_stack = forward_model.buildA(dpt, u, v, row, col, mask)
    A = A_stack[0].toarray()
    row_sums = A.sum(axis=1).reshape(width, height)
    interior = row_sums[lim:-lim, lim:-lim]
    assert np.allclose(interior, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

def test_forward_output_shape():
    width, height = 10, 12
    fs = len(globals.Df)
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    aif = np.ones((width, height, 3), dtype=np.float32)
    indices = forward_model.precompute_indices(width, height)
    stack = forward_model.forward(dpt, aif, indices=indices)
    assert stack.shape == (fs, width, height, 3)


def test_forward_zero_aif_gives_zero():
    width, height = 10, 10
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    aif = np.zeros((width, height, 3), dtype=np.float32)
    indices = forward_model.precompute_indices(width, height)
    stack = forward_model.forward(dpt, aif, indices=indices)
    assert np.allclose(stack, 0.0)


def test_forward_constant_aif_interior_preserved():
    # Blurring a constant image should give back the same constant for interior pixels
    width, height = 20, 20
    K = globals.MAX_KERNEL_SIZE
    lim = K // 2
    constant_val = 128.0
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    aif = np.full((width, height, 3), constant_val, dtype=np.float32)
    indices = forward_model.precompute_indices(width, height)
    stack = forward_model.forward(dpt, aif, indices=indices)
    for frame in stack:
        interior = frame[lim:-lim, lim:-lim, :]
        assert np.allclose(interior, constant_val, atol=1e-3)


def test_forward_deterministic():
    width, height = 8, 8
    rng = np.random.default_rng(0)
    dpt = np.full((width, height), globals.Df[1], dtype=np.float32)
    aif = rng.uniform(0, 255, (width, height, 3)).astype(np.float32)
    indices = forward_model.precompute_indices(width, height)
    stack1 = forward_model.forward(dpt, aif, indices=indices)
    stack2 = forward_model.forward(dpt, aif, indices=indices)
    assert np.array_equal(stack1, stack2)


def test_forward_without_precomputed_indices():
    # forward() should work when indices=None (computes them internally)
    width, height = 8, 8
    dpt = np.full((width, height), globals.Df[0], dtype=np.float32)
    aif = np.ones((width, height, 3), dtype=np.float32)
    stack = forward_model.forward(dpt, aif)
    assert stack.shape == (len(globals.Df), width, height, 3)


# ---------------------------------------------------------------------------
# build_fixed_pattern_csr
# ---------------------------------------------------------------------------

def test_build_fixed_pattern_csr_count():
    width, height, fs = 5, 6, 3
    _, _, row, col, _ = forward_model.precompute_indices(width, height)
    data = np.ones(row.size, dtype=np.float32)
    A_stack, _ = forward_model.build_fixed_pattern_csr(width, height, fs, row, col, data)
    assert len(A_stack) == fs


def test_build_fixed_pattern_csr_shape():
    width, height, fs = 5, 6, 3
    _, _, row, col, _ = forward_model.precompute_indices(width, height)
    data = np.ones(row.size, dtype=np.float32)
    A_stack, _ = forward_model.build_fixed_pattern_csr(width, height, fs, row, col, data)
    for A in A_stack:
        assert A.shape == (width * height, width * height)


def test_build_fixed_pattern_csr_shared_sparsity_pattern():
    # All matrices in the stack should share identical indptr and indices arrays
    width, height, fs = 5, 6, 4
    _, _, row, col, _ = forward_model.precompute_indices(width, height)
    data = np.ones(row.size, dtype=np.float32)
    A_stack, _ = forward_model.build_fixed_pattern_csr(width, height, fs, row, col, data)
    for A in A_stack[1:]:
        assert np.array_equal(A.indptr, A_stack[0].indptr)
        assert np.array_equal(A.indices, A_stack[0].indices)


def test_build_fixed_pattern_csr_order_is_permutation():
    width, height = 5, 6
    _, _, row, col, _ = forward_model.precompute_indices(width, height)
    data = np.ones(row.size, dtype=np.float32)
    _, order = forward_model.build_fixed_pattern_csr(width, height, 1, row, col, data)
    assert order.shape == (row.size,)
    assert np.array_equal(np.sort(order), np.arange(row.size))


def test_build_fixed_pattern_csr_equivalent_to_coo():
    # buildA via template must give identical matrices to direct COO construction,
    # and forward / objective_full results must also match end-to-end.
    IMAGE_RANGE = 255.0
    width, height = 10, 10
    rng = np.random.default_rng(0)

    gt_dpt = np.full((width, height), globals.Df[1], dtype=np.float32)
    gt_aif = rng.uniform(0, IMAGE_RANGE, (width, height, 3)).astype(np.float32)
    indices = forward_model.precompute_indices(width, height)
    defocus_stack = forward_model.forward(gt_dpt, gt_aif, indices=indices)

    fs = len(globals.Df)
    u, v, row, col, mask = indices

    sample_data = rng.random(row.size).astype(np.float32)
    template_A_stack = forward_model.build_fixed_pattern_csr(width, height, fs, row, col, sample_data)
    A_stack_v1 = forward_model.buildA(gt_dpt, u, v, row, col, mask, template_A_stack)
    A_stack_v2 = forward_model.buildA(gt_dpt, u, v, row, col, mask, template_A_stack=None)

    for i in range(fs):
        A = A_stack_v1[i]
        B = A_stack_v2[i]

        assert np.array_equal(A.indices, B.indices)
        assert np.array_equal(A.indptr, B.indptr)
        assert np.array_equal(A.data, B.data)

        A = A.copy(); A.sum_duplicates(); A.sort_indices()
        B = B.copy(); B.sum_duplicates(); B.sort_indices()
        assert A.shape == B.shape

        diff = (A - B).tocoo()
        assert np.allclose(diff.data, 0)
        assert diff.nnz == 0

        x = np.random.rand(width * height) * IMAGE_RANGE
        assert np.allclose(A.dot(x), B.dot(x))
        assert np.allclose(A.T.dot(x), B.T.dot(x))

    b1 = forward_model.forward(gt_dpt, gt_aif, indices=indices, template_A_stack=template_A_stack)
    b2 = forward_model.forward(gt_dpt, gt_aif, indices=indices, template_A_stack=None)
    assert np.allclose(b1, b2)

    o1 = section_search.objective_full(gt_dpt, gt_aif, defocus_stack, indices=indices, template_A_stack=None)
    o2 = section_search.objective_full(gt_dpt, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack)
    assert np.allclose(o1, o2)

    test_zs = [gt_dpt, rng.random(gt_dpt.shape).astype(np.float32), rng.random(gt_dpt.shape).astype(np.float32)]

    for z in test_zs:
        v1 = section_search.objective_full(z, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack)
        v2 = section_search.objective_full(z, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack)
        assert np.allclose(v1, v2), "objective_full not deterministic for template path"

    # Interleaved calls (catches aliasing of shared buffers)
    z1, z2 = test_zs[:2]
    a1 = section_search.objective_full(z1, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack)
    b1 = section_search.objective_full(z2, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack)
    a2 = section_search.objective_full(z1, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack)
    assert np.allclose(a1, a2), "state leaked between evaluations"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    tests = [
        test_compute_u_v_shape,
        test_compute_u_v_center_is_zero,
        test_compute_u_v_range,
        test_compute_shifted_indices_shape,
        test_compute_shifted_indices_center_is_identity,
        test_generate_mask_total_length,
        test_generate_mask_interior_pixel_all_valid,
        test_generate_mask_corner_pixel_has_invalid,
        test_computer_output_shape,
        test_computer_all_values_above_thresh,
        test_computer_at_focus_clamped_to_thresh,
        test_computeG_shape,
        test_computeG_normalized,
        test_computeG_symmetric,
        test_buildA_returns_correct_number_of_matrices,
        test_buildA_matrix_shape,
        test_buildA_interior_row_sums_near_one,
        test_forward_output_shape,
        test_forward_zero_aif_gives_zero,
        test_forward_constant_aif_interior_preserved,
        test_forward_deterministic,
        test_forward_without_precomputed_indices,
        test_build_fixed_pattern_csr_count,
        test_build_fixed_pattern_csr_shape,
        test_build_fixed_pattern_csr_shared_sparsity_pattern,
        test_build_fixed_pattern_csr_order_is_permutation,
        test_build_fixed_pattern_csr_equivalent_to_coo,
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