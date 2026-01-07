
import forward_model
import utils

import nesterov
import section_search
import dataset_loader

import torch
import skimage
import numpy as np
import os

def test_forward_model():
    
    aif, dpt, gt_defocus_stack = utils.load_single_sample()

    # test forward model (NO torch)
    defocus_stack = forward_model.forward(dpt, aif)
    utils.plot_stacks_side_by_side(gt_defocus_stack, defocus_stack, globals.Df)

    # test forward_model (torch)
    defocus_stack = forward_model.forward_torch(dpt, aif)
    utils.plot_stacks_side_by_side(gt_defocus_stack, defocus_stack, globals.Df)

def test_windowed_mse():
    defocus_stack_test = np.random.rand(5, 256, 256, 3) * 255.
    pred = np.random.rand(5, 256, 256, 3) * 255.
    losses_1 = section_search.windowed_mse(defocus_stack_test, pred, window_size=3)
    losses_2 = section_search.windowed_mse_v2(defocus_stack_test, pred, window_size=3)
    
    assert np.allclose(losses_1, losses_2, atol=1e-6)

def test_nesterov():
    criterion = torch.nn.MSELoss()

    aif, dpt, gt_defocus_stack = utils.load_single_sample(res='half')
    
    # least squares
    defocus_stack = forward_model.forward(dpt, aif)

    IMAGE_RANGE = 255. # if defocus stack in [0-255]
    if defocus_stack.max() <= 1.5: # instead in [0-1]
        IMAGE_RANGE = 1.

    recon_aif = nesterov.bounded_fista_3d(dpt, defocus_stack, IMAGE_RANGE)
    utils.plot_compare_rgb(recon_aif, aif)
    plt.show()

    print('Norm between recon and gt aif:', np.linalg.norm(np.array(recon_aif) - np.array(aif)))

    defocus_stack_pred = forward_model.forward(dpt, recon_aif)
    loss = criterion(defocus_stack_pred, defocus_stack)

    print('MSE loss:',loss)


def test_least_squares_synthetic():    

    criterion = torch.nn.MSELoss()

    aif, dpt, gt_defocus_stack = utils.load_single_sample(res='half')
    
    # least squares
    defocus_stack = forward_model.forward(dpt, aif)

    print('Norm between my/their defocus stack:', np.linalg.norm(np.array(defocus_stack) - np.array(gt_defocus_stack)))
    print('MSE between stacks', np.mean((np.array(defocus_stack) - np.array(gt_defocus_stack))**2))


    recon_aif = least_squares.least_squares(dpt, gt_defocus_stack)
    utils.plot_compare_rgb(recon_aif, aif)
    plt.show()

    print('Norm between recon and gt aif:', np.linalg.norm(np.array(recon_aif) - np.array(aif)))

    defocus_stack_pred = forward_model.forward_torch(dpt, torch.from_numpy(recon_aif))
    loss = criterion(defocus_stack_pred, torch.from_numpy(defocus_stack))
    print('MSE loss:',loss)


# def test_k_min_indices_no_overlap():
#     width, height, num_Z = 5, 5, 20
#     k = 3
#     gss_window = 2
    
#     # Create random loss values and sort them to get indices
#     np.random.seed(0)
#     all_losses = np.random.rand(width, height, num_Z)
#     sorted_indices = np.argsort(all_losses, axis=2)
#     print(sorted_indices[0,0,:])
    
#     k_min_indices = section_search.k_min_indices_no_overlap(sorted_indices, k=k, gss_window=gss_window)
    
#     assert k_min_indices.shape == (width, height, k)
#     assert np.all(k_min_indices >= 0)
    
#     for i in range(width):
#         for j in range(height):
#             values = np.sort(k_min_indices[i, j])
#             for a in range(len(values)):
#                 for b in range(a+1, len(values)):
#                     assert abs(values[a] - values[b]) >= gss_window, \
#                         f"Overlap violation at ({i},{j}): {values}"
    
    
#     print("Sample indices at (0,0):", k_min_indices[0, 0])
#     print("Corresponding loss values:", all_losses[0, 0][k_min_indices[0, 0]])

# # test_k_min_indices_no_overlap()


# def test_k_min_indices_no_overlap_k1():
#     width, height, num_Z = 5, 5, 20
#     k = 1
#     gss_window = 2
    
#     # Create random loss values and sort them to get indices
#     np.random.seed(0)
#     all_losses = np.random.rand(width, height, num_Z)
#     sorted_indices = np.argsort(all_losses, axis=2)
#     print(sorted_indices[0,0,:])
    
#     k_min_indices = section_search.k_min_indices_no_overlap(sorted_indices, k=k, gss_window=gss_window)
    
#     assert k_min_indices.shape == (width, height, k)
#     assert np.all(k_min_indices >= 0)

#     assert np.array_equal(k_min_indices.squeeze(), sorted_indices[:,:,0])

def test_load_save_dpt():
    # Parameters
    m, n = 5, 6
    path = "."
    fn = "test_float32"
    file_path = os.path.join(path, fn + '.tiff')
    
    # Create random float array between 0.1 and 10
    dpt_orig = np.random.uniform(0.1, 10.0, size=(m, n)).astype(np.float32)
    
    # Save as TIFF
    utils.save_dpt(path, fn, dpt_orig)
    
    # Load it back
    dpt_loaded = dataset_loader.load_NYUv2_dpt(file_path, resize_frac=1)
    
    # Compare
    same = np.allclose(dpt_orig, dpt_loaded, rtol=1e-6, atol=0)
    assert same
    # print("Arrays identical:", same)
    
    # if not same:
    #     diff = arr_orig - arr_loaded
    #     print("Max abs diff:", np.max(np.abs(diff)))
    #     print("Sample original:", arr_orig.flatten()[:5])
    #     print("Sample loaded:  ", arr_loaded.flatten()[:5])
    
    # Delete test file
    try:
        os.remove(file_path)
        print("Test file deleted.")
    except OSError as e:
        print("Error deleting file:", e)

def test_template_A_stack(gt_dpt, gt_aif, defocus_stack):

    fs, width, height, _ = defocus_stack.shape

    indices = forward_model.precompute_indices(width, height)
    u, v, row, col, mask = indices
        
    sample_data = np.random.rand(indices[2].size).astype(np.float32)
    template_A_stack = forward_model.build_fixed_pattern_csr(width, height, fs, indices[2], indices[3], sample_data)
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
            
    
        x = np.random.rand(width*height) * IMAGE_RANGE
        diff = A.dot(x) - B.dot(x)
        assert np.allclose(diff, 0)
                          
        diff = A.T.dot(x) - B.T.dot(x)
        assert np.allclose(diff, 0)
    
    b1 = forward_model.forward(gt_dpt, gt_aif, indices=indices, template_A_stack=template_A_stack)
    b2 = forward_model.forward(gt_dpt, gt_aif, indices=indices, template_A_stack=None)
    
    assert np.allclose(b1, b2)
    
    o1 = section_search.objective_full(gt_dpt, gt_aif, defocus_stack, indices=indices, template_A_stack=None)
    o2 = section_search.objective_full(gt_dpt, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack)
    
    assert np.allclose(o1, o2)
    

    rng = np.random.default_rng(0)
    test_zs = [gt_dpt, rng.random(gt_dpt.shape, dtype=np.float32), rng.random(gt_dpt.shape, dtype=np.float32)]
    
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
