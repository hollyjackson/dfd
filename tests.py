
import forward_model
import utils

import least_squares

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



def test_least_squares():
    criterion = torch.nn.MSELoss()

    aif, dpt, gt_defocus_stack = utils.load_single_sample(res='half')
    
    # least squares
    defocus_stack = forward_model.forward(dpt, aif)

    recon_aif = least_squares.least_squares(dpt, defocus_stack)
    utils.plot_compare_rgb(recon_aif, aif)
    plt.show()

    print('Norm between recon and gt aif:', np.linalg.norm(np.array(recon_aif) - np.array(aif)))

    defocus_stack_pred = forward_model.forward_torch(dpt, torch.from_numpy(recon_aif))
    loss = criterion(defocus_stack_pred, torch.from_numpy(defocus_stack))

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


def test_k_min_indices_no_overlap():
    width, height, num_Z = 5, 5, 20
    k = 3
    gss_window = 2
    
    # Create random loss values and sort them to get indices
    np.random.seed(0)
    all_losses = np.random.rand(width, height, num_Z)
    sorted_indices = np.argsort(all_losses, axis=2)
    print(sorted_indices[0,0,:])
    
    k_min_indices = section_search.k_min_indices_no_overlap(sorted_indices, k=k, gss_window=gss_window)
    
    assert k_min_indices.shape == (width, height, k)
    assert np.all(k_min_indices >= 0)
    
    for i in range(width):
        for j in range(height):
            values = np.sort(k_min_indices[i, j])
            for a in range(len(values)):
                for b in range(a+1, len(values)):
                    assert abs(values[a] - values[b]) >= gss_window, \
                        f"Overlap violation at ({i},{j}): {values}"
    
    
    print("Sample indices at (0,0):", k_min_indices[0, 0])
    print("Corresponding loss values:", all_losses[0, 0][k_min_indices[0, 0]])

# test_k_min_indices_no_overlap()


def test_k_min_indices_no_overlap_k1():
    width, height, num_Z = 5, 5, 20
    k = 1
    gss_window = 2
    
    # Create random loss values and sort them to get indices
    np.random.seed(0)
    all_losses = np.random.rand(width, height, num_Z)
    sorted_indices = np.argsort(all_losses, axis=2)
    print(sorted_indices[0,0,:])
    
    k_min_indices = section_search.k_min_indices_no_overlap(sorted_indices, k=k, gss_window=gss_window)
    
    assert k_min_indices.shape == (width, height, k)
    assert np.all(k_min_indices >= 0)

    assert np.array_equal(k_min_indices.squeeze(), sorted_indices[:,:,0])

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
    dpt_loaded = utils.load_NYUv2_dpt(file_path, resize_frac=1)
    
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