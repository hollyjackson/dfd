"""Alternating-minimization solver for the joint depth-from-defocus problem.

Alternates between two sub-problems until num_epochs iterations are reached:

  AIF step   — given dpt, solve for aif via bounded FISTA
               (Nesterov's accelerated gradient method).
  Depth step — given aif, solve for dpt via a coarse grid search
               followed by golden-section refinement.
"""
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import forward_model
import globals
import initialization
import nesterov
import outlier_removal
import section_search
import utils

def mse_loss(pred, gt):
    """Mean squared error between pred and gt."""
    return np.mean((gt - pred)**2)


def coordinate_descent(defocus_stack, experiment_folder='experiments', gss_tol=1e-2, gss_window=1, T_0=100, alpha=None, num_epochs=25, nesterov_first=True, save_plots=True, show_plots=False, depth_init=None, aif_init=None, experiment_name='coord-descent', vmin=None, vmax=None, remove_outliers=False, diff_thresh=2, num_Z=100, verbose=True, windowed_mse=False):
    """Run alternating minimization on depth and AIF.

    Parameters
    ----------
    defocus_stack : ndarray, shape (fs, W, H, C)
        Observed focal stack.
    experiment_folder : str
        Root directory for saving outputs.
    gss_tol : float
        Convergence tolerance for golden-section search.
    gss_window : int
        Half-width (in grid steps) of the GSS initial bracket.
    T_0 : int
        Initial FISTA iteration budget.
    alpha : float or None
        If set, multiply T_i by alpha each epoch (decay schedule).
    num_epochs : int
        Number of alternating-minimization iterations.
    nesterov_first : bool
        If True, run the AIF step before the depth step each iteration.
    save_plots : bool
        Save intermediate images and loss curves to disk.
    show_plots : bool
        Display intermediate images interactively.
    depth_init : ndarray, shape (W, H), scalar, or None
        Initial depth map.  None → all-ones; scalar → constant map.
    aif_init : ndarray, shape (W, H, C) or None
        Initial AIF.  Only used when nesterov_first=False.
    experiment_name : str
        Subdirectory name for this run's outputs.
    vmin, vmax : float or None
        Depth colormap range for saved plots.
    remove_outliers : bool
        Apply outlier removal to the final depth map.
    diff_thresh : float
        Threshold parameter for outlier removal.
    num_Z : int
        Number of depth candidates for the coarse grid search.
    verbose : bool
        Print progress messages.
    windowed_mse : bool
        Use spatially-windowed MSE in the grid search.

    Returns
    -------
    dpt : ndarray, shape (W, H)
        Final estimated depth map.
    aif : ndarray, shape (W, H, C)
        Final estimated all-in-focus image.
    T_i : int
        Final FISTA iteration budget (after any decay).
    experiment_folder : str
        Path to the experiment output directory.
    """

    # important initializations ---------------------

    min_Z = globals.min_Z
    max_Z = globals.max_Z
    print('Depth range: [',min_Z,'-',max_Z,']')
    if vmin is None:
        vmin = min_Z
    if vmax is None:
        vmax = max_Z
    
    EXPERIMENT_NAME = experiment_name
    experiment_folder = utils.create_experiment_folder(EXPERIMENT_NAME, base_folder=experiment_folder)

    losses = []
    T_i = T_0
    
    fs = defocus_stack.shape[0]     # defocus stack size
    width = defocus_stack.shape[1]
    height = defocus_stack.shape[2]
    IMAGE_RANGE = 255. # if defocus stack in [0-255]
    if defocus_stack.max() <= 1.5: # instead in [0-1]
        IMAGE_RANGE = 1.
        if verbose:
            print('Images in range [0-1]')
    elif verbose:
        print('Images in range [0-255]')
    
    # precompute indices + sparse matrix stack 
    indices = forward_model.precompute_indices(width, height)
    sample_data = np.random.rand(indices[2].size).astype(np.float32)
    template_A_stack = forward_model.build_fixed_pattern_csr(width, height, fs, indices[2], indices[3], sample_data)

    # ------------------------------------------------
    
    def generate_AIF(aif, dpt, ls_maxiter, iter_folder=None):
        # ------------------------------------------------------
        # bounded FISTA (Nesterov's accelerated gradient method)
        # ------------------------------------------------------
        t0 = time.time()
        aif = nesterov.bounded_fista_3d(dpt, defocus_stack, IMAGE_RANGE, indices=indices, template_A_stack=template_A_stack, maxiter=T_i, verbose=verbose)
        if verbose:
            print('FISTA duration', time.time()-t0)
        
        if verbose:
            print('\nAIF result range: [',aif.min(), ',', aif.max(),']')
        
        loss = mse_loss(forward_model.forward(dpt, aif, indices=indices, template_A_stack=template_A_stack), defocus_stack)
        losses.append(loss)
        if verbose:
            print('Loss:',loss, ', TV:',outlier_removal.total_variation(aif))

        if save_plots or show_plots:
            plt.imshow(aif / IMAGE_RANGE)
            if save_plots:
                plt.savefig(os.path.join(iter_folder,'bounded_fista_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        return aif, dpt

    def generate_DPT(aif, dpt, iter_folder=None, last_dpt=None):
        # ---------------------------
        # grid + backtracking search
        # ---------------------------
        t0 = time.time()

        depth_map, Z, min_indices, all_losses = section_search.grid_search(aif, defocus_stack, indices=indices, min_Z=min_Z, max_Z=max_Z, num_Z=num_Z, verbose=verbose, windowed=windowed_mse)

        if verbose:
            print('GRID SEARCH DURATION', time.time()-t0)

        if save_plots or show_plots:
            plt.imshow(depth_map)
            plt.colorbar()
            if save_plots:
                plt.savefig(os.path.join(iter_folder,'grid_search_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # GSS
        t0 = time.time()
        depth_map_golden = section_search.golden_section_search(Z, min_indices, aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, window=gss_window, tolerance=gss_tol, last_dpt=last_dpt, verbose=verbose, windowed=False)
        if verbose:
            print('GSS DURATION', time.time()-t0)
        
        if save_plots or show_plots:
            plt.imshow(depth_map_golden, vmin=vmin, vmax=vmax)
            plt.colorbar()
            if save_plots:
                plt.savefig(os.path.join(iter_folder,'golden_section_search_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        dpt = depth_map_golden
        
        
        loss = mse_loss(forward_model.forward(dpt, aif, indices=indices, template_A_stack=template_A_stack), defocus_stack)
        losses.append(loss)

        if verbose:
            print('Loss:',loss,', TV:',outlier_removal.total_variation(dpt))
            print('\nDPT result range: [',dpt.min(), ',', dpt.max(),']')

        
        if verbose:
            print()
            print()
        return aif, dpt

    
    # -------------------------------------------------------------------------------------
    # Initialization
    if depth_init is None:
        dpt = np.ones((width, height), dtype=np.float32)
    elif np.isscalar(depth_init):
        dpt = np.ones((width, height), dtype=np.float32) * depth_init
    else:
        dpt = np.array(depth_init, dtype=np.float32)
    
    if not nesterov_first and aif_init is None:
        if verbose:
            print('initializing aif')
        aif = initialization.compute_aif_initialization(defocus_stack, lmbda=0.05, sharpness_measure='sobel_grad')
    else:
        aif = aif_init
        
    # -------------------------------------------------------------------------------------

    
    if show_plots:
        if nesterov_first:
            plt.imshow(dpt)
            plt.title('DPT Initialization')
            plt.show()
        else:
            plt.imshow(aif / IMAGE_RANGE)
            plt.title('AIF Initialization')
            plt.show()

    
    last_dpt = None

    last_loss = float('inf')
    for i in range(num_epochs):
        
        if verbose:
            print('Iteration',i,'\n')
        
        if save_plots:
            iter_folder = os.path.join(experiment_folder,'iteration'+str(i))
            os.makedirs(iter_folder)
        else:
            iter_folder = None

        if i > 0:
            last_loss = mse_loss(forward_model.forward(dpt, aif, indices=indices, template_A_stack=template_A_stack), defocus_stack)

        t0 = time.time()
        if nesterov_first:
            aif, dpt = generate_AIF(aif, dpt, T_i, iter_folder=iter_folder)
            aif, dpt = generate_DPT(aif, dpt, iter_folder=iter_folder, last_dpt=last_dpt)
        else:
            aif, dpt = generate_DPT(aif, dpt, iter_folder=iter_folder, last_dpt=last_dpt)
            aif, dpt = generate_AIF(aif, dpt, T_i, iter_folder=iter_folder)
        if verbose:
            print('FULL ITER DURATION', time.time()-t0)
        
        # save images themselves
        if save_plots:
            utils.save_dpt(iter_folder, 'dpt_'+str(i), dpt)
            utils.save_aif(iter_folder, 'aif_'+str(i), aif / IMAGE_RANGE)
        
        # if similarity_penalty:
        last_dpt = np.copy(dpt)#torch.clone(dpt.detach())

        if alpha != None:
            T_i = int(T_i * alpha)
            if verbose:
                print('T_i updated to',T_i)
            # gss_tol = gss_tol / ls_maxiter_multiplier
            # print('gss_tol updated to', gss_tol)

        # Early stopping criterion
        loss = mse_loss(forward_model.forward(dpt, aif, indices=indices, template_A_stack=template_A_stack), defocus_stack)
        # if loss > 1.05 * last_loss: # 5% greater
        #     print('Loss has started to diverge, terminated early at',i,'iters')
        #     break
        
        

        # loss plot
        x = np.arange(len(losses))
        dx = int(len(losses) / (i+1))
        
        if save_plots:
            plt.figure(figsize=(8, 4))
            plt.plot(x, losses)
            plt.scatter(x[1::dx], losses[1::dx], color='red', marker='x', s=100, label="clipped aif")
            plt.xticks(x[::dx], labels=np.arange(len(x))[::dx]) # 6 pts per iter
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Coordinate Descent")
            plt.savefig(os.path.join(experiment_folder,'loss.png'))
            plt.close()
    
            plt.figure(figsize=(8, 4))
            # x = np.arange(len(losses))
            plt.plot(x, [math.log(loss, 10) for loss in losses])
            plt.scatter(x[1::dx], [math.log(loss, 10) for loss in losses[1::dx]], color='red', marker='x', s=100, label="clipped aif")
            plt.xticks(x[::dx], labels=np.arange(len(x))[::dx]) # 6 pts per iter
            plt.xlabel("Iteration")
            plt.ylabel("log(Loss)")
            plt.title("Coordinate Descent")
            plt.savefig(os.path.join(experiment_folder,'log_loss.png'))
            plt.close()

        with open(os.path.join(experiment_folder,"losses.txt"), "w") as file:
            for j in range(len(losses)):
                if j % dx == 0:
                    file.write("iter "+str(j // dx)+":\n")
                item = losses[j]
                file.write(f"{item}\n")
    
        if verbose:
            print()
            print()
            print('--------------------------')
            print()


    # OPTIONAL -- remove outliers
    if remove_outliers:
        dpt, _ = outlier_removal.remove_outliers(dpt, aif, diff_thresh = diff_thresh, tv_thresh = tv_thresh, patch_type = outlier_patch_type)
        if save_plots or show_plots:
            plt.imshow(dpt, vmin=vmin, vmax=vmax)
            plt.colorbar()
            if save_plots:
                plt.savefig(os.path.join(iter_folder,'oulier_removal_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()
        
    return dpt, aif, T_i, experiment_folder

