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
import section_search
import utils

def mse_loss(pred, gt):
    """Mean squared error between pred and gt."""
    return np.mean((gt - pred)**2)


def coordinate_descent(
        defocus_stack, experiment_folder='experiments', gss_tol=1e-2, gss_window=1, T_0=100,
        alpha=None, num_epochs=25, nesterov_first=True, save_plots=True, show_plots=False, save_losses=True,
        depth_init=None, aif_init=None, experiment_name='coord-descent', vmin=None, vmax=None,
        num_Z=100, verbose=True, windowed_mse=False):
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
        Final FISTA iteration budget.
    experiment_folder : str
        Path to the experiment output directory.
    """

    # ============================================================================
    # INITIALIZATION: Setup depth range, output folder, and forward model
    # ============================================================================

    # Load global depth bounds and set visualization range for plots (if not provided)
    min_Z = globals.min_Z
    max_Z = globals.max_Z
    print('Depth range: [',min_Z,'-',max_Z,']')
    if vmin is None:
        vmin = min_Z
    if vmax is None:
        vmax = max_Z

    # Create output directory for saving results and intermediate visualizations
    if save_plots or save_losses:
        experiment_folder = utils.create_experiment_folder(experiment_name, base_folder=experiment_folder)

    # Initialize tracking arrays for convergence monitoring
    losses = []              # MSE loss after each sub-problem step
    T_i = T_0                # FISTA iteration budget

    # Extract input dimensions from the focal stack
    fs = defocus_stack.shape[0]     # number of focal planes
    width = defocus_stack.shape[1]  # image width
    height = defocus_stack.shape[2] # image height

    # Auto-detect image intensity range for proper normalization
    IMAGE_RANGE = 255.  # default assumes [0-255] range
    if defocus_stack.max() <= 1.5:  # check if images are normalized to [0-1]
        IMAGE_RANGE = 1.
        if verbose:
            print('Images in range [0-1]')
    elif verbose:
        print('Images in range [0-255]')

    # CRITICAL: Precompute sparse matrix structure for the forward model
    # This builds the fixed sparsity pattern used in all forward/backward passes,
    # speeding up matrix operations throughout the optimization
    indices = forward_model.precompute_indices(width, height)
    sample_data = np.random.rand(indices[2].size).astype(np.float32)
    template_A_stack = forward_model.build_fixed_pattern_csr(width, height, fs, indices[2], indices[3], sample_data)

    # ============================================================================
    # SUB-PROBLEM SOLVERS: Define AIF and depth optimization steps
    # ============================================================================

    def generate_AIF(aif, dpt, T_i, iter_folder=None):
        """AIF step: solve for aif given dpt via bounded FISTA (Nesterov's accelerated gradient method)."""

        # Run bounded FISTA optimization to reconstruct the all-in-focus image
        # ie given the current depth map, find the sharp image that best explains the observed focal stack
        t0 = time.time()
        aif = nesterov.bounded_fista_3d(dpt, defocus_stack, IMAGE_RANGE, indices=indices, template_A_stack=template_A_stack, maxiter=T_i, verbose=verbose)
        if verbose:
            print('FISTA duration', time.time()-t0)
            print('\nAIF result range: [',aif.min(), ',', aif.max(),']')

        # Compute MSE between observed focal stack and synthesized stack
        # from current AIF and depth estimates via the forward model
        loss = mse_loss(forward_model.forward(dpt, aif, indices=indices, template_A_stack=template_A_stack), defocus_stack)
        losses.append(loss)
        if verbose:
            # TV (total variation) measures image smoothness - useful for detecting artifacts
            print('Loss:',loss, ', TV:',utils.total_variation(aif)) # TODO: move this to utils

        # Visualize the reconstructed all-in-focus image (normalized to [0,1] for display)
        if save_plots or show_plots:
            plt.imshow(aif / IMAGE_RANGE)
            if save_plots:
                assert iter_folder is not None
                plt.savefig(os.path.join(iter_folder,'bounded_fista_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        return aif, dpt

    def generate_DPT(aif, dpt, iter_folder=None, last_dpt=None):
        """Depth step: solve for dpt given aif via coarse grid search and golden-section search."""
        
        # STEP 1: Coarse grid search
        # Evaluate num_Z uniformly-spaced depth candidates across [min_Z, max_Z] to find
        # the best approximate depth value for each pixel. This gives us a rough depth map
        # and identifies promising regions for refinement.
        t0 = time.time()
        depth_map, Z, min_indices, all_losses = section_search.grid_search(
            aif, defocus_stack, indices=indices, min_Z=min_Z, max_Z=max_Z, num_Z=num_Z,
            verbose=verbose, windowed=windowed_mse
        )
        if verbose:
            print('GRID SEARCH DURATION', time.time()-t0)

        # Visualize the coarse depth estimates from grid search
        if save_plots or show_plots:
            plt.imshow(depth_map)
            plt.colorbar()
            if save_plots:
                assert iter_folder is not None
                plt.savefig(os.path.join(iter_folder,'grid_search_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # STEP 2: Golden-section search (GSS) refinement
        # Refine each pixel's depth estimate to sub-grid accuracy using golden-section search.
        t0 = time.time()
        depth_map_golden = section_search.golden_section_search(
            Z, min_indices, aif, defocus_stack, indices=indices, template_A_stack=template_A_stack,
            window=gss_window, tolerance=gss_tol, last_dpt=last_dpt, verbose=verbose, windowed=False
        )
        if verbose:
            print('GSS DURATION', time.time()-t0)

        # Visualize the refined depth map with consistent color scale across iterations
        if save_plots or show_plots:
            plt.imshow(depth_map_golden, vmin=vmin, vmax=vmax)
            plt.colorbar()
            if save_plots:
                plt.savefig(os.path.join(iter_folder,'golden_section_search_'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        # Use the refined depth map as our final estimate for this iteration
        dpt = depth_map_golden

        # Compute reconstruction loss: measure how well our current depth and AIF estimates
        # reconstruct the observed focal stack via the forward model
        loss = mse_loss(forward_model.forward(dpt, aif, indices=indices, template_A_stack=template_A_stack), defocus_stack)
        losses.append(loss)

        if verbose:
            # TV (total variation) measures depth map smoothness - helps detect noisy regions
            print('Loss:',loss,', TV:',utils.total_variation(dpt))
            print('\nDPT result range: [',dpt.min(), ',', dpt.max(),']')


        if verbose:
            print()
            print()
        return aif, dpt


    # ============================================================================
    # INITIALIZATION: Set starting values for depth map and all-in-focus image
    # ============================================================================

    # Initialize depth map (dpt): (nesterov_first=True)
    # - None → uniform depth (all-ones), assumes scene at constant depth initially
    # - Scalar → constant depth map with user-specified value
    # - Array → use provided depth initialization
    if nesterov_first:
        if depth_init is None:
            dpt = np.ones((width, height), dtype=np.float32)
        elif np.isscalar(depth_init):
            dpt = np.ones((width, height), dtype=np.float32) * depth_init
        else:
            dpt = np.array(depth_init, dtype=np.float32)
    else:
        dpt = None

    # Initialize all-in-focus image (aif): (nesterov_first=False)
    # Use sharpness-based fusion (based on AIF stitching from Suwajanakorn et al.)
    # to combine the sharpest regions from each focal plane.
    if not nesterov_first:
        if aif_init is None:
            if verbose:
                print('initializing aif')
            aif = initialization.compute_aif_initialization(defocus_stack, lmbda=0.05, sharpness_measure='sobel_grad')
        else:
            aif = aif_init
    else:
        aif = None

    # Display the initial estimates for visual verification
    if show_plots:
        if nesterov_first:
            plt.imshow(dpt)
            plt.title('DPT Initialization')
            plt.show()
        else:
            plt.imshow(aif / IMAGE_RANGE)
            plt.title('AIF Initialization')
            plt.show()

    # ============================================================================
    # MAIN ALTERNATING MINIMIZATION LOOP
    # ============================================================================
    # Alternates between two optimization steps until max iterations:
    # 1. AIF step: Given current depth, solve for all-in-focus image (via bounded FISTA)
    # 2. Depth step: Given current AIF, solve for depth map (via grid + golden-section search)
    #
    # Each iteration refines both estimates. The order can be controlled via nesterov_first.

    last_dpt = None  # Track previous depth estimate
    for i in range(num_epochs):

        if verbose:
            print('Iteration',i,'\n')

        # Setup output directory for this iteration's intermediate results
        if save_plots:
            iter_folder = os.path.join(experiment_folder,'iteration'+str(i))
            os.makedirs(iter_folder)
        else:
            iter_folder = None

        # ---------------------------------------------------------------------
        # ALTERNATING MINIMIZATION: Solve two sub-problems in sequence
        # ---------------------------------------------------------------------
        # Each sub-problem fixes one variable and optimizes the other.
        t0 = time.time()
        if nesterov_first:
            # AIF → Depth
            aif, dpt = generate_AIF(aif, dpt, T_i, iter_folder=iter_folder)
            aif, dpt = generate_DPT(aif, dpt, iter_folder=iter_folder, last_dpt=last_dpt)
        else:
            # Depth → AIF
            aif, dpt = generate_DPT(aif, dpt, iter_folder=iter_folder, last_dpt=last_dpt)
            aif, dpt = generate_AIF(aif, dpt, T_i, iter_folder=iter_folder)
        if verbose:
            print('FULL ITER DURATION', time.time()-t0)

        # Save intermediate depth maps and AIF images for visualization and debugging
        if save_plots:
            utils.save_dpt(iter_folder, 'dpt_'+str(i), dpt)
            utils.save_aif(iter_folder, 'aif_'+str(i), aif / IMAGE_RANGE)

        # Store current depth estimate
        last_dpt = np.copy(dpt)

        # Decay schedule for FISTA iteration budget (if alpha is provided)
        if alpha is not None:
            T_i = int(T_i * alpha)
            if verbose:
                print('T_i updated to',T_i)

        # ---------------------------------------------------------------------
        # CONVERGENCE MONITORING: Generate and save loss curves
        # ---------------------------------------------------------------------
        # Plot both linear and log-scale loss to visualize convergence behavior.
        x = np.arange(len(losses))
        dx = int(len(losses) / (i+1))
        
        if save_plots:
            plt.figure(figsize=(8, 4))
            plt.plot(x, losses)
            plt.scatter(x[1::dx], losses[1::dx], color='red', marker='x', s=100, label="clipped aif")
            plt.xticks(x[::dx], labels=np.arange(len(x))[::dx])
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Coordinate Descent")
            plt.savefig(os.path.join(experiment_folder,'loss.png'))
            plt.close()
    
            plt.figure(figsize=(8, 4))
            plt.plot(x, [math.log(loss, 10) for loss in losses])
            plt.scatter(x[1::dx], [math.log(loss, 10) for loss in losses[1::dx]], color='red', marker='x', s=100, label="clipped aif")
            plt.xticks(x[::dx], labels=np.arange(len(x))[::dx])
            plt.xlabel("Iteration")
            plt.ylabel("log(Loss)")
            plt.title("Coordinate Descent")
            plt.savefig(os.path.join(experiment_folder,'log_loss.png'))
            plt.close()

        if save_losses:
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

    return dpt, aif, T_i, experiment_folder

