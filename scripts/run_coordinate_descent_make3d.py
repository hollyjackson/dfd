"""
Make3D Dataset Processing Pipeline

This script runs the coordinate descent depth-from-defocus algorithm on images
from the Make3D dataset. It performs the following steps:
1. Load ground truth AIF and depth from Make3D dataset
2. Generate defocus stack using forward model
3. Initialize AIF image using MRF optimization
4. Run coordinate descent to jointly optimize depth map and AIF image
5. Compute and save accuracy metrics

Usage:
    python scripts/run_coordinate_descent_make3d.py <split> <img_name>

    split: 'train' or 'test'
    img_name: Image filename (e.g., 'img-math7-p-282t0.jpg')
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np

import utils
import forward_model
import globals
import coordinate_descent
import initialization
import dataset_loader
from config import MAKE3D


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_generate_defocus_stack(img_name, split, config):
    """
    Load Make3D ground truth and generate synthetic defocus stack.

    Args:
        img_name: Image filename (e.g., 'img-math7-p-282t0.jpg')
        split: Dataset split ('train' or 'test')
        config: Configuration object containing dataset parameters

    Returns:
        Tuple of (defocus_stack, gt_aif, gt_dpt) where:
            - defocus_stack: Synthetic defocus stack
            - gt_aif: Ground truth all-in-focus image
            - gt_dpt: Ground truth depth map
    """
    # Initialize dataset-specific globals
    globals.init_Make3D()
    globals.window_size = config.window_size

    # Load ground truth AIF and depth
    gt_aif, gt_dpt = dataset_loader.load_single_sample_Make3D(
        img_name=img_name, split=split, data_dir=config.data_dir
    )

    # Set adaptive kernel size based on image dimensions
    width, height, _ = gt_aif.shape
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    print(f'Adaptive kernel size set to {max_kernel_size}')
    utils.update_max_kernel_size(max_kernel_size)

    # Generate synthetic defocus stack using forward model
    defocus_stack = forward_model.forward(gt_dpt, gt_aif)

    return defocus_stack, gt_aif, gt_dpt



# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    """Run the Make3D depth-from-defocus pipeline."""
    # Parse command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python scripts/run_coordinate_descent_make3d.py <split> <img_name>")
        print("  split: 'train' or 'test'")
        print("  img_name: Image filename (e.g., 'img-math7-p-282t0.jpg')")
        sys.exit(1)

    split = sys.argv[1]
    img_name = sys.argv[2]
    config = MAKE3D

    # Generate experiment name
    experiment_name = config.get_experiment_name(split, img_name)
    print(f"Running experiment: {experiment_name}")

    # Load ground truth and generate defocus stack
    print("Loading ground truth and generating defocus stack...")
    defocus_stack, gt_aif, gt_dpt = load_and_generate_defocus_stack(img_name, split, config)

    # Compute AIF initialization
    print("Computing AIF initialization...")
    aif_init = initialization.compute_aif_initialization(
        defocus_stack,
        lmbda=config.aif_lambda,
        sharpness_measure=config.aif_sharpness_measure
    )

    # Run coordinate descent optimization
    print("Running coordinate descent optimization...")
    dpt, aif, _, exp_folder = coordinate_descent.coordinate_descent(
        defocus_stack,
        experiment_folder=config.experiment_folder,
        show_plots=config.show_plots,
        save_plots=config.save_plots,
        experiment_name=experiment_name,
        num_epochs=config.num_epochs,
        nesterov_first=config.nesterov_first,
        aif_init=aif_init,
        num_Z=config.num_z,
        T_0=config.t_0,
        alpha=config.alpha,
        verbose=config.verbose,
        windowed_mse=config.use_windowed_mse
    )

    # Save final results
    print("Saving results...")
    utils.save_dpt_npy(exp_folder, 'dpt', dpt)
    utils.save_aif(exp_folder, 'aif', aif)

    # Compute and save accuracy metrics
    print("Computing accuracy metrics...")
    rms = utils.compute_RMS(dpt, gt_dpt)
    rel = utils.compute_AbsRel(dpt, gt_dpt)
    deltas = utils.compute_accuracy_metrics(dpt, gt_dpt)

    outfile = os.path.join(exp_folder, "accuracy_metrics.txt")
    delta_str = ", ".join(f"{float(deltas[d]):.6f}" for d in sorted(deltas.keys()))
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"RMS: {float(rms):.6f}\n")
        f.write(f"Rel: {float(rel):.6f}\n")
        f.write(f"Accuracy (δ1, δ2, δ3): {delta_str}\n")

    print(f"Results saved to: {exp_folder}")
    print(f"RMS: {float(rms):.6f}, Rel: {float(rel):.6f}")


if __name__ == "__main__":
    main()