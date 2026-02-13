"""
Mobile Depth Dataset Processing Pipeline

This script runs the coordinate descent depth-from-defocus algorithm on images
from the Mobile Depth dataset. It performs the following steps:
1. Load defocus stack from the Mobile Depth dataset
2. Initialize all-in-focus (AIF) image using MRF optimization
3. Run coordinate descent to jointly optimize depth map and AIF image
4. Save results to experiment folder

Usage:
    python run_coordinate_descent_mobile_depth.py <example_name>

Valid example names:
    keyboard, bottles, fruits, metals, plants, telephone, window,
    largemotion, smallmotion, zeromotion, balls
"""

import sys
import numpy as np

import utils
import globals
import coordinate_descent
import initialization
import dataset_loader
from config import MOBILE_DEPTH


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_preprocess_image(example_name, config):
    """
    Load and preprocess a defocus stack from the Mobile Depth dataset.

    Args:
        example_name: Name of the example to load (e.g., "keyboard", "bottles")
        config: Configuration object containing dataset parameters

    Returns:
        Preprocessed defocus stack of shape with edge padding applied
    """
    # Validate example name
    assert example_name in config.valid_examples, \
        f"Invalid example name. Must be one of: {config.valid_examples}"

    # Initialize dataset-specific globals
    globals.init_MobileDepth()
    globals.window_size = config.window_size

    # Load defocus stack from dataset
    defocus_stack, dpt_result, scale_mat = dataset_loader.load_single_sample_MobileDepth(example_name, data_dir=config.data_dir)

    # Apply edge padding based on window size
    rad = config.window_size // 2
    defocus_stack = np.stack([
        np.pad(img, ((rad, rad), (rad, rad), (0, 0)), mode='edge')
        for img in defocus_stack
    ], axis=0)

    # Set adaptive kernel size based on image dimensions
    _, width, height, _ = defocus_stack.shape
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    print(f'Adaptive kernel size set to {max_kernel_size}')
    utils.update_max_kernel_size(max_kernel_size)

    return defocus_stack



# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Run the Mobile Depth depth-from-defocus pipeline."""
    # Parse command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python run_coordinate_descent_mobile_depth.py <example_name>")
        print(f"Valid examples: {', '.join(MOBILE_DEPTH.valid_examples)}")
        sys.exit(1)

    example_name = sys.argv[1]
    config = MOBILE_DEPTH

    # Generate experiment name
    experiment_name = config.get_experiment_name(example_name)
    print(f"Running experiment: {experiment_name}")

    # Load and preprocess defocus stack
    print("Loading defocus stack...")
    defocus_stack = load_and_preprocess_image(example_name, config)

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

    print(f"Results saved to: {exp_folder}")


if __name__ == "__main__":
    main()
