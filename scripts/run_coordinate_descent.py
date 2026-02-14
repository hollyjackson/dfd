"""
Unified Depth-from-Defocus Pipeline

This script runs the coordinate descent depth-from-defocus algorithm on images
from Make3D, Mobile Depth, or NYUv2 datasets.

Usage:
    # Make3D
    python scripts/run_coordinate_descent.py make3d <split> <img_name> [--verbose]

    # Mobile Depth
    python scripts/run_coordinate_descent.py mobiledepth <example_name> [--verbose]

    # NYUv2
    python scripts/run_coordinate_descent.py nyuv2 <split> <image_number> [--verbose]

Options:
    --verbose      Print configuration and dataset parameters
    --show_plots   Display plots interactively during optimization
    --save_plots   Save plots to experiment folder

Examples:
    python scripts/run_coordinate_descent.py make3d train img-math7-p-282t0.jpg
    python scripts/run_coordinate_descent.py mobiledepth keyboard --verbose
    python scripts/run_coordinate_descent.py nyuv2 test 123 --save_plots
    python scripts/run_coordinate_descent.py make3d test img-op29-p-295t000.jpg --verbose --show_plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np

import utils
import forward_model
import coordinate_descent
import initialization
import dataset_loader
from dataset_params import DatasetParams
from config import MAKE3D, MOBILE_DEPTH, NYUV2


# =============================================================================
# Dataset-Specific Loading Functions
# =============================================================================

def load_make3d(img_name, split, config, verbose=False):
    """Load Make3D ground truth and generate synthetic defocus stack."""
    dataset_params = DatasetParams.for_Make3D()

    # Load ground truth AIF and depth
    gt_aif, gt_dpt = dataset_loader.load_single_sample_Make3D(
        img_name=img_name, dataset_params=dataset_params,
        split=split, data_dir=config.data_dir, verbose=verbose
    )

    # Set adaptive kernel size
    width, height, _ = gt_aif.shape
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    if verbose:
        print(f'Adaptive kernel size set to {max_kernel_size}')

    # Generate synthetic defocus stack
    defocus_stack = forward_model.forward(gt_dpt, gt_aif, dataset_params, max_kernel_size)

    return defocus_stack, dataset_params, max_kernel_size, gt_aif, gt_dpt


def load_mobiledepth(example_name, config, verbose=False):
    """Load and preprocess defocus stack from Mobile Depth dataset."""
    # Validate example name
    assert example_name in config.valid_examples, \
        f"Invalid example name. Must be one of: {config.valid_examples}"

    dataset_params = DatasetParams.for_MobileDepth()

    # Load defocus stack
    defocus_stack, dpt_result, scale_mat = dataset_loader.load_single_sample_MobileDepth(
        example_name, dataset_params=dataset_params, data_dir=config.data_dir, verbose=verbose
    )

    # Apply edge padding
    rad = config.window_size // 2
    defocus_stack = np.stack([
        np.pad(img, ((rad, rad), (rad, rad), (0, 0)), mode='edge')
        for img in defocus_stack
    ], axis=0)

    # Set adaptive kernel size
    _, width, height, _ = defocus_stack.shape
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    if verbose:
        print(f'Adaptive kernel size set to {max_kernel_size}')

    return defocus_stack, dataset_params, max_kernel_size, None, None


def load_nyuv2(image_number, split, config, verbose=False):
    """Load NYUv2 ground truth and generate synthetic defocus stack."""
    dataset_params = DatasetParams.for_NYUv2()

    # Load ground truth AIF and depth
    gt_aif, gt_dpt = dataset_loader.load_single_sample_NYUv2(
        sample=image_number, set=split, res='half',
        data_dir=config.data_dir, verbose=verbose
    )

    # Set adaptive kernel size
    width, height = gt_dpt.shape
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    if verbose:
        print(f'Adaptive kernel size set to {max_kernel_size}')

    # Generate synthetic defocus stack
    defocus_stack = forward_model.forward(gt_dpt, gt_aif, dataset_params, max_kernel_size)

    return defocus_stack, dataset_params, max_kernel_size, gt_aif, gt_dpt


# =============================================================================
# Main Pipeline
# =============================================================================

def save_accuracy_metrics(exp_folder, dpt, gt_dpt, verbose=False):
    """Compute and save accuracy metrics."""
    rms = utils.compute_RMS(dpt, gt_dpt)
    rel = utils.compute_AbsRel(dpt, gt_dpt)
    deltas = utils.compute_accuracy_metrics(dpt, gt_dpt)

    outfile = os.path.join(exp_folder, "accuracy_metrics.txt")
    delta_str = ", ".join(f"{float(deltas[d]):.6f}" for d in sorted(deltas.keys()))
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"RMS: {float(rms):.6f}\n")
        f.write(f"Rel: {float(rel):.6f}\n")
        f.write(f"Accuracy (δ1, δ2, δ3): {delta_str}\n")

    if verbose:
        print(f"RMS: {float(rms):.6f}, Rel: {float(rel):.6f}")


def print_config(config, dataset_params):
    """Print configuration and dataset parameters."""
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    for key in dir(config):
        if not key.startswith('_') and not callable(getattr(config, key)):
            print(f"  {key}: {getattr(config, key)}")

    print("\n" + "="*60)
    print("Dataset Parameters:")
    print("="*60)
    for key in ['f', 'D', 'ps', 'Zf']:
        if hasattr(dataset_params, key):
            value = getattr(dataset_params, key)
            print(f"  {key}: {value}")
    print("="*60 + "\n")


def main():
    """Run the unified depth-from-defocus pipeline."""
    # Parse dataset type and check for optional flags
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Check for optional flags
    verbose_mode = '--verbose' in sys.argv
    if verbose_mode:
        sys.argv.remove('--verbose')

    show_plots = '--show_plots' in sys.argv
    if show_plots:
        sys.argv.remove('--show_plots')

    save_plots = '--save_plots' in sys.argv
    if save_plots:
        sys.argv.remove('--save_plots')

    dataset = sys.argv[1].lower()

    # Dataset-specific argument parsing and loading
    if dataset == 'make3d':
        if len(sys.argv) != 4:
            print("Usage: python scripts/run_coordinate_descent.py make3d <split> <img_name>")
            print("  split: 'train' or 'test'")
            print("  img_name: Image filename (e.g., 'img-math7-p-282t0.jpg')")
            sys.exit(1)

        split = sys.argv[2]
        img_name = sys.argv[3]
        config = MAKE3D
        experiment_name = config.get_experiment_name(split, img_name)

        if verbose_mode:
            print(f"Running Make3D experiment: {experiment_name}")
            print("Loading ground truth and generating defocus stack...")
        defocus_stack, dataset_params, max_kernel_size, gt_aif, gt_dpt = load_make3d(img_name, split, config, verbose=verbose_mode)

    elif dataset == 'mobiledepth':
        if len(sys.argv) != 3:
            print("Usage: python scripts/run_coordinate_descent.py mobiledepth <example_name>")
            print(f"Valid examples: {', '.join(MOBILE_DEPTH.valid_examples)}")
            sys.exit(1)

        example_name = sys.argv[2]
        config = MOBILE_DEPTH
        experiment_name = config.get_experiment_name(example_name)

        if verbose_mode:
            print(f"Running Mobile Depth experiment: {experiment_name}")
            print("Loading defocus stack...")
        defocus_stack, dataset_params, max_kernel_size, gt_aif, gt_dpt = load_mobiledepth(example_name, config, verbose=verbose_mode)

    elif dataset == 'nyuv2':
        if len(sys.argv) != 4:
            print("Usage: python scripts/run_coordinate_descent.py nyuv2 <split> <image_number>")
            print("  split: 'train' or 'test'")
            print("  image_number: Image number from the dataset")
            sys.exit(1)

        split = sys.argv[2]
        image_number = sys.argv[3]
        config = NYUV2
        experiment_name = config.get_experiment_name(split, image_number)

        if verbose_mode:
            print(f"Running NYUv2 experiment: {experiment_name}")
            print("Loading ground truth and generating defocus stack...")
        defocus_stack, dataset_params, max_kernel_size, gt_aif, gt_dpt = load_nyuv2(image_number, split, config, verbose=verbose_mode)

    else:
        print(f"Unknown dataset: {dataset}")
        print("Supported datasets: make3d, mobiledepth, nyuv2")
        sys.exit(1)

    # Print configuration and dataset parameters if verbose
    if verbose_mode:
        print_config(config, dataset_params)

    # Common pipeline for all datasets

    # Compute AIF initialization
    if verbose_mode:
        print("Computing AIF initialization...")
    aif_init = initialization.compute_aif_initialization(
        defocus_stack,
        lmbda=config.aif_lambda,
        sharpness_measure=config.aif_sharpness_measure
    )

    # Run coordinate descent optimization
    if verbose_mode:
        print("Running coordinate descent optimization...")
    cd_params = {
        'defocus_stack': defocus_stack,
        'dataset_params': dataset_params,
        'max_kernel_size': max_kernel_size,
        'experiment_folder': config.experiment_folder,
        'show_plots': show_plots or config.show_plots,  # Override with --show_plots flag
        'save_plots': save_plots or config.save_plots,  # Override with --save_plots flag
        'experiment_name': experiment_name,
        'num_epochs': config.num_epochs,
        'nesterov_first': config.nesterov_first,
        'aif_init': aif_init,
        'num_Z': config.num_z,
        'T_0': config.t_0,
        'alpha': config.alpha,
        'verbose': verbose_mode,  # Override config.verbose with --verbose flag
        'windowed_mse': config.use_windowed_mse,
    }

    # Add window_size parameter if it exists in config
    if hasattr(config, 'window_size'):
        cd_params['window_size'] = config.window_size

    dpt, aif, _, exp_folder = coordinate_descent.coordinate_descent(**cd_params)

    # Save final results
    if verbose_mode:
        print("Saving results...")
    if dataset == 'nyuv2':
        utils.save_dpt(exp_folder, 'dpt', dpt)
    else:
        utils.save_dpt_npy(exp_folder, 'dpt', dpt)
    utils.save_aif(exp_folder, 'aif', aif)

    # Compute and save accuracy metrics (if ground truth is available)
    if gt_dpt is not None:
        if verbose_mode:
            print("Computing accuracy metrics...")
        save_accuracy_metrics(exp_folder, dpt, gt_dpt, verbose=verbose_mode)

    if verbose_mode:
        print(f"Results saved to: {exp_folder}")


if __name__ == "__main__":
    main()
