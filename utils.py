"""
Utility functions for depth-from-defocus analysis.

This module provides a collection of utility functions organized into:
- Mathematical/Statistical Functions: metrics for depth estimation evaluation
- Image Processing Functions: image format conversion utilities
- Visualization/Plotting Functions: plotting and analysis visualizations
- File I/O Functions: reading/writing depth maps and images
- General Utility Functions: experiment management and helpers
"""

import os
import math
import struct
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import skimage

import globals


# ============================================================================
# Mathematical/Statistical Functions
# ============================================================================

def total_variation(image):
    """
    Calculate the total variation of an image.

    Total variation is the sum of absolute gradients in both x and y directions.
    It's often used as a regularization term to encourage smooth solutions.

    Args:
        image: 2D numpy array

    Returns:
        float: Scalar total variation value
    """
    tv_x = np.abs(image[:, 1:] - image[:, :-1])  # Horizontal gradients
    tv_y = np.abs(image[1:, :] - image[:-1, :])  # Vertical gradients
    return np.sum(tv_x) + np.sum(tv_y)

def compute_RMS(pred, gt):
    """
    Compute Root Mean Square (RMS) error between predicted and ground truth depth.

    Args:
        pred: Predicted depth map
        gt: Ground truth depth map

    Returns:
        float: RMS error value
    """
    diff_sq = (pred - gt) ** 2
    return np.sqrt(np.mean(diff_sq))


def compute_Rel(pred, gt):
    """
    Compute relative error with absolute values.

    Calculates mean(|pred - gt| / |gt|) with epsilon for numerical stability.

    Args:
        pred: Predicted depth map
        gt: Ground truth depth map

    Returns:
        float: Mean relative error
    """
    rel = np.abs(pred - gt) / (np.abs(gt) + 1e-8)
    return np.mean(rel)


def compute_AbsRel(pred, gt):
    """
    Compute absolute relative error.

    Calculates mean(|pred - gt| / gt) with epsilon for numerical stability.

    Args:
        pred: Predicted depth map
        gt: Ground truth depth map

    Returns:
        float: Mean absolute relative error
    """
    rel = np.abs(pred - gt) / (gt + 1e-8)
    return np.mean(rel)


def compute_accuracy_metrics(pred, gt):
    """
    Compute accuracy metrics as defined by Eigen et al.

    Returns δ1, δ2, δ3 metrics where:
    δ_k = fraction of pixels with max(pred/gt, gt/pred) < 1.25^k

    Args:
        pred: Predicted depth map
        gt: Ground truth depth map

    Returns:
        dict: Dictionary containing delta1, delta2, and delta3 values
    """
    ratio = np.maximum(pred / (gt + 1e-8), gt / (pred + 1e-8))
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25**2)
    delta3 = np.mean(ratio < 1.25**3)

    return {"delta1": delta1, "delta2": delta2, "delta3": delta3}


# ============================================================================
# Visualization/Plotting Functions
# ============================================================================

def get_worst_diff_pixels(recon, gt, num_worst_pixels=5, vmin=0.7, vmax=1.9):
    """
    Identify and visualize pixels with the worst reconstruction error.

    Args:
        recon: Reconstructed depth map
        gt: Ground truth depth map
        num_worst_pixels: Number of worst pixels to identify (default: 5)
        vmin: Minimum value for color scale (default: 0.7)
        vmax: Maximum value for color scale (default: 1.9)

    Returns:
        list: List of (row, col) tuples for worst pixels
    """
    diff = np.abs(recon - gt)

    # Use argpartition for efficient k-largest selection
    worst_indices = np.argpartition(diff.ravel(), -num_worst_pixels)[-num_worst_pixels:]
    worst_indices = worst_indices[np.argsort(diff.ravel()[worst_indices])[::-1]]
    worst_coords = [(idx // diff.shape[1], idx % diff.shape[1]) for idx in worst_indices]

    # Visualize reconstruction with worst pixels marked
    plt.imshow(recon, vmin=vmin, vmax=vmax)
    plt.scatter([y for x, y in worst_coords], [x for x, y in worst_coords],
                color='red', marker='x', s=100, label='Worst Diff Pixels')
    plt.title('Worst Difference Pixels Over Image')
    plt.legend()
    plt.show()

    return worst_coords

def plot_single_stack(recon, setting, recon_max=None, title=None):
    """
    Plot a stack of images in a grid layout (useful for defocus stack).

    Args:
        recon: List of images to plot
        setting: List of settings/labels corresponding to each image (e.g., focus distances)
        recon_max: Maximum value for normalization (default: None, uses max of each image)
        title: Optional title for the figure

    Note:
        Images are arranged in a grid with 5 columns
    """
    num_images = len(recon)
    cols = 5
    rows = math.ceil(len(recon) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 3))
    if rows == 1:
        axes = np.array([axes])  # Ensure axes is always 2D for consistency
    axes = axes.flatten()

    for i in range(num_images):
        if recon_max is None:
            recon_max = recon[i].max()
        axes[i].imshow(recon[i] / recon_max)
        axes[i].axis('off')
        axes[i].set_title(f"{setting[i]:.3f}")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_stacks_side_by_side(gt, recon, setting, title=None):
    """
    Plot ground truth and reconstructed image stacks side by side.

    Args:
        gt: List of ground truth images
        recon: List of reconstructed images
        setting: List of settings corresponding to each image pair
        title: Optional title for the figure
    """
    assert len(gt) == len(recon), "Ground truth and reconstruction stacks must have same length"

    num_images = len(gt)
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))

    for i in range(num_images):
        # Plot ground truth
        axes[i, 0].imshow(gt[i] / gt[i].max())
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Ground Truth - {setting[i]}")

        # Plot reconstruction
        axes[i, 1].imshow(recon[i] / recon[i].max())
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Reconstructed - {setting[i]}")

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()




def plot_compare_rgb(recon, gt):
    """
    Compare reconstructed and ground truth RGB images side by side.

    Args:
        recon: Reconstructed RGB image
        gt: Ground truth RGB image
    """
    recon_plot = to_uint8(recon)
    gt_plot = to_uint8(gt)

    # Plot the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot reconstructed image
    axes[0].imshow(recon_plot)
    axes[0].axis('off')
    axes[0].set_title("Recon")

    # Plot ground truth image
    axes[1].imshow(gt_plot)
    axes[1].axis('off')
    axes[1].set_title("GT")

    plt.tight_layout()
    plt.show()

def plot_compare_greyscale(recon, gt, vmin=None, vmax=None):
    """
    Compare reconstructed and ground truth grayscale images side by side.

    Args:
        recon: Reconstructed grayscale image
        gt: Ground truth grayscale image
        vmin: Minimum value for color scale (default: None, uses min of both images)
        vmax: Maximum value for color scale (default: None, uses max of both images)
    """
    if vmin is None:
        vmin = min(recon.min(), gt.min())
    if vmax is None:
        vmax = max(recon.max(), gt.max())

    # Plot the images side by side with consistent color scale
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot reconstructed image
    axes[0].imshow(recon, vmin=vmin, vmax=vmax)
    axes[0].axis('off')
    axes[0].set_title("Recon")

    # Plot ground truth image
    axes[1].imshow(gt, vmin=vmin, vmax=vmax)
    axes[1].axis('off')
    axes[1].set_title("GT")

    plt.tight_layout()
    plt.show()

def plot_grid_search_on_pixel(i, j, Z, all_losses, gt_dpt=None):
    """
    Plot loss curve for grid search on a single pixel.

    Visualizes the MSE loss as a function of depth for a specific pixel,
    showing ground truth depth, predicted depth, and derivative information.

    Args:
        i: Row index of the pixel
        j: Column index of the pixel
        Z: Array of depth values searched
        all_losses: 3D array of losses [height, width, num_depths]
        gt_dpt: Ground truth depth map (optional)
    """
    plt.figure(figsize=(10, 5))

    # Plot loss curve for this pixel
    plt.plot(Z, all_losses[i, j, :], label='Loss Curve',
             linestyle='-', marker='.', markersize=4, color='black')

    # Mark ground truth depth if available
    if gt_dpt is not None:
        plt.scatter([gt_dpt[i, j]], [0],
                    color='red', marker='x', s=100, label='Ground Truth Depth')

    # Mark minimum loss depth(s)
    min_loss_idx = np.argmin(all_losses[i, j])
    plt.scatter([Z[min_loss_idx]], [all_losses[i, j, min_loss_idx]],
                color='green', marker='x', s=100, label='Depth with Min Loss')
    
    # Plot derivative of loss curve
    plt.plot(Z[1:] - (3 - 0.1) / 100 / 2, np.diff(all_losses[i, j]), label='diff',
             linestyle='-', marker='.', markersize=4, color='green')

    # Find and print zero crossings in derivative (potential local minima/maxima)
    diff = np.diff(all_losses[i, j])
    cross_indices = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0] + 1

    for idx in cross_indices:
        print(idx, Z[idx], diff[idx], diff[idx + 1])

    plt.xticks(Z[::2], labels=np.round(Z[::2], 2), rotation=45)
    plt.xlabel('Depth (m)')
    plt.ylabel('MSE between Predicted and Ground Truth Defocus Stack')
    plt.title(f'Pixel at ({i}, {j})')

    plt.legend()
    plt.tight_layout()
    plt.show()



# ============================================================================
# Image Processing Functions
# ============================================================================

def to_uint8(image):
    """
    Convert image to uint8 range [0, 255].

    Args:
        image: Input image as numpy array or PyTorch tensor

    Returns:
        Image clipped to valid uint8 range

    Raises:
        TypeError: If input is neither numpy array nor PyTorch tensor
    """
    if isinstance(image, np.ndarray):
        return np.clip(image.astype(int), 0, 255)
    # Note: PyTorch support requires torch to be imported
    # elif isinstance(image, torch.Tensor):
    #     return torch.clamp(image.int(), 0, 255)
    else:
        raise TypeError("Input must be a NumPy array")


# ============================================================================
# File I/O Functions
# ============================================================================

def read_bin_file(path_to_file):
    """
    Read binary image file with OpenCV-style header.
    Adapted from Suwajanakorn et al. MATLAB code.

    Reads a binary file containing:
    - 1 byte: type code
    - 4 bytes: height (int)
    - 4 bytes: width (int)
    - Remaining: image data

    Args:
        path_to_file: Path to the binary file

    Returns:
        numpy.ndarray: Image as float32 array
    """
    with open(path_to_file, "rb") as f:
        # Read header
        t = np.frombuffer(f.read(1), np.uint8)[0]     # Type code
        h = struct.unpack("i", f.read(4))[0]          # Height
        w = struct.unpack("i", f.read(4))[0]          # Width

        # Map OpenCV type code to NumPy dtype
        cv_depth = t & 7
        depth_map = {
            0: np.uint8,
            1: np.int8,
            2: np.uint16,
            3: np.int16,
            4: np.int32,
            5: np.float32,
            6: np.float64,
        }
        dtype = depth_map[cv_depth]

        # Read image data and reshape
        data = np.frombuffer(f.read(), dtype=dtype)
        mat = data.reshape(h, w)

    return mat.astype(np.float32)


def exif_to_float(tag):
    """
    Convert EXIF tag to float value.

    Args:
        tag: EXIF tag object

    Returns:
        float: Converted value, or None if tag is None
    """
    if tag is None:
        return None
    try:
        val = tag.values[0]
        return float(val.num) / float(val.den)
    except AttributeError:
        # Sometimes it's already a float or simple number
        return float(tag.values[0]) if hasattr(tag, "values") else float(tag)


def save_dpt(path, fn, dpt):
    """
    Save depth map as TIFF file. Scales depth by 1e4 before saving.

    Args:
        path: Directory path
        fn: Filename (without extension)
        dpt: Depth map as numpy array
    """
    dpt_scaled = (dpt * 1e4).astype(np.float32)
    skimage.io.imsave(os.path.join(path, fn + '.tiff'), dpt_scaled)


def save_dpt_npy(path, fn, dpt):
    """
    Save depth map as numpy .npy file.

    Args:
        path: Directory path
        fn: Filename (without extension)
        dpt: Depth map as numpy array
    """
    np.save(os.path.join(path, fn + '.npy'), dpt, allow_pickle=True)


def load_dpt_npy(path, fn):
    """
    Load depth map from numpy .npy file.

    Args:
        path: Directory path
        fn: Filename (without extension)

    Returns:
        numpy.ndarray: Loaded depth map
    """
    return np.load(os.path.join(path, fn + '.npy'), allow_pickle=True)


def save_aif(path, fn, aif):
    """
    Save all-in-focus image as TIFF file.

    Args:
        path: Directory path
        fn: Filename (without extension)
        aif: All-in-focus image as numpy array
    """
    skimage.io.imsave(os.path.join(path, fn + '.tiff'), aif.squeeze().astype(np.float32))


def load_aif(path, fn):
    """
    Load all-in-focus image from TIFF file.

    Args:
        path: Directory path
        fn: Filename (without extension)

    Returns:
        numpy.ndarray: Loaded all-in-focus image as float32
    """
    return skimage.io.imread(os.path.join(path, fn + '.tiff')).astype(np.float32)


# ============================================================================
# General Utility Functions
# ============================================================================

def create_experiment_folder(experiment_name, base_folder="experiments"):
    """
    Create a timestamped folder for experiment results.

    Args:
        experiment_name: Name of the experiment
        base_folder: Base directory for experiments (default: "experiments")

    Returns:
        str: Path to created experiment folder
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{experiment_name}_{timestamp}"
    experiment_folder = os.path.join(base_folder, folder_name)
    os.makedirs(experiment_folder)

    print(f"Created experiment folder: {experiment_folder}")
    return experiment_folder


def format_number(x):
    """
    Format number for display with appropriate precision.

    Uses scientific notation for very small numbers (< 0.001),
    otherwise uses fixed-point notation with 6 decimal places.

    Args:
        x: Number to format

    Returns:
        str: Formatted number string
    """
    return f"{x:.6e}" if abs(x) < 1e-3 else f"{x:.6f}"


def update_max_kernel_size(new_value):
    """
    Update the global maximum kernel size setting.

    Args:
        new_value: New maximum kernel size value
    """
    globals.MAX_KERNEL_SIZE = new_value


def kernel_size_heuristic(width, height):
    """
    Calculate appropriate kernel size based on image dimensions.

    Uses a heuristic of ~3.9% of average image dimension,
    with a minimum of 7 pixels. Ensures odd kernel size.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        int: Odd kernel size >= 7
    """
    size = round(0.039 * (width + height) / 2)
    size = max(7, size)
    if size % 2 == 0:
        return size + 1
    return size
