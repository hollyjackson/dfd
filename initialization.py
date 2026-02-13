"""
All-in-Focus (AIF) Image Initialization Module

This module provides methods for generating all-in-focus images from defocus stacks.
Two approaches are implemented:
1. Trivial initialization: Select the sharpest pixel from each image in the stack
2. MRF-based initialization: Use Markov Random Field optimization with graph cuts
   to enforce spatial smoothness while maximizing local sharpness (based on AIF
   stitching algorithm from Suwajanakorn et al.)
"""

import numpy as np
import cv2
import scipy
import gco
import utils
import globals


# =============================================================================
# Trivial AIF Initialization (Pixel-wise Selection)
# =============================================================================

def compute_pixel_sharpness(image):
    """
    Compute per-pixel sharpness measure using gradient-based metric.

    This sharpness measure combines gradient magnitude with intensity deviation
    from the mean, as described in Si et al. (2023).

    Args:
        image: Input image, either grayscale or color

    Returns:
        Sharpness map of shape with higher values indicating sharper pixels

    Reference:
        Si et al. (2023) - DEReD
    """
    if image.ndim == 3:
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey_image = image

    grad_y, grad_x = np.gradient(grey_image)
    mu = np.average(grey_image) # Mean pixel value

    # Sharpness combines gradient magnitude with normalized intensity deviation
    # from Si et al. (2023)
    sharpness = (grad_x**2 + grad_y**2) - np.abs((grey_image - mu) / mu) - np.pow(grey_image - mu, 2)

    return sharpness


def trivial_aif_initialization(defocus_stack):
    """
    Create an all-in-focus image by selecting the sharpest pixel at each location.

    This is a simple baseline approach that independently selects the sharpest
    value for each pixel across the defocus stack, without enforcing spatial
    coherence.

    Args:
        defocus_stack: Image stack of shape (N, W, H, C) where N is the number
                      of images with different focus levels

    Returns:
        All-in-focus image of shape (W, H, C) constructed by selecting the
        sharpest pixel at each spatial location
    """
    _, width, height, _ = defocus_stack.shape

    # Compute sharpness for each image in the stack
    sharpness_stack = np.zeros(defocus_stack.shape[:3])
    for i in range(len(defocus_stack)):
        sharpness = compute_pixel_sharpness(defocus_stack[i])
        sharpness_stack[i] = sharpness

    utils.plot_single_stack(sharpness_stack, globals.Df)

    # Select sharpest pixel at each location
    aif = np.zeros((width, height, 3))
    for i in range(width):
        for j in range(height):
            sharpest_idx = np.argmax(sharpness_stack[:, i, j])
            aif[i, j, :] = defocus_stack[sharpest_idx, i, j, :]

    return aif


# =============================================================================
# MRF-Based AIF Initialization (based on Suwajanakorn et al.)
# =============================================================================

def compute_defocus_term(image, sigma=1.0, sharpness_measure='laplacian'):
    """
    Compute the defocus term (unary cost) for MRF optimization from Suwajanakorn et al.

    Args:
        image: Input image, either grayscale (H x W) or color (H x W x 3)
        sigma: Gaussian kernel standard deviation for smoothing (default: 1.0)
        sharpness_measure: Method for computing sharpness, one of:
            - 'laplacian': Absolute value of Laplacian operator
            - 'log': Laplacian of Gaussian
            - 'sobel_grad': Magnitude of Sobel gradients

    Returns:
        Defocus cost map of shape (H x W) with more negative values indicating
        sharper (less defocused) regions
    """
    assert sharpness_measure in ['laplacian', 'log', 'sobel_grad']

    # Convert to grayscale and normalize to [0, 1]
    if image.ndim == 3:
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
    else:
        grey_image = image / 255.

    # Compute sharpness using selected measure
    if sharpness_measure == 'sobel_grad':
        # Magnitude of image derivative
        sobel_h = scipy.ndimage.sobel(grey_image, 0) # Horizontal gradient
        sobel_v = scipy.ndimage.sobel(grey_image, 1) # Vertical gradient
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    elif sharpness_measure == 'log':
        # Laplacian of Gaussian
        log_response = scipy.ndimage.gaussian_laplace(grey_image, sigma=sigma)
        magnitude = np.abs(log_response)
    elif sharpness_measure == 'laplacian':
        # Absolute Laplacian
        laplacian = cv2.Laplacian(grey_image, ddepth=cv2.CV_32F, ksize=3)
        magnitude = np.abs(laplacian)

    # Exponential transformation to emphasize sharp regions
    sharpness = np.exp(magnitude)

    # Apply Gaussian blur to approximate patch-based computation
    # Negate so that sharper regions have more negative (lower) cost
    defocus = -scipy.ndimage.gaussian_filter(sharpness, sigma=sigma)

    return defocus

def mrf_optimization(defocus_stack, lmbda=0.05, sharpness_measure='laplacian'):
    """
    Perform MRF optimization using graph cuts to select optimal focus levels
    from Suwajanakorn et al.

    This function formulates the AIF selection as a discrete labeling problem
    where each pixel is assigned a label (focus level) that minimizes:
        E = sum_p D_p(l_p) + lambda * sum_{p,q} |l_p - l_q|

    where D_p is the unary cost (defocus term) and the pairwise term penalizes
    depth discontinuities between neighboring pixels.

    Args:
        defocus_stack: Image stack of shape (N, W, H, C) where N is the number
                      of different focus levels
        lmbda: Smoothness weight controlling the trade-off between data fidelity
              and spatial smoothness (default: 0.05)
        sharpness_measure: Method for computing sharpness (see compute_defocus_term)

    Returns:
        Label map of shape (W x H) where each value in [0, N-1] indicates the
        selected focus level for that pixel

    Reference:
        Suwajanakorn et al. - Depth from defocus on your mobile phone
    """
    n_labels, width, height, _ = defocus_stack.shape

    # Compute unary costs
    unary_cost = np.stack([
        compute_defocus_term(image, sharpness_measure=sharpness_measure)
        for image in defocus_stack
    ], axis=2).astype(np.int32)

    # Pairwise cost matrix: (N x N) penalizing label differences
    pairwise_cost = lmbda * np.abs(
        np.subtract.outer(np.arange(n_labels), np.arange(n_labels))
    ).astype(np.int32)

    # Run graph cut optimization on 4-connected grid
    labels = gco.cut_grid_graph_simple(
        unary_cost, pairwise_cost, n_iter=-1, connect=4, algorithm="expansion"
    )

    return labels

def compute_aif_initialization(defocus_stack, lmbda=0.05, sharpness_measure='laplacian'):
    """
    Compute an all-in-focus image using MRF-based optimization based on Suwajanakorn et al.

    This is the main function for generating an AIF image with spatial coherence.
    It uses graph cuts to optimize both local sharpness and smoothness of the
    depth assignment.

    Args:
        defocus_stack: Image stack of shape (N, W, H, C) where N is the number
                      of different focus levels
        lmbda: Smoothness weight for MRF optimization (default: 0.05)
        sharpness_measure: Method for computing sharpness (see compute_defocus_term)

    Returns:
        All-in-focus image of shape (W, H, C) constructed by selecting pixels
        according to the MRF-optimized label assignment
    """
    _, width, height, _ = defocus_stack.shape

    # Ensure input is in [0, 255] range for integer cost computation
    multiplier = 1.
    if defocus_stack.max() <= 1.5:
        multiplier = 255.

    # Run MRF optimization to get label assignment
    labels = mrf_optimization(
        defocus_stack * multiplier, lmbda=lmbda, sharpness_measure=sharpness_measure
    )

    # Construct AIF image using optimal label assignment
    rows = np.arange(width)[:, None]
    cols = np.arange(height)
    aif = defocus_stack[labels.reshape((width, height)), rows, cols]

    return aif
