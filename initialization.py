import numpy as np
import cv2 as cv
import scipy
import gco
import matplotlib.pyplot as plt
import section_search
import utils
import globals

# AIF initialization
def compute_pixel_sharpness(image):
    if image.ndim == 3:
        grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        grey_image = image
        
    grad_y, grad_x = np.gradient(grey_image)
    mu = np.average(grey_image) # mean pixel value 
    sharpness = (grad_x**2 + grad_y**2) - np.abs((grey_image - mu) / mu) - np.pow(grey_image - mu, 2) # from si et al 23
    return sharpness


def trivial_aif_initialization(defocus_stack):
    _, width, height, _ = defocus_stack.shape
    
    sharpness_stack = np.zeros(defocus_stack.shape[:3])
    for i in range(len(defocus_stack)):
        # sharpness = section_search.compute_tv_map(defocus_stack[i])
        sharpness = compute_pixel_sharpness(defocus_stack[i])
        sharpness_stack[i] = sharpness
    
    utils.plot_single_stack(sharpness_stack, globals.Df)
    
    aif = np.zeros((width, height, 3))
    for i in range(width):
        for j in range(height):
            # choose"sharpest"
            aif[i,j,:] = defocus_stack[np.argmax(sharpness_stack[:,i,j]), i, j, :]
    # plt.imshow(aif)
    # plt.show()

    return aif

def compute_defocus_term(image, sigma=1.0, sharpness_measure='laplacian'):
    assert sharpness_measure in ['laplacian', 'log', 'sobel_grad']
    if image.ndim == 3:
        grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) / 255.
    else:
        grey_image = image / 255.

    if sharpness_measure == 'sobel_grad':
        # magnitude of image derivative
        sobel_h = scipy.ndimage.sobel(grey_image, 0)  # horizontal gradient
        sobel_v = scipy.ndimage.sobel(grey_image, 1)  # vertical gradient
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    elif sharpness_measure == 'log':
        # laplacian of gaussian
        log_response = scipy.ndimage.gaussian_laplace(grey_image, sigma=sigma)
        magnitude = np.abs(log_response)    
    elif sharpness_measure == 'laplacian':
        # absolute laplacian
        laplacian = cv.Laplacian(grey_image, ddepth=cv.CV_32F, ksize=3)
        magnitude = np.abs(laplacian)

    # sum of hf coeffs
    # hf = np.abs(np.fft.fft2(grey_image))
    # magnitude = np.sum(np.abs(hf[low_cut:]))

    # wavelet??
    
    sharpness = np.exp(magnitude)
    
    # blur first to approximate Gaussian patches
    defocus = -scipy.ndimage.gaussian_filter(sharpness, sigma=sigma)

    # plt.imshow(defocus)
    # plt.colorbar()
    # plt.show()
    
    return defocus

def mrf_optimization(defocus_stack, lmbda=0.05, sharpness_measure='laplacian'):
    n_labels, width, height, _ = defocus_stack.shape

    # width x height x n_labels
    unary_cost = np.stack([compute_defocus_term(image, sharpness_measure=sharpness_measure) for image in defocus_stack], axis=2).astype(np.int32)
    # unary_cost = compute_unary(defocus_stack)
    print(unary_cost.shape)
    # pairwise cost matrix (n_labels x n_labels)
    pairwise_cost = lmbda * np.abs(np.subtract.outer(np.arange(n_labels), np.arange(n_labels))).astype(np.int32)

    # automatically constructs 4-connected grid graph based on unary cost
    labels = gco.cut_grid_graph_simple(unary_cost, pairwise_cost, n_iter=-1, connect=4, algorithm="expansion")

    return labels

def compute_aif_initialization(defocus_stack, lmbda=0.05, sharpness_measure='laplacian'):
    _, width, height, _ = defocus_stack.shape

    # make range from 0 to 255
    multiplier = 1.
    if defocus_stack.max() <= 1.5:
        multiplier = 255.
    labels = mrf_optimization(defocus_stack*multiplier, lmbda=lmbda, sharpness_measure=sharpness_measure)
    
    rows = np.arange(width)[:, None]
    cols = np.arange(height)
    aif = defocus_stack[labels.reshape((width, height)), rows, cols]
    # plt.imshow(aif / 255.)
    # plt.show()

    return aif
