import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import utils
import forward_model
import globals
import gradient_descent
import least_squares
import section_search
import coordinate_descent
import initialization

import torch

# globals
IMAGE_RANGE = 255.
FORWARD_KERNEL_TYPE = 'gaussian'
EXPERIMENT_NAME = 'coord-descent-'

def load_image(image_number):
    globals.init_NYUv2()

    global EXPERIMENT_NAME
    EXPERIMENT_NAME += image_number

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # load data 
    gt_aif, gt_dpt = utils.load_single_sample(sample=image_number, set='train', fs=5, res='half')
    # gt_aif, gt_dpt, _ = utils.load_sample_image(fs=5, res='half')
    gt_aif = gt_aif * IMAGE_RANGE
    
    # plt.imshow(gt_aif / IMAGE_RANGE)
    # plt.show()
    
    # plt.imshow(gt_dpt)
    # plt.colorbar()
    # plt.show()
    
    width, height = gt_dpt.shape
    
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    print('adaptive kernel size set to',max_kernel_size)
    utils.update_max_kernel_size(max_kernel_size)

    return gt_aif, gt_dpt

def gt_defocus_stack(gt_dpt, gt_aif):
    # forward model (torch)
    defocus_stack = forward_model.forward(gt_dpt, gt_aif, kernel=FORWARD_KERNEL_TYPE)
    defocus_stack_torch = forward_model.forward_torch(gt_dpt, gt_aif, kernel=FORWARD_KERNEL_TYPE)#.float() / 255.0)
    utils.plot_single_stack(defocus_stack_torch / IMAGE_RANGE, globals.Df)
    
    return defocus_stack, defocus_stack_torch

def aif_initialization(defocus_stack):
    # AIF initialization

    # aif_init = initialization.trivial_aif_initialization(defocus_stack)
    aif_init = initialization.compute_aif_initialization(defocus_stack, lmbda=0.05, sharpness_measure='sobel_grad')
    plt.imshow(aif_init / IMAGE_RANGE)
    plt.show()
    
    plt.imshow(defocus_stack[1] / IMAGE_RANGE)
    plt.show()

    return aif_init

def coord_descent(defocus_stack, least_squares_first = True, depth_init = 1, aif_init = None,
                  vmin = 0.1, vmax = 10):
    # -------------------
    # COORDINATE DESCENT
    # -------------------
    
    dpt, aif, _ = coordinate_descent.coordinate_descent(defocus_stack, show_plots=False,
                                                     save_plots=True, experiment_name = EXPERIMENT_NAME, 
                                                        num_epochs=40,
                                                     least_squares_first=least_squares_first, depth_init=depth_init,
                                                     aif_init=aif_init, 
                                                        k = 5, aif_method='fista', finite_differences=False, num_Z=100, 
                                                     ls_maxiter=200, ls_maxiter_multiplier=1.05,#1.075, 
                                                     use_CUDA=False, vmin = vmin, vmax = vmax)

    return dpt, aif


def main():
    # Ensure correct usage
    if len(sys.argv) != 2:
        print("Usage: python run_coordinate_descent.py <image_number>")
        sys.exit(1)

    image_number = sys.argv[1]

    # load image
    gt_aif, gt_dpt = load_image(image_number)

    # generate defocus stack
    defocus_stack, defocus_stack_torch = gt_defocus_stack(gt_dpt, gt_aif)

    # coord descent
    dpt, aif = coord_descent(defocus_stack, vmin=gt_dpt.min(), vmax=gt_dpt.max(),
                             depth_init=1)

if __name__ == "__main__":
    main()
