import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import utils
import forward_model
import globals
#import gradient_descent
import least_squares
import section_search
import coordinate_descent
import initialization

#import torch

# globals
IMAGE_RANGE = 255.
#FORWARD_KERNEL_TYPE = 'gaussian'
EXPERIMENT_NAME = 'all-test-'

def load_image(image_number):
    globals.init_NYUv2()

    global EXPERIMENT_NAME
    EXPERIMENT_NAME += image_number

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    
    # load data 
    gt_aif, gt_dpt = utils.load_single_sample(sample=image_number, set='test', fs=5, res='half')
    # gt_aif, gt_dpt, _ = utils.load_sample_image(fs=5, res='half')
    gt_aif = gt_aif * IMAGE_RANGE
    
    # plt.imshow(gt_aif / IMAGE_RANGE)
    # plt.show()
    
    # plt.imshow(gt_dpt)
    # plt.colorbar()
    # plt.show()
    
    width, height = gt_dpt.shape
    
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    # print('adaptive kernel size set to',max_kernel_size)
    utils.update_max_kernel_size(max_kernel_size)

    return gt_aif, gt_dpt

def gt_defocus_stack(gt_dpt, gt_aif):
    # forward model (torch)
    defocus_stack = forward_model.forward(gt_dpt, gt_aif)
#    defocus_stack_torch = forward_model.forward_torch(gt_dpt, gt_aif, kernel=FORWARD_KERNEL_TYPE)#.float() / 255.0)
    utils.plot_single_stack(defocus_stack / IMAGE_RANGE, globals.Df)
    
    return defocus_stack

def aif_initialization(defocus_stack):
    # AIF initialization

    # aif_init = initialization.trivial_aif_initialization(defocus_stack)
    aif_init = initialization.compute_aif_initialization(defocus_stack, lmbda=0.05, sharpness_measure='sobel_grad')
    plt.imshow(aif_init / IMAGE_RANGE)
    plt.show()
    
    plt.imshow(defocus_stack[1] / IMAGE_RANGE)
    plt.show()

    return aif_init

def coord_descent(defocus_stack, num_epochs = 40,
                  save_plots = True, 
                  least_squares_first = False,
                  depth_init = None, aif_init = None,
                  vmin = 0.1, vmax = 10):
    # -------------------
    # COORDINATE DESCENT
    # -------------------
    
    dpt, aif, _, exp_folder = coordinate_descent.coordinate_descent(
            defocus_stack,
            experiment_folder='/data/holly_jackson/experiments',
            show_plots=False, save_plots=save_plots,
            experiment_name = EXPERIMENT_NAME, 
            num_epochs = num_epochs,
            least_squares_first = least_squares_first,
            depth_init = depth_init, aif_init = aif_init, 
            k = 1, aif_method = 'fista',
            finite_differences = False, num_Z = 100, 
            ls_maxiter = 200, ls_maxiter_multiplier = 1.05, 
            vmin = vmin, vmax = vmax,
            verbose = False
    )

    
    return dpt, aif, exp_folder


def main():
    # Ensure correct usage
    if len(sys.argv) != 2:
        print("Usage: python run_coordinate_descent.py <image_number>")
        sys.exit(1)

    image_number = sys.argv[1]

    # load image
    gt_aif, gt_dpt = load_image(image_number)

    # generate defocus stack
    defocus_stack = gt_defocus_stack(gt_dpt, gt_aif)

    # coord descent
    dpt, aif, exp_folder = coord_descent(
        defocus_stack, save_plots = False,
        num_epochs = 40, least_squares_first = False,
        vmin = gt_dpt.min(), vmax = gt_dpt.max()
    )

    # save final 
    utils.save_dpt(exp_folder, 'dpt', dpt)
    # print(aif.min(), aif.max())
    # assert aif.min() > 0 and aif.max() > 1 and aif.max() <= 255 # remove later
    utils.save_aif(exp_folder, 'aif', aif)

    # final metrics save to file
    rms = utils.compute_RMS(dpt, gt_dpt)
    rel = utils.compute_AbsRel(dpt, gt_dpt)
    deltas = utils.compute_accuracy_metrics(dpt, gt_dpt)
    outfile = os.path.join(exp_folder, "accuracy_metrics.txt")
    delta_str = ", ".join(f"{float(deltas[d]):.6f}" for d in sorted(deltas.keys()))
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"RMS: {float(rms):.6f}\n")
        f.write(f"Rel: {float(rel):.6f}\n")
        f.write(f"Accuracy (δ1, δ2, δ3): {delta_str}\n")

if __name__ == "__main__":
    main()
