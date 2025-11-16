import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage

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
EXPERIMENT_NAME = 'make3d-'
windowed_MSE = True
globals.window_size = 5
globals.thresh = 0.5
if windowed_MSE:
    EXPERIMENT_NAME += "windowed"+str(globals.window_size)+"-"
EXPERIMENT_NAME += "thresh"+str(globals.thresh)+"-"

def load_image(img_name = "img-math7-p-282t0.jpg"):

    globals.init_Make3D()

    global EXPERIMENT_NAME
    EXPERIMENT_NAME += img_name.split("img-")[1].split(".jpg")[0]

    IIMAGE_RANGE = 255.
    gt_aif, gt_dpt = utils.load_single_sample_Make3D(img_name = img_name, data_dir = "/data/holly_jackson/", split='train')
    gt_aif = gt_aif * IMAGE_RANGE
    
    
    gt_dpt_resized = skimage.transform.resize(
        gt_dpt,
        output_shape=(460, 345), # (height, width)
        order=1,                  # bilinear interpolation
        anti_aliasing=True,
        preserve_range=True       # keep values in [0, 255] if original was uint8
    )
    
    plt.imshow(gt_dpt_resized)
    plt.colorbar()
    plt.show()
    
    width, height, _ = gt_aif.shape
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    print('adaptive kernel size set to',max_kernel_size)
    utils.update_max_kernel_size(max_kernel_size)

    defocus_stack = forward_model.forward(gt_dpt_resized, gt_aif)
    
    return defocus_stack, gt_aif, gt_dpt_resized


def aif_initialization(defocus_stack):
    # AIF initialization

    # aif_init = initialization.trivial_aif_initialization(defocus_stack)
    aif_init = initialization.compute_aif_initialization(defocus_stack, lmbda=0.05, sharpness_measure='sobel_grad')
    # plt.imshow(aif_init / IMAGE_RANGE)
    # plt.show()
    
    # plt.imshow(defocus_stack[1] / IMAGE_RANGE)
    # plt.show()

    return aif_init

def coord_descent(defocus_stack, num_epochs = 40,
                  save_plots = True, 
                  least_squares_first = False,
                  depth_init = None, aif_init = None,
                  vmin = 0.1, vmax = 10, num_Z = 100, windowed_MSE = False):
    # -------------------
    # COORDINATE DESCENT
    # -------------------
    
    dpt, aif, _, exp_folder = coordinate_descent.coordinate_descent(
            defocus_stack,
            experiment_folder='/data/holly_jackson/experiments/all-make3d-train-v2',
            show_plots=False, save_plots=save_plots,
            experiment_name = EXPERIMENT_NAME, 
            num_epochs = num_epochs,
            least_squares_first = least_squares_first,
            depth_init = depth_init, aif_init = aif_init, 
            k = 1, aif_method = 'fista',
            finite_differences = False, num_Z = num_Z, 
            ls_maxiter = 200, ls_maxiter_multiplier = 1.05, 
            vmin = vmin, vmax = vmax,
            min_Z = globals.min_Z, max_Z = globals.max_Z,
            verbose = False, windowed_mse = windowed_MSE
    )

    
    return dpt, aif, exp_folder


def main():
    # Ensure correct usage
    if len(sys.argv) != 2:
        print("Usage: python run_coordinate_descent.py <example_name>")
        sys.exit(1)

    example_name = sys.argv[1]

    # load image
    defocus_stack, gt_aif, gt_dpt = load_image(example_name)

    # aif initialization
    # aif_init = aif_initialization(defocus_stack)
    depth_init = 40
    
    # coord descent
    dpt, aif, exp_folder = coord_descent(
        defocus_stack, save_plots = False,
        num_epochs = 5, least_squares_first = True,
        depth_init = depth_init, #aif_init = aif_init,
        vmin = globals.min_Z, vmax = globals.max_Z,
        num_Z = 100, windowed_MSE = windowed_MSE
    )

    # save final 
    utils.save_dpt_npy(exp_folder, 'dpt', dpt)
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
