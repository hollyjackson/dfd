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
EXPERIMENT_NAME = 'mobile-depth-'
windowed_MSE = False
#globals.window_size = 3
globals.thresh = 0.5
if windowed_MSE:
    EXPERIMENT_NAME += "windowed"+str(globals.window_size)+"-"
EXPERIMENT_NAME += "thresh"+str(globals.thresh)+"-"

def load_image(example_name):

    globals.init_MobileDepth()

    global EXPERIMENT_NAME
    EXPERIMENT_NAME += example_name

    
    IMAGE_RANGE = 255.
    assert example_name in ["keyboard", "bottles", "fruits", "metals", "plants", "telephone", "window", "largemotion", "smallmotion", "zeromotion", "balls"]
    
    defocus_stack, dpt_result, scale_mat = utils.load_single_sample_MobileDepth(example_name)
    
    defocus_stack = np.stack([
        skimage.transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True)
        for img in defocus_stack
    ], axis=0) # half size so its easy to compute
    
    defocus_stack *= IMAGE_RANGE 
    
    # print(defocus_stack.shape, defocus_stack.min(), defocus_stack.max())
    globals.ps = 4.54e-3 / 3264 * (3264 / defocus_stack.shape[1]) # from Samsung Galaxy S3 sensor size
    print('Pixel size:', globals.ps)
    
    
    fs, width, height, _ = defocus_stack.shape
    print(fs, width, height)
    print(dpt_result.dtype, defocus_stack.dtype)
    
    globals.min_Z = max(0.001, globals.Df.min() - 0.05)
    globals.max_Z = min(3, globals.Df.max() + 0.5)
    print('Depth range', globals.min_Z,'-', globals.max_Z)
    
    
    max_kernel_size = utils.kernel_size_heuristic(width, height)
    print('adaptive kernel size set to',max_kernel_size)
    utils.update_max_kernel_size(max_kernel_size)

    return defocus_stack


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
                  vmin = 0.1, vmax = 10, windowed_MSE = True):
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
            min_Z = globals.min_Z, max_Z = globals.max_Z,
            verbose = False
    )

    
    return dpt, aif, exp_folder


def main():
    # Ensure correct usage
    if len(sys.argv) != 2:
        print("Usage: python run_coordinate_descent.py <example_name>")
        sys.exit(1)

    example_name = sys.argv[1]

    # load image
    defocus_stack = load_image(example_name)

    # coord descent
    # globals.window = 3
    dpt, aif, exp_folder = coord_descent(
        defocus_stack, save_plots = False,
        num_epochs = 40, least_squares_first = False,
        vmin = globals.min_Z, vmax = globals.max_Z,
        windowed_MSE = windowed_MSE
    )

    # save final 
    utils.save_dpt(exp_folder, 'dpt', dpt)
    # print(aif.min(), aif.max())
    # assert aif.min() > 0 and aif.max() > 1 and aif.max() <= 255 # remove later
    utils.save_aif(exp_folder, 'aif', aif)

    # final metrics save to file
    # rms = utils.compute_RMS(dpt, gt_dpt)
    # rel = utils.compute_AbsRel(dpt, gt_dpt)
    # deltas = utils.compute_accuracy_metrics(dpt, gt_dpt)
    # outfile = os.path.join(exp_folder, "accuracy_metrics.txt")
    # delta_str = ", ".join(f"{float(deltas[d]):.6f}" for d in sorted(deltas.keys()))
    # with open(outfile, "w", encoding="utf-8") as f:
    #     f.write(f"RMS: {float(rms):.6f}\n")
    #     f.write(f"Rel: {float(rel):.6f}\n")
    #     f.write(f"Accuracy (δ1, δ2, δ3): {delta_str}\n")

if __name__ == "__main__":
    main()
