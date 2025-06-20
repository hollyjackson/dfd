import os
import numpy as np
import matplotlib.pyplot as plt

import utils
import forward_model
import globals
import gradient_descent
import least_squares
import section_search

import torch


if __name__ == "__main__":

    globals.init_NYUv2()

    EXPERIMENT_NAME = 'coord-descent-bracket-search'

    experiment_folder = utils.create_experiment_folder(EXPERIMENT_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load data 
    gt_aif, gt_dpt, gt_defocus_stack = utils.load_single_sample(fs=5, res='full')
    width, height = gt_dpt.shape

    max_kernel_size = utils.kernel_size_heuristic(width, height)
    print('adaptive kernel size set to',max_kernel_size)
    utils.update_max_kernel_size(max_kernel_size)

    # for kernel_size in [7, 9, 11, 13, 15, 17]:
    #     print('\n\n\n\n')
    #     print(kernel_size)

    #     globals.init_NYUv2()
    #     utils.update_max_kernel_size(kernel_size)

        

    

    # forward model (torch)
    defocus_stack = forward_model.forward(gt_dpt, gt_aif)
    defocus_stack_torch = forward_model.forward_torch(gt_dpt, gt_aif)#.float() / 255.0)
    utils.plot_stacks_side_by_side(gt_defocus_stack, defocus_stack_torch, globals.Df)
    
    # gt aif --> depth map
    
    depth_map, Z, argmin_indices, all_losses = section_search.grid_search(gt_aif, defocus_stack_torch)

    # utils.plot_compare_greyscale(depth_map, gt_dpt, vmin=0.7, vmax=1.9)
    # plt.show()

    plt.imshow(depth_map, vmin=0.7, vmax=1.9)
    plt.colorbar()
    plt.show()

    criterion = torch.nn.MSELoss()
    defocus_stack_pred = forward_model.forward_torch(torch.from_numpy(depth_map), gt_aif)
    
    # utils.plot_stacks_side_by_side(defocus_stack_torch, defocus_stack_pred, globals.Df)

    loss = criterion(defocus_stack_pred, defocus_stack_torch)
    print('Loss:',loss.item())
    print('----------------------')
    

    depth_map_golden = section_search.golden_section_search(Z, argmin_indices, gt_aif, defocus_stack_torch)
    
    # utils.plot_compare_greyscale(depth_map_golden, gt_dpt, vmin=0.7, vmax=1.9)
    # plt.show()

    plt.imshow(depth_map_golden, vmin=0.7, vmax=1.9)
    plt.colorbar()
    plt.show()

    criterion = torch.nn.MSELoss()
    defocus_stack_pred = forward_model.forward_torch(torch.from_numpy(depth_map_golden), gt_aif)
    # utils.plot_stacks_side_by_side(defocus_stack_torch, defocus_stack_pred, globals.Df)
    
    loss = criterion(defocus_stack_pred, defocus_stack_torch)
    print('Loss:',loss.item())

    worst_coords = utils.get_worst_diff_pixels(torch.from_numpy(depth_map), gt_dpt,
        num_worst_pixels = 5)
    for i, j in worst_coords:
        section_search.plot_grid_search_on_pixel(i.item(), j.item(), Z, all_losses, gt_dpt)


    depth_map_golden = section_search.remove_outliers(depth_map_golden, gt_aif)
    
    # utils.plot_compare_greyscale(depth_map_golden, gt_dpt, vmin=0.7, vmax=1.9)
    # plt.show()

    plt.imshow(depth_map_golden, vmin=0.7, vmax=1.9)
    plt.colorbar()
    plt.show()

    criterion = torch.nn.MSELoss()
    defocus_stack_pred = forward_model.forward_torch(torch.from_numpy(depth_map_golden), gt_aif)
    # utils.plot_stacks_side_by_side(defocus_stack_torch, defocus_stack_pred, globals.Df)
    
    loss = criterion(defocus_stack_pred, defocus_stack_torch)
    print('Loss:',loss.item())

    # # plt.imshow(gt_aif)
    # # plt.scatter([y.item() for x, y in worst_coords], [x.item() for x, y in worst_coords], color='red', marker='x', s=100, label='Worst Diff Pixels')
    # # plt.title('Worst Difference Pixels Over Image')
    # # plt.legend()
    # # plt.show()


    # print('----------------------')

    # # depth map to gt aif 
    # depth_map_golden = torch.from_numpy(depth_map_golden)
    # aif = least_squares.least_squares(depth_map_golden, defocus_stack)
    
    # print('aif range:',aif.min(),aif.max())
    
    # utils.plot_compare_rgb(aif, gt_aif)
    # plt.savefig(os.path.join(experiment_folder,'ls.png'))
    # plt.close()

    # print(np.linalg.norm(np.array(aif) - np.array(gt_aif)))

    # criterion = torch.nn.MSELoss()
    # aif = torch.from_numpy(aif)
    # defocus_stack_pred = forward_model.forward_torch(depth_map_golden, aif)
    
    # utils.plot_stacks_side_by_side(defocus_stack_torch, defocus_stack_pred, globals.Df)
    
    
    # loss = criterion(defocus_stack_pred, defocus_stack_torch)
    # print('Loss:',loss.item())
    # print('----------------------')
