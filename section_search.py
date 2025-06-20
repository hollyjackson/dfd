import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

import utils
import forward_model
import globals
import gradient_descent
import least_squares

from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import torch

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi


def total_variation_torch(image):
    tv_x = torch.abs(image[:, 1:] - image[:, :-1])
    tv_y = torch.abs(image[1:, :] - image[:-1, :])
    return tv_x.sum() + tv_y.sum()

def total_variation(image):
    tv_x = np.abs(image[:, 1:] - image[:, :-1])
    tv_y = np.abs(image[1:, :] - image[:-1, :])
    return np.sum(tv_x) + np.sum(tv_y)

def compute_tv_map(image, patch_size = None):
    if patch_size is not None:
        assert patch_size % 2 != 0 # must be odd
    if patch_size is None:
        patch_size = globals.MAX_KERNEL_SIZE
    
    pad = patch_size // 2
    width, height = image.shape[:2]

    tv_map = np.zeros((width, height))
    for i in range(pad, width - pad):
        for j in range(pad, height - pad):
            window = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            tv_map[i, j] = total_variation(window) / patch_size**2

    return tv_map

def gaussian(x, A, mu, sigma, C):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + C

def plot_defocus_stack(defocus_stack, depth):

    # Create the figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Plot the images
    for i in range(5):
        ax = axes[i]
        ax.imshow(defocus_stack[i] / defocus_stack[i].max())
        ax.axis('off')  # Hide axes
        ax.set_title(globals.Df[i])  # Set individual image titles

    # Hide the last empty subplot
    axes[5].axis('off')

    # Set the main title
    fig.suptitle('depth = '+f"{depth:.1f}"+' m')

    plt.tight_layout()

def objective(Z, i, j, gt_aif, defocus_stack_torch):
    pred = forward_model.forward_single_pixel(i, j, Z, gt_aif)
    gt = defocus_stack_torch[:,i,j,:]
    loss = torch.nn.functional.mse_loss(gt, pred)
    return loss.item()

def objective_full(depth_map, gt_aif, defocus_stack_torch, beta=0, proxy=None, gamma=0, last_dpt=None): 

    if isinstance(depth_map, np.ndarray):
        depth_map_torch = torch.from_numpy(depth_map).to(defocus_stack_torch.device)
    else:
        depth_map_torch = depth_map
    pred = forward_model.forward_torch(depth_map_torch, gt_aif)
    # loss = torch.nn.functional.mse_loss(defocus_stack_torch, pred)
    # return loss.item()
    loss = torch.mean((defocus_stack_torch - pred)**2, dim=(0, -1))
    if proxy is not None:
        # loss += beta * (depth_map_torch - proxy)**2
        loss += beta * torch.norm(depth_map_torch - proxy.to(gt_aif.device))**2
    if last_dpt is not None:
        loss += gamma * (depth_map_torch - last_dpt.to(depth_map_torch.device))**2
    return loss.detach().cpu().numpy()

# def optimize_z(i, j, gt_aif, defocus_stack_torch, brackets):
    
#     def objective(Z):
#         pred = forward_model.forward_single_pixel(i, j, Z, gt_aif)
#         gt = defocus_stack_torch[:,i,j,:]
#         loss = torch.nn.functional.mse_loss(gt, pred)
#         return loss.item()
    
#     print(i,j)
#     bracket = brackets[i,j]
#     res = scipy.optimize.minimize_scalar(
#             objective,
#             bracket = (bracket[0], bracket[1]),
#             method='golden')#,
#             # options = {'disp': True})

#     return (i, j, res.x)
    

def largest_quasiconvex_window(loss, argmin):
    pass

def grid_search(gt_aif, defocus_stack_torch, min_Z = 0.1, max_Z = 10, num_Z = 100, beta=0, proxy=None, gamma=0, last_dpt=None):
    # try many values of Z
    Z = np.linspace(min_Z, max_Z, num_Z)

    width, height, _ = gt_aif.shape
    indices = forward_model.precompute_indices(width, height)

    all_losses = np.zeros((width, height, num_Z))
    # for i in range(num_Z):
    for i in tqdm(range(num_Z), desc="Grid search".ljust(20), ncols=80):
        # print(i,'/',num_Z)
        dpt = torch.full((width,height), Z[i]).to(gt_aif.device)
        defocus_stack = forward_model.forward_torch(dpt, gt_aif, indices=indices)
        all_losses[:,:,i] = objective_full(dpt, gt_aif, defocus_stack_torch, beta=beta, proxy=proxy, gamma=gamma, last_dpt=last_dpt)
        # all_losses[:,:,i] = torch.mean((defocus_stack_torch - defocus_stack) ** 2, dim=[0,3]).cpu().numpy()
        # if proxy is not None:
        #     all_losses[:,:,i] += (beta * (dpt - proxy)**2).cpu().numpy()

    argmin_indices = np.argmin(all_losses, axis=2)
    depth_map = Z[argmin_indices]
    
    return depth_map, Z, argmin_indices, all_losses

def golden_section_search_nested(Z, argmin_indices, gt_aif, defocus_stack_torch,
    method='brent', window = 2):

    width, height = argmin_indices.shape

    # build a grid around each min
    brackets = np.zeros((width, height, 2))
    num_Z = len(Z)
    brackets[:,:,0] = Z[np.maximum(argmin_indices-window,0)]
    brackets[:,:,1] = Z[np.minimum(argmin_indices+window,num_Z-1)]

    depth_map_golden = np.zeros((width, height))

    nit = 0

    # for i in range(width):
    for i in tqdm(range(width), desc="Bracket search".ljust(20), ncols=80):
        for j in range(height):
            # print(i, j)
            bracket = brackets[i,j]

            depth_map_golden[i,j] = golden_section_search_single(
                bracket[0], bracket[1], i, j,
                gt_aif, defocus_stack_torch, tolerance=1e-5)
    
    return depth_map_golden

def golden_section_search_single(a, b, i, j, gt_aif, defocus_stack_torch, tolerance=1e-5):

    while b - a > tolerance:
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        f_c = objective(c, i, j, gt_aif, defocus_stack_torch)
        f_d = objective(d, i, j, gt_aif, defocus_stack_torch)
        if f_c < f_d:
            b = d
        else: # f(c) > f(d) to find the maximum
            a = c

    return (b + a) / 2


def find_high_tv_patches(dpt, tv_thresh = 0.15, patch_size = None):
    
    tv_map = compute_tv_map(dpt, patch_size = patch_size)
    problem_pixels = np.argwhere(tv_map > tv_thresh)

    return problem_pixels, tv_map

def find_constant_patches(aif, diff_thresh = 2, patch_size = None):
    if patch_size is not None:
        assert patch_size % 2 != 0 # must be odd
    width, height, _ = aif.shape
    if patch_size is None:
        patch_size = globals.MAX_KERNEL_SIZE
    rad = patch_size // 2
    
    # problem_pixels = []
    # for i in range(width):
    #     for j in range(height):
    #         imin = max(i-rad, 0)
    #         imax = min(i+rad+1, width)
    #         jmin = max(j-rad, 0)
    #         jmax = min(j+rad+1, height)
    #         patch = aif[imin:imax, jmin:jmax]
    #         red_diff = patch[:,:,0].max() - patch[:,:,0].min()
    #         green_diff = patch[:,:,1].max() - patch[:,:,1].min()
    #         blue_diff = patch[:,:,2].max() - patch[:,:,2].min()
    #         if red_diff <= 2  and green_diff <= 2 and blue_diff <=2:
    #             problem_pixels.append((i,j))

    padded_aif = np.pad(aif, ((rad, rad), (rad, rad), (0, 0)), mode='edge')
    patches = np.lib.stride_tricks.sliding_window_view(padded_aif, (patch_size, patch_size, 3))
    patches = np.squeeze(patches)

    max_vals = patches.max(axis=(2, 3))
    min_vals = patches.min(axis=(2, 3))
    color_diffs = max_vals - min_vals

    mask = np.all(color_diffs <= diff_thresh, axis=2)

    problem_pixels = np.argwhere(mask)

    return problem_pixels

def finite_difference(gt_aif, defocus_stack_torch, last_d, t_initial=None, epsilon=1e-3, max_iter=100, alpha=0.3, beta=0.8, max_ls_iters=80, save_plots=False, show_plots=False, iter_folder=None, vmin=0.9, vmax=1.7):
    width, height = last_d.shape
    if t_initial == None:
        print('Initialized t to 1')
        t_initial = np.ones((width, height))
    elif np.isscalar(t_initial):
        print('Initialized t to',t)
        t_initial = np.ones((width, height)) * t_initial
    
    last_d = last_d.cpu().numpy()
    # for i in tqdm(range(max_iter), desc="Finite differences".ljust(20), ncols=80):
    for i in range(max_iter):
        t = t_initial.copy() # reset step size 
        
        # f(x) and f(x + eps)
        f_d = objective_full(last_d, gt_aif, defocus_stack_torch)
        print('mse:',np.mean(f_d))
        f_d_eps = objective_full(last_d + epsilon, gt_aif, defocus_stack_torch)

        # grad = (f(x + eps) - f(x)) / (2 eps)
        grad = (f_d_eps - f_d) / (2 * epsilon)
        
        # plt.imshow(grad)
        # plt.colorbar()
        # plt.title('grad = (f(x + eps) - f(x)) / (2 eps)')
        # plt.show()

        # backtracking line search from boyd (p.464)
        print('Backtracking line search...')
        for ls_iter in range(max_ls_iters):#tqdm(range(max_ls_iters), desc="Line search".ljust(20), ncols=80): # in lieu of while loop
            # evaluate f(x + t d) for current t
            d_temp = last_d - t * grad
            f_d_t_eps = objective_full(d_temp, gt_aif, defocus_stack_torch)

            mask = (f_d_t_eps > f_d - alpha * t * grad ** 2)
            t[mask] = beta * t[mask]

            # print(np.sum(mask),'values unsatisfied')
            if not np.any(mask): # all satisfied
                print('...satisfied after',ls_iter,'iters')
                break
        
        print(np.sum(mask),'values unsatisfied')
        print('...done')
            
        # plt.imshow(t)
        # plt.colorbar()
        # plt.title('t')
        # plt.show()

        # gradient step
        d = last_d - t * grad
        last_d = d

        if save_plots or show_plots:
            plt.imshow(d, vmin=vmin, vmax=vmax)
            plt.colorbar()
            if save_plots:
                plt.savefig(os.path.join(iter_folder,'finite_differences_iter'+str(i)+'.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()


    return d

def golden_section_search(Z, argmin_indices, gt_aif, defocus_stack_torch,
    window=1, tolerance=1e-5, convergence_error=0, max_iter=100, beta=0, proxy=None, gamma=0, last_dpt=None, a_b_init=None):
    assert convergence_error >= 0 and convergence_error < 1
    print("Golden-section search...")
    
    # build a grid around each min
    if a_b_init is None:
        num_Z = len(Z)
        a = Z[np.maximum(argmin_indices-window,0)]
        b = Z[np.minimum(argmin_indices+window,num_Z-1)]
    else:
        a, b = a_b_init

    print('...searching for',(1 - convergence_error)*100,'% convergence')

    i = 0
    while (((convergence_error == 0 and np.any(b - a > tolerance))
            or (convergence_error != 0 and np.sum((b - a) > tolerance) / a.size > (1 - convergence_error)))
            and (i < max_iter)): # 99% convergence
        # print(i)
        # # update converged values
        # mask = np.abs(b - a) < tolerance
        # avg = (b + a) / 2
        # a[mask] = avg[mask]
        # b[mask] = avg[mask]

        # gss code from wiki
        c = b - (b - a) * invphi
        d = a + (b - a) * invphi
        f_c = objective_full(c, gt_aif, defocus_stack_torch, beta=beta, proxy=proxy, gamma=gamma, last_dpt=last_dpt)
        f_d = objective_full(d, gt_aif, defocus_stack_torch, beta=beta, proxy=proxy, gamma=gamma, last_dpt=last_dpt)

        mask = f_c < f_d
        b[mask] = d[mask]

        mask = f_c > f_d
        a[mask] = c[mask]

        i += 1

    if (i >= max_iter):
        print('Failed to converge after',i,'iterations')
        print(np.sum((b - a) <= tolerance) / a.size * 100, '% convergence achieved')

    print("...done")

    return (b + a) / 2


def remove_outliers(depth_map, gt_aif, patch_type = 'tv', diff_thresh = 2, tv_thresh = 0.15, to_plot=True):
    assert patch_type in ['tv', 'constant']
    print("Removing outliers...")

    # COULD REMOVE OUTLIERS IN HIGH-TV AREAS OF THE DEPTH MAP

    if patch_type == 'constant':
        problem_pixels = find_constant_patches(gt_aif, diff_thresh = diff_thresh)
    else:
        problem_pixels, tv_map = find_high_tv_patches(depth_map, tv_thresh = tv_thresh)
    
    problem_pixel_set = set(map(tuple, problem_pixels))
    print('found',len(problem_pixel_set),'outliers')
    
    # target_pixel = np.array([21, 188])
    # is_present = np.any(np.all(problem_pixels == target_pixel, axis=1))
    # print(is_present)

    if to_plot:
        if patch_type == 'constant':
            plt.imshow(gt_aif / 255.)
        else:
            plt.imshow(tv_map)
            plt.colorbar()
        plt.scatter([y for x, y in problem_pixels], [x for x, y in problem_pixels],
            color='red', marker='x', s=50)
        plt.title("Outliers ("+patch_type+')')
        plt.show()
    
    removed = 0
    lim = int((float(globals.MAX_KERNEL_SIZE) - 1) / 2.)
    for i, j in problem_pixels:
        patch = []
        for dx in range(-lim, lim+1, 1):
            for dy in range(-lim, lim+1, 1):
                if i + dx < 0 or i + dx >= gt_aif.shape[0]:
                    continue
                if j + dy < 0 or j + dy >= gt_aif.shape[1]:
                    continue
                if (i + dx, j + dy) in problem_pixel_set:
                    continue
                if dx == 0 and dy == 0:
                    continue
                patch.append(depth_map[i+dx,j+dy])
        if len(patch) != 0:
            removed += 1
            avg_depth = np.array(patch).mean(axis=0, keepdims=True)
            depth_map[i,j] = avg_depth
            problem_pixel_set.remove((i, j)) # remove so it can be used later 

    print(removed,'/',len(problem_pixels),'outliers removed')
    
    return depth_map, (len(problem_pixels)/(depth_map.shape[0] * depth_map.shape[1]))

def plot_grid_search_on_pixel(i, j, Z, all_losses, gt_dpt=None):

    plt.figure(figsize=(10,5))

    plt.plot(Z, all_losses[i, j, :], label='Loss Curve',
            linestyle='-', marker='.', markersize=4, color='black')

    if gt_dpt is not None:
        plt.scatter([gt_dpt[i,j]], [0],
                color='red', marker='x', s=100, label='Ground Truth Depth')
            
    min_loss_idx = np.argmin(all_losses[i,j])
    plt.scatter([Z[min_loss_idx]], [all_losses[i,j,min_loss_idx]], 
            color='green', marker='x', s=100, label='Depth with Min Loss')
    
    plt.xticks(Z[::2], labels=np.round(Z[::2], 2), rotation=45)
    plt.xlabel('Depth (m)')
    plt.ylabel('MSE between Predicted and Ground Truth Defocus Stack')
    plt.title(f'Pixel at {(i, j)}')
    
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def bracket_search(Z, argmin_indices, gt_aif, defocus_stack_torch,
    method='brent', window = 2):

    width, height = argmin_indices.shape

    # build a grid around each min
    brackets = np.zeros((width, height, 2))
    # TODO: smart bracket selection 
    # select a quasiconvex window 
    num_Z = len(Z)
    # window = 2
    brackets[:,:,0] = Z[np.maximum(argmin_indices-window,0)]
    brackets[:,:,1] = Z[np.minimum(argmin_indices+window,num_Z-1)]

    depth_map_golden = np.zeros((width, height))

    nit = 0

    # for i in range(width):
    for i in tqdm(range(width), desc="Bracket search".ljust(20), ncols=80):
        for j in range(height):
            # print(i, j)
            bracket = brackets[i,j]
            # print('bracket:',(bracket[0], bracket[1]))
            my_args = (i, j, gt_aif, defocus_stack_torch)
            res = scipy.optimize.minimize_scalar(
                objective,
                bracket = (bracket[0], bracket[1]),
                args = my_args,
                method = method)
                # options = {'disp': True})

            # print(res.x,depth_map[i,j],gt_dpt[i,j])
            depth_map_golden[i,j] = res.x
            nit += res.nit

    # print(nit,'iterations')

    return depth_map_golden

def grid_plus_bactracking(gt_aif, defocus_stack_torch, method='brent', window = 2, to_plot=True, path=None):
    
    depth_map, Z, argmin_indices, all_losses = grid_search(gt_aif, defocus_stack_torch)


    fn = 'depth_map_grid_search.png'
    if path != None:
        fn = os.path.join(path, fn)


    plt.imshow(depth_map, vmin=0.7, vmax=1.9)
    plt.colorbar()
    plt.savefig(fn)
    plt.close()

    criterion = torch.nn.MSELoss()

    defocus_stack_pred = forward_model.forward_torch(torch.from_numpy(depth_map), gt_aif)
    loss = criterion(defocus_stack_pred, defocus_stack_torch)
    print('Loss after grid search:',loss.item())

    depth_map_golden = golden_section_search(Z, argmin_indices, gt_aif, defocus_stack_torch, window = 2)
    
    defocus_stack_pred = forward_model.forward_torch(torch.from_numpy(depth_map_golden), gt_aif)
    loss = criterion(defocus_stack_pred, defocus_stack_torch)
    print('Loss after bracket search:',loss.item())


    fn = 'depth_map_bracket_search.png'
    if path != None:
        fn = os.path.join(path, fn)

    plt.imshow(depth_map_golden, vmin=0.7, vmax=1.9)
    plt.colorbar()
    plt.savefig(fn)
    plt.close()

    return depth_map_golden

