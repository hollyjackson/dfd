import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

import utils
import forward_model
import globals
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


def objective(Z, i, j, gt_aif, defocus_stack_torch):
    pred = forward_model.forward_single_pixel(i, j, Z, gt_aif)
    gt = defocus_stack_torch[:,i,j,:]
    loss = torch.nn.functional.mse_loss(gt, pred)
    return loss.item()

def objective_full(depth_map, gt_aif, defocus_stack_torch, pred=None, beta=0, proxy=None, gamma=0, similarity_penalty=False, last_dpt=None): 

    if isinstance(depth_map, np.ndarray):
        depth_map_torch = torch.from_numpy(depth_map).to(defocus_stack_torch.device)
    else:
        depth_map_torch = depth_map
    if pred is None:
        pred = forward_model.forward_torch(depth_map_torch, gt_aif)
    # loss = torch.nn.functional.mse_loss(defocus_stack_torch, pred)
    # return loss.item()
    loss = torch.mean((defocus_stack_torch - pred)**2, dim=(0, -1))
    if proxy is not None:
        # loss += beta * (depth_map_torch - proxy)**2
        loss += beta * torch.norm(depth_map_torch - proxy.to(gt_aif.device))**2
    if similarity_penalty:#last_dpt is not None:
        loss += gamma * (depth_map_torch - last_dpt.to(depth_map_torch.device))**2
    return loss.detach().cpu().numpy()


def k_min_indices_no_overlap(sorted_indices, k, gss_window=1):
    width, height, num_Z = sorted_indices.shape
    k_min_indices = np.zeros((width, height, k), dtype=np.int32) - 1
    k_min_indices[:,:,0] = sorted_indices[:,:,0]
    
    # last_z = sorted_indices[:,:,0]
    kk = np.ones((width, height), dtype=np.int32)
    
    for z in range(1, num_Z):
        # mask = (abs(sorted_indices[:,:,z] - last_z) > gss_window) & (kk < k)
        mask = kk < k
        for l in range(k):
            too_close = (abs(sorted_indices[:,:,z] - k_min_indices[:,:,l]) <= gss_window) & (k_min_indices[:,:,l] >= 0)
            mask = mask & ~too_close
        if not np.any(mask):
            continue
        i, j = np.where(mask)
        k_min_indices[i, j, kk[i,j]] = sorted_indices[:,:,z][i, j]
        # last_z[i, j] = sorted_indices[:,:,z][i,j]
        kk[i, j] += 1
        if np.all(kk >= k):
            break

    assert np.all(k_min_indices >= 0)
        
    return k_min_indices



def grid_search_opt_k(gt_aif, defocus_stack_torch, min_Z = 0.1, max_Z = 10, num_Z = 100, k = 3, beta=0, proxy=None, gamma=0, similarity_penalty=False, last_dpt=None, gss_window=1):
    # try many values of Z
    Z = np.linspace(min_Z, max_Z, num_Z)

    width, height, num_channels = gt_aif.shape
    u, v = forward_model.compute_u_v()

    all_losses = np.zeros((width, height, num_Z))
    # for i in range(num_Z):
    for i in tqdm(range(num_Z), desc="Grid search".ljust(20), ncols=80):
        # print(i,'/',num_Z)
        r = forward_model.computer(torch.tensor([[Z[i]]]), globals.Df).unsqueeze(-1).unsqueeze(-1)
        # print(r)
        # print(r.shape, u.shape, v.shape)
        G, _ = forward_model.computeG(r, u, v)
        # print(G.shape)
        # print(G)
        G = G.squeeze()    
        defocus_stack = np.zeros((G.shape[0], width, height, num_channels))
        for j in range(G.shape[0]): # each focal setting
            kernel = G[j]
            for c in range(num_channels):
                defocus_stack[j,:,:,c] = scipy.ndimage.convolve(gt_aif[:,:,c].cpu().numpy(), kernel, mode='constant')
        
        dpt = torch.full((width,height), Z[i]).to(gt_aif.device)
        all_losses[:,:,i] = objective_full(dpt, gt_aif, defocus_stack_torch, pred=torch.from_numpy(defocus_stack).to(gt_aif.device), beta=beta, proxy=proxy, gamma=gamma, similarity_penalty=similarity_penalty, last_dpt=last_dpt)

    sorted_indices = np.argsort(all_losses, axis=2)
    # k_min_indices = sorted_indices[:,:,:k] # axis = 2
    k_min_indices = k_min_indices_no_overlap(sorted_indices, k, gss_window=gss_window)
    # print(k_min_indices.shape)
    
    depth_maps = Z[k_min_indices]
    print(k_min_indices.shape, depth_maps.shape)
    
    return depth_maps, Z, k_min_indices, all_losses




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
    window=1, tolerance=1e-5, convergence_error=0, max_iter=100, beta=0, proxy=None, gamma=0, similarity_penalty=False, last_dpt=None, a_b_init=None):
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
        f_c = objective_full(c, gt_aif, defocus_stack_torch, beta=beta, proxy=proxy, gamma=gamma, similarity_penalty=similarity_penalty, last_dpt=last_dpt)
        f_d = objective_full(d, gt_aif, defocus_stack_torch, beta=beta, proxy=proxy, gamma=gamma, similarity_penalty=similarity_penalty, last_dpt=last_dpt)

        mask = f_c < f_d
        b[mask] = d[mask]

        mask = f_c > f_d
        a[mask] = c[mask]

        i += 1

    if (i >= max_iter):
        print('Failed to converge after',i,'iterations')
        print(np.sum((b - a) <= tolerance) / a.size * 100, '% convergence achieved')

    print("...done")

    dpt = (b + a) / 2

    if last_dpt is not None:
        mse = objective_full(dpt, gt_aif, defocus_stack_torch, beta=beta, proxy=proxy, gamma=0, similarity_penalty=False, last_dpt=None)
        last_mse = objective_full(last_dpt, gt_aif, defocus_stack_torch, beta=beta, proxy=proxy, gamma=0, similarity_penalty=False, last_dpt=None)
        dpt = np.where(mse <= last_mse, dpt, last_dpt)

    return dpt
    


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


