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


def windowed_mse_gss(depth_map, gt_aif, defocus_stack, indices=None, template_A_stack=None):
    # Not used in practice because too slow 
    
    if globals.window_size % 2 == 0:
        globals.window_size += 1
    rad = globals.window_size // 2
    _, width, height, _ = defocus_stack.shape

    losses = np.zeros((width, height), dtype=np.float32)
    denom = np.zeros((width, height), dtype=np.float32)
    for i in range(-rad, rad+1):
        x_shifted = np.roll(depth_map, shift=i, axis=0)
        for j in range(-rad, rad+1):
            shifted = np.roll(x_shifted, shift=j, axis=1)
            # compute mse 
            pred = forward_model.forward(shifted, gt_aif, indices=indices, template_A_stack=template_A_stack)
            mse = np.mean((defocus_stack - pred)**2, axis=(0, -1))
            i_start = -i if i < 0 else 0
            i_end = width-i if i > 0 else width
            j_start = -j if j < 0 else 0
            j_end = height-j if j > 0 else height
            losses[i_start:i_end, j_start:j_end] += mse[i_start+i:i_end+i, j_start+j:j_end+j]
            denom[i_start:i_end, j_start:j_end] += 1
            
    return losses / denom

def windowed_mse_grid(defocus_stack, pred):#, window_size=5):
    if globals.window_size % 2 == 0:
        globals.window_size += 1
    rad = globals.window_size // 2
    # rad = globals.MAX_KERNEL_SIZE // 2
    _, width, height, _ = defocus_stack.shape
    mse = np.mean((defocus_stack - pred)**2, axis=(0, -1))
    losses = np.zeros((width, height), dtype=np.float32)
    denom = np.zeros((width, height), dtype=np.float32)
    row = np.arange(width)
    col = np.arange(height)
    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            i_start = -i if i < 0 else 0
            i_end = width-i if i > 0 else width
            j_start = -j if j < 0 else 0
            j_end = height-j if j > 0 else height
            losses[i_start:i_end, j_start:j_end] += mse[i_start+i:i_end+i, j_start+j:j_end+j]
            denom[i_start:i_end, j_start:j_end] += 1
    return losses / denom


# def windowed_mse_grid_v2(defocus_stack, pred):
#     if globals.window_size % 2 == 0:
#         globals.window_size += 1
#     mse = np.mean((defocus_stack - pred)**2, axis=(0, -1))
#     win_mean = scipy.ndimage.uniform_filter(mse, size=globals.window_size, mode='nearest')
#     return win_mean
    
def objective_full(depth_map, gt_aif, defocus_stack, indices=None, template_A_stack=None, pred=None, last_dpt=None, windowed=False): 

    grid_search = True if np.all(np.isclose(depth_map, depth_map[0][0])) else False
    
    if windowed:
        if grid_search:
            if pred is None:
                pred = forward_model.forward(depth_map, gt_aif, indices=indices, template_A_stack=template_A_stack)
            loss = windowed_mse_grid(defocus_stack, pred)
        else:
            print('here')
            loss = windowed_mse_gss(depth_map, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack)
    else:
        if pred is None:
            pred = forward_model.forward(depth_map, gt_aif, indices=indices, template_A_stack=template_A_stack)
        loss = np.mean((defocus_stack - pred)**2, axis=(0, -1))
    return loss


# def k_min_indices_no_overlap(sorted_indices, k, gss_window=1):
#     width, height, num_Z = sorted_indices.shape
#     k_min_indices = np.zeros((width, height, k), dtype=np.int32) - 1
#     k_min_indices[:,:,0] = sorted_indices[:,:,0]
    
#     # last_z = sorted_indices[:,:,0]
#     kk = np.ones((width, height), dtype=np.int32)
    
#     for z in range(1, num_Z):
#         # mask = (abs(sorted_indices[:,:,z] - last_z) > gss_window) & (kk < k)
#         mask = kk < k
#         for l in range(k):
#             too_close = (abs(sorted_indices[:,:,z] - k_min_indices[:,:,l]) <= gss_window) & (k_min_indices[:,:,l] >= 0)
#             mask = mask & ~too_close
#         if not np.any(mask):
#             continue
#         i, j = np.where(mask)
#         k_min_indices[i, j, kk[i,j]] = sorted_indices[:,:,z][i, j]
#         # last_z[i, j] = sorted_indices[:,:,z][i,j]
#         kk[i, j] += 1
#         if np.all(kk >= k):
#             break

#     assert np.all(k_min_indices >= 0)
        
#     return k_min_indices

# def new_local_min_heuristic_vec(all_losses):
#     # TODO -- takes forever, need to optimize if i actually want to keep this

#     width, height, num_Z = all_losses.shape
#     local_mins = np.zeros((width, height), dtype=int)
#     sorted_indices = np.argsort(all_losses, axis=-1)

#     window_length = int(0.1 * num_Z)
#     if window_length < 5:
#         window_length = 5
#     if window_length % 2 == 0:
#         window_length += 1

#     diff = np.diff(all_losses, axis=-1)
#     # d2 = np.gradient(np.gradient(losses, edge_order=2, axis=-1), edge_order=2, axis=-1)
#     d2_savgol_filter = scipy.signal.savgol_filter(all_losses, axis=-1, window_length=window_length, polyorder=3, deriv=2, delta=1, mode='interp')
#     # cross_indices = np.where(np.sign(diff[:,:,:-1]) * np.sign(diff[:,:,1:]) < 0)[0] + 1

#     pass
    

def new_local_min_heuristic(all_losses):
    # TODO -- takes forever, need to optimize if i actually want to keep this

    width, height, num_Z = all_losses.shape
    local_mins = np.zeros((width, height), dtype=int)
    sorted_indices = np.argsort(all_losses, axis=-1)

    window_length = int(0.1 * num_Z)
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1

    # diff = np.diff(all_losses, axis=-1)
    # d2 = np.gradient(np.gradient(losses, edge_order=2, axis=-1), edge_order=2, axis=-1)
    # d2_savgol_filter = scipy.signal.savgol_filter(all_losses, axis=-1, window_length=window_length, polyorder=3, deriv=2, delta=1, mode='interp')
    # cross_indices = np.where(np.sign(diff[:,:,:-1]) * np.sign(diff[:,:,1:]) < 0)[0] + 1
    


    for i in range(width):
        for j in range(height):
    
            losses = all_losses[i, j]
        
            diff = np.diff(losses)
        
            d2_savgol_filter = scipy.signal.savgol_filter(losses, window_length=window_length, polyorder=3, deriv=2, delta=1, mode='interp')
            
            cross_indices = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0] + 1
                           
            if len(cross_indices) > 0:
                local_min = cross_indices[np.argmax(d2_savgol_filter[cross_indices])]
                if d2_savgol_filter[local_min] > 0: # not positive 
                    local_mins[i,j] = local_min
                    continue

            # just use minimum 
            local_mins[i,j] = sorted_indices[i,j,0]

    return local_mins

def grid_search_opt_k(gt_aif, defocus_stack, indices=None, min_Z = 0.1, max_Z = 10, num_Z = 100, last_dpt=None, gss_window=1, verbose=True, windowed=False):
    # try many values of Z
    Z = np.linspace(min_Z, max_Z, num_Z, dtype=np.float32)

    width, height, num_channels = gt_aif.shape
    if indices is None:
        u, v = forward_model.compute_u_v()
    else:
        u, v, _, _, _ = indices

    all_losses = np.zeros((width, height, num_Z), dtype=np.float32)
    # for i in range(num_Z):
    for i in tqdm(range(num_Z), desc="Grid search".ljust(20), ncols=80, disable=(not verbose)):
        # print(i,'/',num_Z)
        r = forward_model.computer(np.array([[Z[i]]], dtype=np.float32), globals.Df)[...,None,None]
        # print(r)
        # print(r.shape, u.shape, v.shape)
        G, _ = forward_model.computeG(r, u, v)
        # print(G.shape)
        # print(G)
        G = G.squeeze()    
        defocus_stack_pred = np.zeros((G.shape[0], width, height, num_channels), dtype=np.float32)
        for j in range(G.shape[0]): # each focal setting
            kernel = G[j]
            for c in range(num_channels):
                defocus_stack_pred[j,:,:,c] = scipy.ndimage.convolve(gt_aif[:,:,c], kernel, mode='constant')
        
        dpt = np.ones((width,height), dtype=np.float32) * Z[i]
        all_losses[:,:,i] = objective_full(dpt, gt_aif, defocus_stack, indices=indices, pred=defocus_stack_pred, last_dpt=last_dpt, windowed=windowed)
        # all_losses[:,:,i] = objective_full(dpt, gt_aif, defocus_stack, indices=indices, pred=defocus_stack_pred, beta=beta, proxy=proxy, gamma=gamma, similarity_penalty=similarity_penalty, last_dpt=last_dpt, windowed=True)

    sorted_indices = np.argsort(all_losses, axis=2)

    # three diff methods
    min_indices = sorted_indices[:,:,0] # axis = 2
    # k_min_indices = k_min_indices_no_overlap(sorted_indices, k, gss_window=gss_window)
    # k_min_indices = np.expand_dims(new_local_min_heuristic(all_losses), axis=-1)
    
    depth_maps = Z[min_indices]
    # print(k_min_indices.shape, depth_maps.shape)
    
    return depth_maps, Z, min_indices, all_losses



def golden_section_search(Z, argmin_indices, gt_aif, defocus_stack, indices=None, template_A_stack=None,
    window=1, tolerance=1e-5, convergence_error=0, max_iter=100, last_dpt=None, a_b_init=None, verbose=True, windowed=False):
    assert convergence_error >= 0 and convergence_error < 1
    if verbose:
        print("\nGolden-section search...")
    
    # build a grid around each min
    if a_b_init is None:
        num_Z = len(Z)
        a = Z[np.maximum(argmin_indices-window,0)]
        b = Z[np.minimum(argmin_indices+window,num_Z-1)]
    else:
        a, b = a_b_init

    if verbose:
        print('...searching for',(1 - convergence_error)*100,'% convergence')


    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    
    f_c = objective_full(c, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, last_dpt=last_dpt, windowed=windowed)
    f_d = objective_full(d, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, last_dpt=last_dpt, windowed=windowed)
   

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

        # # gss code from wiki
        # c = b - (b - a) * invphi
        # d = a + (b - a) * invphi
        
        # f_c = objective_full(c, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, beta=beta, proxy=proxy, gamma=gamma, similarity_penalty=similarity_penalty, last_dpt=last_dpt)
        # f_d = objective_full(d, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, beta=beta, proxy=proxy, gamma=gamma, similarity_penalty=similarity_penalty, last_dpt=last_dpt)
        # dtype_report("f", f_c=f_c, f_d=f_d)
        
        # tie-safe implementation
        active = (b - a) > tolerance
        go_left = (f_c <= f_d) & active
        go_right = (~go_left) & active

        if np.any(go_left):
            b[go_left] = d[go_left]
            d[go_left] = c[go_left]
            f_d[go_left] = f_c[go_left]
            c[go_left] = b[go_left] - (b[go_left] - a[go_left]) * invphi

        if np.any(go_right):
            a[go_right] = c[go_right]
            c[go_right] = d[go_right]
            f_c[go_right] = f_d[go_right]
            d[go_right] = a[go_right] + (b[go_right] - a[go_right]) * invphi

        if np.any(go_left):
            f_c = objective_full(c, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, last_dpt=last_dpt, windowed=windowed)

        if np.any(go_right):
            f_d = objective_full(d, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, last_dpt=last_dpt, windowed=windowed)
    
        
        # b[go_left] = d[go_left]
        # a[go_right] = c[go_right]
            
        # mask = f_c < f_d
        # b[mask] = d[mask]

        # mask = f_c > f_d
        # a[mask] = c[mask]

        i += 1

    if (i >= max_iter) and verbose:
        print('Failed to converge after',i,'iterations')
        print(np.sum((b - a) <= tolerance) / a.size * 100, '% convergence achieved')

    if verbose:
        print("...done")

    dpt = (b + a) / 2

    if last_dpt is not None:
        mse = objective_full(dpt, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, last_dpt=None, windowed=windowed)
        last_mse = objective_full(last_dpt, gt_aif, defocus_stack, indices=indices, template_A_stack=template_A_stack, last_dpt=None, windowed=windowed)
        dpt = np.where(mse <= last_mse, dpt, last_dpt)

    return dpt
    


# Outlier removal 


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


