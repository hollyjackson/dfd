import numpy as np
import matplotlib.pyplot as plt
import scipy
import math

import forward_model
import globals

import tqdm

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

def windowed_mse_grid(defocus_stack, pred):
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
#     # more efficient version, forgot to use in final version, TODO: put back in but double check its producing the same exact results 
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

def grid_search(gt_aif, defocus_stack, indices=None, min_Z = 0.1, max_Z = 10, num_Z = 100, last_dpt=None, gss_window=1, verbose=True, windowed=False):
    # try many values of Z
    Z = np.linspace(min_Z, max_Z, num_Z, dtype=np.float32)

    width, height, num_channels = gt_aif.shape
    if indices is None:
        u, v = forward_model.compute_u_v()
    else:
        u, v, _, _, _ = indices

    all_losses = np.zeros((width, height, num_Z), dtype=np.float32)
    # for i in range(num_Z):
    for i in tqdm.tqdm(range(num_Z), desc="Grid search".ljust(20), ncols=80, disable=(not verbose)):
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
    


