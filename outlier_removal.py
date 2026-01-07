import numpy as np
import matplotlib.pyplot as plt

import globals


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
