import os
import math

import numpy as np
import scipy
import matplotlib.pyplot as plt
from datetime import datetime

import skimage
from PIL import Image
import cv2

import OpenEXR
import struct

import torch
torch.cuda.empty_cache()
import torch_sparse

import globals

def get_worst_diff_pixels(recon, gt, num_worst_pixels = 5, vmin=0.7, vmax=1.9):

    diff = np.abs(recon - gt)

    worst_indices = np.argpartition(diff.ravel(), -num_worst_pixels)[-num_worst_pixels:]
    worst_indices = worst_indices[np.argsort(diff.ravel()[worst_indices])[::-1]]
    # worst_indices = torch.topk(diff.view(-1), num_worst_pixels).indices
    worst_coords = [(idx // diff.shape[1], idx % diff.shape[1]) for idx in worst_indices]

    plt.imshow(recon,vmin=vmin, vmax=vmax)
    plt.scatter([y for x, y in worst_coords], [x for x, y in worst_coords], color='red', marker='x', s=100, label='Worst Diff Pixels')
    plt.title('Worst Difference Pixels Over Image')
    plt.legend()
    plt.show()

    return worst_coords

def plot_single_stack(recon, setting, recon_max = None, title=None):
    # assert len(recon) in [5, 10]
    

    num_images = len(recon)
    cols = 5
    rows = math.ceil(len(recon) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 3))
    if rows == 1:
        axes = np.array([axes])  # make it a list for consistency
    axes = axes.flatten()

    for i in range(num_images):
        if recon_max is None:
            recon_max = recon[i].max()
        axes[i].imshow(recon[i] / recon_max)
        axes[i].axis('off')
        axes[i].set_title(f"{setting[i]:.3f}")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_stacks_side_by_side(gt, recon, setting, title=None):
    assert len(gt) == len(recon)
    
    num_images = len(gt)
    fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))
    
    for i in range(num_images):
        axes[i, 0].imshow(gt[i] / gt[i].max())#, cmap='gray')
        axes[i, 0].axis('off')  # Hide axes
        axes[i, 0].set_title(f"Ground Truth - {setting[i]}")
        
        # if isinstance(recon[i], np.ndarray):
        #     recon_int = recon[i].astype(int)
        # else:
        #     recon_int = recon[i].int()
        # print(recon[i].min(), recon[i].max())
        # print(np.where(np.isnan(recon[i])))
        axes[i, 1].imshow(recon[i] / recon[i].max())#, cmap='gray')
        axes[i, 1].axis('off')  # Hide axes
        axes[i, 1].set_title(f"Reconstructed - {setting[i]}")
    
    if title != None:
        fig.suptitle(title)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    # return fig

def to_uint8(image):
    if isinstance(image, np.ndarray):
        return np.clip(image.astype(int), 0, 255)
    elif isinstance(image, torch.Tensor):
        return torch.clamp(image.int(), 0, 255)
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor")


def plot_compare_rgb(recon, gt):
    
    recon_lot = to_uint8(recon)
    gt_plot = to_uint8(gt)
    
    # Plot the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Plot the first image
    axes[0].imshow(recon_lot)
    axes[0].axis('off')  # Remove axes
    axes[0].set_title("Recon")

    # Plot the second image
    axes[1].imshow(gt_plot)
    axes[1].axis('off')  # Remove axes
    axes[1].set_title("GT")

    # Adjust layout and display
    plt.tight_layout()
    # plt.show()

def plot_compare_greyscale(recon, gt, vmin=None, vmax=None):
    
    if vmin == None:
        vmin = min(recon.min(), gt.min())
    if vmax == None:
        vmax = max(recon.max(), gt.max())
    
    # Plot the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Plot the first image
    axes[0].imshow(recon, vmin=vmin, vmax=vmax)
    axes[0].axis('off')  # Remove axes
    axes[0].set_title("Recon")

    # Plot the second image
    axes[1].imshow(gt, vmin=vmin, vmax=vmax)
    axes[1].axis('off')  # Remove axes
    axes[1].set_title("GT")

    # Adjust layout and display
    plt.tight_layout()
    # plt.show()

def load_NYUv2_dpt(path_to_file, resize_frac=2):
    dpt = skimage.io.imread(path_to_file).astype(np.float32)
    width, height = dpt.shape
    dpt /= 1e4  # scale depth values
    dpt = np.clip(dpt, 0.1, 10.0) # optionally clip
    # print(width, height)
    # resize with anti-aliasing
    if resize_frac != 1:
        dpt = skimage.transform.resize(
            dpt,
            output_shape=(width//resize_frac,height//resize_frac), # (height, width)
            order=1,                  # bilinear interpolation
            anti_aliasing=True,
            preserve_range=True       # don't normalize to [0, 1]
        )
    # convert to torch tensor
    return dpt#torch.from_numpy(dpt)

def load_NYUv2_aif(path_to_file, resize_frac=2):
    aif = skimage.io.imread(path_to_file).astype(np.float32) / 255.0
    width, height, _ = aif.shape
    # print(width, height)
    # resize if needed
    if resize_frac != 1:
        aif = skimage.transform.resize(
            aif,
            output_shape=(width//resize_frac,height//resize_frac), # (height, width)
            order=1,                  # bilinear interpolation
            anti_aliasing=True,
            preserve_range=True       # keep values in [0, 255] if original was uint8
        )
    # convert to torch tensor
    return aif#torch.from_numpy(aif)

def load_single_sample(sample='0045', set='train', fs=5, res='half'):
    assert fs == 5 or fs == 10
    assert res == 'full' or res == 'half'
    
    ext = str(fs)
    if fs == 10:
        globals.Df = torch.tensor([0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6])  # m 
    if res == 'half':
        ext += '_halfres'
    
    data_path = os.path.join(os.getcwd(),'NYUv2')#+str(fs))
    img_name = sample + '.png'

    # dpt = Image.open(os.path.join(data_path,set+'_depth',img_name))
    # if res == 'half':
    #     # dpt = dpt.resize((640//2, 480//2))
    #     dpt = np.asarray(dpt, dtype=np.float32)
    #     dpt = np.clip(np.asarray(dpt, dtype=np.float32) / 1e4, 0.1, 10)
    #     dpt = cv2.resize(dpt, (640//2, 480//2))
    # dpt = torch.tensor(np.asarray(dpt, dtype=np.float32))# / 1e4
    
    # dpt = skimage.io.imread(os.path.join(data_path, set+'_depth', img_name)).astype(np.float32)
    # dpt /= 1e4  # scale depth values
    # dpt = np.clip(dpt, 0.1, 10.0) # optionally clip
    # if res == 'half':
    #     # resize with anti-aliasing
    #     dpt = skimage.transform.resize(
    #         dpt,
    #         output_shape=(480//2,640//2), # (height, width)
    #         order=1,                  # bilinear interpolation
    #         anti_aliasing=True,
    #         preserve_range=True       # don't normalize to [0, 1]
    #     )
    # # convert to torch tensor
    # dpt = torch.from_numpy(dpt)
    resize_frac = 2 if res == 'half' else 1
    dpt = load_NYUv2_dpt(os.path.join(data_path, set+'_depth', img_name), resize_frac=resize_frac)

    # aif = cv2.cvtColor(cv2.imread(os.path.join(data_path,set+'_rgb',img_name)), cv2.COLOR_BGR2RGB)
    # if res == 'half':
    #     aif = cv2.resize(aif,(640//2,480//2))
    # aif = torch.tensor(aif)
    # load RGB image (automatically returns float in [0, 255] if it's uint8)
    # aif = skimage.io.imread(os.path.join(data_path, set+'_rgb', img_name)).astype(np.float32) / 255.0
    # # resize if needed
    # if res == 'half':
    #     aif = skimage.transform.resize(
    #         aif,
    #         output_shape=(480//2,640//2), # (height, width)
    #         order=1,                  # bilinear interpolation
    #         anti_aliasing=True,
    #         preserve_range=True       # keep values in [0, 255] if original was uint8
    #     )
    # # convert to torch tensor
    # aif = torch.from_numpy(aif)
    aif = load_NYUv2_aif(os.path.join(data_path, set+'_rgb', img_name), resize_frac=resize_frac)

    #files = os.listdir(os.path.join(data_path,set+'_fs'+ext))
    #files = sorted(files)
    #gt_defocus_stack = [cv2.cvtColor(cv2.imread(os.path.join(data_path,set+'_fs'+ext,file)), cv2.COLOR_BGR2RGB) for file in files if sample + '_' in file]
    #assert len(gt_defocus_stack) == fs
    return aif, dpt#, gt_defocus_stack


def load_single_sample_DefocusNet(sample='000373'):
    data_path = os.path.join(os.getcwd(),'DefocusNet')

    # get depth map
    exr_name = sample + 'Dpt.exr'
    dpt = load_dpt_DefocusNet(os.path.join(data_path, exr_name))

    # get defocus stack
    filenames = [f"{sample}_{i:02d}All.tif" for i in range(5)]
    paths = sorted([os.path.join(data_path, name) for name in filenames if os.path.exists(os.path.join(data_path, name))])
    assert len(paths) == 5 # expecting 5 images in focal stack
    
    defocus_stack = []
    for path_to_image in paths:
        im = np.array(Image.open(path_to_image), dtype=np.float32) / 255.
        defocus_stack.append(im)
    defocus_stack = np.stack(defocus_stack, 0)
    
    return dpt, defocus_stack


def load_dpt_DefocusNet(img_dpt_path):
    # modified from defocus-net/source/util_func.py
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.fromstring(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    return dpt

def read_bin_file(path_to_file):
    with open(path_to_file, "rb") as f:
        # Read header
        t = np.frombuffer(f.read(1), np.uint8)[0]     # type code
        h = struct.unpack("i", f.read(4))[0]          # height
        w = struct.unpack("i", f.read(4))[0]          # width

        # Map OpenCV type code to NumPy dtype
        cv_depth = t & 7
        depth_map = {
            0: np.uint8,
            1: np.int8,
            2: np.uint16,
            3: np.int16,
            4: np.int32,
            5: np.float32,
            6: np.float64,
        }
        dtype = depth_map[cv_depth]

        # Read the rest of the data
        data = np.frombuffer(f.read(), dtype=dtype)
        mat = data.reshape(h, w)

    return mat.astype(np.float32)

def load_single_sample_MobileDepth(example_name="keyboard"):
    assert example_name in ["keyboard", "bottles", "fruits", "metals", "plants", "telephone", "window", "largemotion", "smallmotion", "zeromotion", "balls"]
    # no calibration data: "bucket", "kitchen", 

    # retrieve aligned focal stack
    data_path = os.path.join(os.getcwd(),'MobileDepth')
    focal_stack_dir = os.path.join(data_path, 'aligned-focus-stack', 'Aligned')

    example_directory = None
    for name in os.listdir(focal_stack_dir):
        subdir = os.path.join(focal_stack_dir, name)
        if os.path.isdir(subdir):
            candidate = os.path.join(subdir, example_name)
            if os.path.isdir(candidate):
                example_directory = candidate
                print("Found at:", os.path.abspath(candidate))
                break
    
    assert example_directory is not None

    defocus_stack = []
    for filename in os.listdir(example_directory):
        if filename.startswith("a") and filename.endswith(".jpg"):
            defocus_stack.append(os.path.join(example_directory, filename))
    defocus_stack = sorted(defocus_stack)

    for i in range(len(defocus_stack)):
        defocus_stack[i] = np.array(Image.open(defocus_stack[i]), dtype=np.float32) / 255.

    defocus_stack = np.stack(defocus_stack, 0)
    
    # camera parameters

    # resolve weird naming inconsistencies
    calib_name = example_name
    if example_name == "largemotion":
        calib_name = "GTLarge"
    if example_name == "smallmotion":
        calib_name = "GTSmall"
    if example_name == "zeromotion":
        calib_name = "GT"
    if example_name == "metals":
        calib_name = "metal"
        
    calib_dir = os.path.join(data_path, 'photos-calibration-results', 'calibration', calib_name)

    calib_file = os.path.join(calib_dir, "calibrated.txt")
    focal_depths = []
    apertures = []
    focal_length = None
    
    with open(calib_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()] # removes blank lines
    
    # last line is focal length
    focal_length = float(lines[-1])
    
    # all previous lines: focal depth / aperture
    for line in lines[:-1]:
        parts = line.split()
        if len(parts) >= 2:
            focal_depths.append(float(parts[0]))
            apertures.append(float(parts[1]))

    assert len(set(apertures)) == 1

    # set globals
    globals.Df = np.array(focal_depths) * 0.0254 # inches --> meters
    globals.f = focal_length * 0.0254 # inches --> meters
    globals.D = globals.f / set(apertures).pop() # f-number

    
    
    print("Focal depths:", globals.Df)
    print("Apertures:", globals.D)
    print("Focal length:", globals.f)
    # print("Pixel size:", globals.ps)
    print(defocus_stack.shape)
    # their depth result
    dpt_res_file = os.path.join(calib_dir, "depth_var.bin")
    mat = read_bin_file(dpt_res_file)
    # with open(dpt_res_file, "rb") as f:
    #     # Read header
    #     t = np.frombuffer(f.read(1), np.uint8)[0]     # type code
    #     h = struct.unpack("i", f.read(4))[0]          # height
    #     w = struct.unpack("i", f.read(4))[0]          # width

    #     # Map OpenCV type code to NumPy dtype
    #     cv_depth = t & 7
    #     depth_map = {
    #         0: np.uint8,
    #         1: np.int8,
    #         2: np.uint16,
    #         3: np.int16,
    #         4: np.int32,
    #         5: np.float32,
    #         6: np.float64,
    #     }
    #     dtype = depth_map[cv_depth]

    #     # Read the rest of the data
    #     data = np.frombuffer(f.read(), dtype=dtype)
    #     mat = data.reshape(h, w)

    dpt_result = 1.0 / mat # invert 
    mn, mx = np.min(dpt_result), np.max(dpt_result)
    dpt_result = (dpt_result - mn) / (mx - mn)

    
    scale_file = os.path.join(calib_dir, "scaleMatrix.bin")
    scale_mat = read_bin_file(scale_file)

    return defocus_stack, dpt_result, scale_mat

def compute_RMS(pred, gt):
    diff_sq = (pred - gt) ** 2
    return np.sqrt(np.mean(diff_sq))

def compute_Rel(pred, gt):
    # absolute Relative error: mean(|pred - gt| / gt).
    rel = np.abs(pred - gt) / (np.abs(gt) + 1e-8)
    return np.mean(rel)


def compute_AbsRel(pred, gt):
    # absolute Relative error: mean(|pred - gt| / gt).
    rel = np.abs(pred - gt) / (gt + 1e-8)
    return np.mean(rel)

def compute_accuracy_metrics(pred, gt):
    # Returns δ1, δ2, δ3 as defined by Eigen et al.
    # δ_k = fraction of pixels with max(pred/gt, gt/pred) < 1.25^k
    
    ratio = np.maximum(pred / (gt+1e-8), gt / (pred+1e-8))
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25**2)
    delta3 = np.mean(ratio < 1.25**3)
    
    return {"delta1": delta1, "delta2": delta2, "delta3": delta3}

def save_dpt(path, fn, dpt):
    dpt_scaled = (dpt * 1e4).astype(np.float32)
    skimage.io.imsave(os.path.join(path, fn + '.tiff'), dpt_scaled)

def save_aif(path, fn, aif):
    skimage.io.imsave(os.path.join(path, fn + '.tiff'), aif.squeeze().astype(np.float32))

def load_sample_image(fs=5, res='half'):
    assert fs == 5 or fs == 10
    assert res == 'full' or res == 'half'
    
    ext = str(fs)
    if fs == 10:
        globals.Df = np.array([0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6], dtpe=np.float32)  # m 
    if res == 'half':
        ext += '_halfres'
    
    data_path = os.path.join(os.getcwd(),'NYUv2_single'+str(fs))
    img_name = '0045.png'

    # si et al.
    # dpt = Image.open(os.path.join(data_path,'test_depth',img_name))
    # dpt = np.asarray(dpt, dtype=np.float32)
    # dpt = np.clip(np.asarray(dpt, dtype=np.float32) / 1e4, 0.1, 10)
    # if res == 'half':
    #     dpt = cv2.resize(dpt,(640//2, 480//2))
    # dpt = torch.from_numpy(dpt)
    
    # dpt = skimage.io.imread(os.path.join(data_path, 'test_depth', img_name)).astype(np.float32)
    # dpt /= 1e4  # scale depth values
    # dpt = np.clip(dpt, 0.1, 10.0) # optionally clip
    # if res == 'half':
    #     # resize with anti-aliasing
    #     dpt = skimage.transform.resize(
    #         dpt,
    #         output_shape=(480//2,640//2), # (height, width)
    #         order=1,                  # bilinear interpolation
    #         anti_aliasing=True,
    #         preserve_range=True       # don't normalize to [0, 1]
    #     )
    # # convert to torch tensor
    # dpt = torch.from_numpy(dpt)
    resize_frac = 2 if res == 'half' else 1
    dpt = load_NYUv2_dpt(os.path.join(data_path, 'test_depth', img_name), resize_frac=resize_frac)

    # si et al.
    # aif = cv2.cvtColor(cv2.imread(os.path.join(data_path,'test_rgb',img_name)), cv2.COLOR_BGR2RGB)
    # if res == 'half':
    #     aif = cv2.resize(aif,(640//2,480//2))
    # aif = torch.from_numpy(aif/255.).type(torch.float32).contiguous()

    # load RGB image (automatically returns float in [0, 255] if it's uint8)
    # aif = skimage.io.imread(os.path.join(data_path, 'test_rgb', img_name)).astype(np.float32) / 255.0
    # print(aif[100,100])
    # # resize if needed
    # if res == 'half':
    #     aif = skimage.transform.resize(
    #         aif,
    #         output_shape=(480//2,640//2), # (height, width)
    #         order=1,                  # bilinear interpolation
    #         anti_aliasing=True,
    #         preserve_range=True       # keep values in [0, 255] if original was uint8
    #     )
    # # convert to torch tensor
    # aif = torch.from_numpy(aif)
    aif = load_NYUv2_aif(os.path.join(data_path, 'test_rgb', img_name), resize_frac=resize_frac)
    # print('after',aif.min(), aif.max())
    
    files = os.listdir(os.path.join(data_path,'test_fs'+ext))
    files = sorted(files)
    gt_defocus_stack = [cv2.cvtColor(cv2.imread(os.path.join(data_path,'test_fs'+ext,file)), cv2.COLOR_BGR2RGB) for file in files]
    assert len(gt_defocus_stack) == fs
    
    return aif, dpt, gt_defocus_stack


def create_experiment_folder(experiment_name, base_folder="experiments"):
    # Create "experiments" folder and a timestamped subfolder for the experiment
    # base_folder = "experiments"
    # if not os.path.exists(base_folder):
    #     os.makedirs(base_folder)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{experiment_name}_{timestamp}"
    experiment_folder = os.path.join(base_folder, folder_name)
    os.makedirs(experiment_folder)
    
    print(f"Created experiment folder: {experiment_folder}")
    return experiment_folder


def format_number(x):
    return f"{x:.6e}" if abs(x) < 1e-3 else f"{x:.6f}"  # Use scientific notation if < 0.001

def update_max_kernel_size(new_value):
    globals.MAX_KERNEL_SIZE = new_value


def kernel_size_heuristic(width, height):
    size = round(0.039 * (width + height) / 2)
    size = max(7, size)
    if size % 2 == 0:
        return size + 1
    return size


def strongest_curvature_region(i, j, all_losses, window=11):
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window % 2 == 0:
        window += 1
        
    losses = all_losses[i, j]
    d2 = np.gradient(np.gradient(losses, edge_order=2), edge_order=2)

    # lets try to get a more accurate second deriv

    # # approach 1: cdsm
    # #First derivatives:
    # df = np.diff(losses)
    # cf = np.convolve(losses, [1,-1],'same')
    # gf = ndimage.gaussian_filter1d(losses, sigma=1, order=1, mode='wrap')
    
    # #Second derivatives:
    # ddf = np.diff(losses, 2)
    # ccf = np.convolve(losses, [1, -2, 1],'same')
    # ggf = ndimage.gaussian_filter1d(losses, sigma=1, order=2, mode='wrap')


    # approach 2:
    d_savgol_filter = scipy.signal.savgol_filter(losses, window_length=11, polyorder=3, deriv=1, delta=1, mode='interp')
    d2_savgol_filter = scipy.signal.savgol_filter(losses, window_length=11, polyorder=3, deriv=2, delta=1, mode='interp')


    kernel = np.ones(window) / window
    avg_d2 = np.convolve(d2, kernel, mode="same")

    # region with most negative average curvature
    # offset = (window - 1) // 2
    idx_center_region = int(np.argmax(d2)) # concave up/positive second deriv (local min region)
    # idx_center_region = min_idx + offset
    return idx_center_region, d2, d_savgol_filter, d2_savgol_filter
    
def plot_grid_search_on_pixel(i, j, Z, all_losses, gt_dpt=None, k_min_indices=None):

    plt.figure(figsize=(10,5))

    plt.plot(Z, all_losses[i, j, :], label='Loss Curve',
            linestyle='-', marker='.', markersize=4, color='black')

    if gt_dpt is not None:
        # get interpolated loss value
        # idx_b = np.searchsorted(Z, gt_dpt[i, j])
        # idx_a = idx_b - 1
        # c = (gt_dpt[i,j] - Z[idx_a]) / (Z[idx_b] - Z[idx_a])
        # interpolated_val = c * Z[idx_a] + (1-c) * Z[idx_b]
        plt.scatter([gt_dpt[i,j]], [0],
                color='red', marker='x', s=100, label='Ground Truth Depth')
            
    if k_min_indices is None:
        min_loss_idx = np.argmin(all_losses[i,j])
        plt.scatter([Z[min_loss_idx]], [all_losses[i,j,min_loss_idx]], 
                color='green', marker='x', s=100, label='Depth with Min Loss')
    else:
        for k in range(len(k_min_indices)):
            plt.scatter([Z[k_min_indices[k]]], [all_losses[i,j,k_min_indices[k]]],
                        color='green', marker='x', s=100, label='Depth with '+str(k)+' Min Loss')

    idx_center_point, d2, d_savgol_filter, d2_savgol_filter = strongest_curvature_region(i, j, all_losses, window=5)
    # plt.scatter([Z[idx_center_point]], [all_losses[i,j,idx_center_point]], 
    #         color='blue', marker='x', s=100, label='Depth with Most Neg 2nd Deriv')
    
    # d = np.gradient(all_losses[i,j], edge_order=2)
    # local_min_candidates = np.where(abs(d_savgol_filter) < 0.05)[0]
    # local_min = local_min_candidates[np.argmax(d2_savgol_filter[local_min_candidates])]
    # for idx in np.argsort(abs(d)):
    #     if d2[idx] > 0:
    #         local_min = idx
    #         break
    plt.plot(Z, d_savgol_filter, label='1st deriv',
            linestyle='-', marker='.', markersize=4, color='cyan')
    plt.plot(Z, d2_savgol_filter, label='2nd deriv',
            linestyle='-', marker='.', markersize=4, color='blue')
    plt.plot(Z[1:]-(3-0.1)/100/2, np.diff(all_losses[i,j]), label='diff',
            linestyle='-', marker='.', markersize=4, color='green')

    # find where the derivative passes over to 0
    diff = np.diff(all_losses[i,j])
    
    cross_indices = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0] + 1
                   
    # cross_indices = np.where(np.sign(d_savgol_filter[:-1]) * np.sign(d_savgol_filter[1:]) < 0)[0]
    for idx in cross_indices:
        # print(idx, Z[idx], d_savgol_filter[idx], d_savgol_filter[idx+1])
        print(idx, Z[idx], diff[idx], diff[idx+1])
    if len(cross_indices) > 0:
        local_min = cross_indices[np.argmax(d2_savgol_filter[cross_indices])]
        plt.scatter([Z[local_min]], [all_losses[i,j,local_min]], color='purple', marker='x', s=100, label='local min')
            
    plt.xticks(Z[::2], labels=np.round(Z[::2], 2), rotation=45)
    plt.xlabel('Depth (m)')
    plt.ylabel('MSE between Predicted and Ground Truth Defocus Stack')
    plt.title(f'Pixel at {(i, j)}')
    
    plt.legend()
    
    plt.tight_layout()
    plt.show()
