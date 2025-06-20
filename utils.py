import os
import math

import numpy as np
import scipy
import matplotlib.pyplot as plt
from datetime import datetime

import skimage
from PIL import Image
import cv2

import torch
torch.cuda.empty_cache()
import torch_sparse

import globals

def get_worst_diff_pixels(recon, gt, num_worst_pixels = 5, vmin=0.7, vmax=1.9):

    diff = torch.abs(recon - gt)
    
    worst_indices = torch.topk(diff.view(-1), num_worst_pixels).indices
    worst_coords = [(idx // diff.shape[1], idx % diff.shape[1]) for idx in worst_indices]

    plt.imshow(recon.numpy(),vmin=vmin, vmax=vmax)
    plt.scatter([y.item() for x, y in worst_coords], [x.item() for x, y in worst_coords], color='red', marker='x', s=100, label='Worst Diff Pixels')
    plt.title('Worst Difference Pixels Over Image')
    plt.legend()
    plt.show()

    return worst_coords

def plot_single_stack(recon, setting, recon_max = None, title=None):
    assert len(recon) in [5, 10]
    

    num_images = len(recon)
    cols = 5
    rows = round(len(recon) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 3))
    if rows == 1:
        axes = np.array([axes])  # make it a list for consistency
    axes = axes.flatten()

    for i in range(num_images):
        if recon_max is None:
            recon_max = recon[i].max()
        axes[i].imshow(recon[i] / recon_max)
        axes[i].axis('off')
        axes[i].set_title(f"{setting[i]}")

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
        
    dpt = torch.tensor(np.asarray(dpt, dtype=np.float32))# / 1e4
    dpt = skimage.io.imread(os.path.join(data_path, set+'_depth', img_name)).astype(np.float32)
    dpt /= 1e4  # scale depth values
    dpt = np.clip(dpt, 0.1, 10.0) # optionally clip
    if res == 'half':
        # resize with anti-aliasing
        dpt = skimage.transform.resize(
            dpt,
            output_shape=(480//2,640//2), # (height, width)
            order=1,                  # bilinear interpolation
            anti_aliasing=True,
            preserve_range=True       # don't normalize to [0, 1]
        )
    # convert to torch tensor
    dpt = torch.from_numpy(dpt)

    # aif = cv2.cvtColor(cv2.imread(os.path.join(data_path,set+'_rgb',img_name)), cv2.COLOR_BGR2RGB)
    # if res == 'half':
    #     aif = cv2.resize(aif,(640//2,480//2))
    # aif = torch.tensor(aif)
    # load RGB image (automatically returns float in [0, 255] if it's uint8)
    aif = skimage.io.imread(os.path.join(data_path, set+'_rgb', img_name)).astype(np.float32) / 255.0
    # resize if needed
    if res == 'half':
        aif = skimage.transform.resize(
            aif,
            output_shape=(480//2,640//2), # (height, width)
            order=1,                  # bilinear interpolation
            anti_aliasing=True,
            preserve_range=True       # keep values in [0, 255] if original was uint8
        )
    # convert to torch tensor
    aif = torch.from_numpy(aif)

    #files = os.listdir(os.path.join(data_path,set+'_fs'+ext))
    #files = sorted(files)
    #gt_defocus_stack = [cv2.cvtColor(cv2.imread(os.path.join(data_path,set+'_fs'+ext,file)), cv2.COLOR_BGR2RGB) for file in files if sample + '_' in file]
    #assert len(gt_defocus_stack) == fs
    return aif, dpt#, gt_defocus_stack


def load_sample_image(fs=5, res='half'):
    assert fs == 5 or fs == 10
    assert res == 'full' or res == 'half'
    
    ext = str(fs)
    if fs == 10:
        globals.Df = torch.tensor([0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6])  # m 
    if res == 'half':
        ext += '_halfres'
    
    data_path = os.path.join(os.getcwd(),'NYUv2_single'+str(fs))
    img_name = '0045.png'

    # si et al.
    dpt = Image.open(os.path.join(data_path,'test_depth',img_name))
    dpt = np.asarray(dpt, dtype=np.float32)
    dpt = np.clip(np.asarray(dpt, dtype=np.float32) / 1e4, 0.1, 10)
    if res == 'half':
        dpt = cv2.resize(dpt,(640//2, 480//2))
    dpt = torch.from_numpy(dpt)
    
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

    # si et al.
    aif = cv2.cvtColor(cv2.imread(os.path.join(data_path,'test_rgb',img_name)), cv2.COLOR_BGR2RGB)
    if res == 'half':
        aif = cv2.resize(aif,(640//2,480//2))
    aif = torch.from_numpy(aif/255.).type(torch.float32).contiguous()

    # # load RGB image (automatically returns float in [0, 255] if it's uint8)
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
    # print('after',aif.min(), aif.max())
    
    files = os.listdir(os.path.join(data_path,'test_fs'+ext))
    files = sorted(files)
    gt_defocus_stack = [cv2.cvtColor(cv2.imread(os.path.join(data_path,'test_fs'+ext,file)), cv2.COLOR_BGR2RGB) for file in files]
    assert len(gt_defocus_stack) == fs
    
    return aif, dpt, gt_defocus_stack


def create_experiment_folder(experiment_name):
    # Create "experiments" folder and a timestamped subfolder for the experiment
    base_folder = "experiments"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
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