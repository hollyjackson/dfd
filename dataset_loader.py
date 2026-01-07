import os
import math

import numpy as np
import scipy
import matplotlib.pyplot as plt
from datetime import datetime

import skimage
from PIL import Image, ImageOps
import cv2

import OpenEXR
import struct
import exifread
from scipy.io import loadmat

import torch
torch.cuda.empty_cache()
import torch_sparse

import globals

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

    resize_frac = 2 if res == 'half' else 1
    dpt = load_NYUv2_dpt(os.path.join(data_path, set+'_depth', img_name), resize_frac=resize_frac)

    aif = load_NYUv2_aif(os.path.join(data_path, set+'_rgb', img_name), resize_frac=resize_frac)

    #files = os.listdir(os.path.join(data_path,set+'_fs'+ext))
    #files = sorted(files)
    #gt_defocus_stack = [cv2.cvtColor(cv2.imread(os.path.join(data_path,set+'_fs'+ext,file)), cv2.COLOR_BGR2RGB) for file in files if sample + '_' in file]
    #assert len(gt_defocus_stack) == fs
    return aif, dpt#, gt_defocus_stack


def load_single_sample_MobileDepth(example_name="keyboard", res='full'):
    assert example_name in ["keyboard", "bottles", "fruits", "metals", "plants", "telephone", "window", "largemotion", "smallmotion", "zeromotion", "balls"]
    assert res == 'full' or res == 'half'

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

    resize_frac = 2 if res == 'half' else 1

    for i in range(len(defocus_stack)):
        defocus_stack[i] = np.array(Image.open(defocus_stack[i]), dtype=np.float32) / 255.
        width, height, _ = defocus_stack[i].shape
        if resize_frac != 1:
            defocus_stack[i] = skimage.transform.resize(
                defocus_stack[i],
                output_shape=(width//resize_frac,height//resize_frac), # (height, width)
                order=1,                  # bilinear interpolation
                anti_aliasing=True,
                preserve_range=True       # don't normalize to [0, 1]
            )

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
    globals.Df = np.array(focal_depths, dtype=np.float32) # unitless
    globals.f = focal_length # unitless
    globals.D = set(apertures).pop() # unitless confirmed by Supasorn

    
    
    print("Focal depths:", globals.Df)
    print("Apertures:", globals.D)
    print("Focal length:", globals.f)
    # print("Pixel size:", globals.ps)
    print(defocus_stack.shape)
    # their depth result
    dpt_res_file = os.path.join(calib_dir, "depth_var.bin")
    mat = utils.read_bin_file(dpt_res_file)
    plt.imshow(mat)
    plt.colorbar()
    plt.show()
    print(mat.min(), mat.max())
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

    dpt_result = mat.astype(np.float32)
    # dpt_result = 1.0 / mat # invert 
    # mn, mx = np.min(dpt_result), np.max(dpt_result)
    # dpt_result = (dpt_result - mn) / (mx - mn)

    
    scale_file = os.path.join(calib_dir, "scaleMatrix.bin")
    scale_mat = utils.read_bin_file(scale_file)

    return defocus_stack, dpt_result, scale_mat




def load_single_sample_Make3D(img_name = "img-math7-p-282t0.jpg", data_dir = "/data/holly_jackson/", split='train'):
    assert split in ['test', 'train']
    img_subdir = 'Test134Img' if split == 'test' else 'Train400Img'
    dpt_subdir = 'Test134Depth' if split == 'test' else 'Train400Depth'
    
    img_filename = os.path.join(data_dir, 'Make3D', img_subdir, img_name)

    # filename = "img-math7-p-282t0.jpg"
    with open(img_filename, 'rb') as f:
        tags = exifread.process_file(f)

    # print a few useful fields
    print("Camera:", tags.get("Image Make"), tags.get("Image Model"))
    focal_length_mm = utils.exif_to_float(tags.get("EXIF FocalLength"))
    globals.f = focal_length_mm * 1e-3
    print("Focal length (m):", globals.f)
    f_number = utils.exif_to_float(tags.get("EXIF FNumber"))
    print("F-number:", f_number)
    globals.D = globals.f / f_number
    print("Aperture diameter (m):", globals.D)

    # load image 
    if img_name == "img-op29-p-295t000.jpg": # this img is broken
        im = Image.open(img_filename)
        im = ImageOps.exif_transpose(im)
        im_rgb = im.convert("RGB")
        arr = np.array(im_rgb, dtype=np.float32) / 255.
    else:
        aif = np.array(Image.open(img_filename), dtype=np.float32) / 255.
    
    image_width_px = aif.shape[0]   # Make3D image width
    # resize data as recommended by Saxena papers and Gur (460 × 345)
    # plt.imshow(aif)
    # plt.show()
    print(aif.shape)

    aif = skimage.transform.resize(
        aif,
        output_shape=(460, 345), # (height, width)
        order=1,                  # bilinear interpolation
        anti_aliasing=True,
        preserve_range=True       # keep values in [0, 255] if original was uint8
    )
    # plt.imshow(aif)
    # plt.show()
    # print(aif.shape)
    
    # https://www.digicamdb.com/specs/canon_powershot-s40/
    # 1/1.8" (~ 7.11 x 5.33 mm)  
    sensor_width_m = 7.11e-3  # Canon PowerShot S40, lookup value
    globals.ps = sensor_width_m / image_width_px * (image_width_px / aif.shape[0])
    # print(sensor_width_m,'/',image_width_px, '* (', image_width_px,'/', aif.shape[0],')')
    print("Pixel size (m/pix):", globals.ps)

    part = img_name.split("img-")[1].split(".jpg")[0]
    dpt_name = "depth_sph_corr-" + part + ".mat"
    dpt_filename = os.path.join(data_dir, 'Make3D', dpt_subdir, dpt_name)
    
    data = loadmat(dpt_filename)
    
    # print(data.keys())          # list all variable names
    dpt = np.array(data["Position3DGrid"], dtype=np.float32)    # access one variable
    # print(data['__header__'], data['__version__'], data['__globals__'])
    # print(dpt.shape, dpt.dtype)
    # assert np.all(dpt[:,:,:3] == 0)
    # for idx in range(3):
    #     plt.imshow(dpt[:,:,idx])
    #     plt.colorbar()
    #     plt.show()
    dpt = dpt[:,:,3]
    
    # plt.imshow(dpt)
    # plt.colorbar()
    # plt.show()

    print('GT DPT Range:', dpt.min(),'-',dpt.max())
    
    return aif, dpt



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
