"""Dataset loaders for depth-from-defocus experiments.

Three datasets are supported:

NYUv2
    Indoor RGB-D images.  Camera parameters are fixed; call
    ``globals.init_NYUv2()`` before using these loaders.

Make3D
    Outdoor images with LIDAR depth.  Camera parameters (focal length,
    aperture, pixel size) are read from EXIF and set on ``globals`` by
    the loader.

MobileDepth (mobile phone focal stacks)
    Phone-captured focal stacks with per-scene calibration files.
    Camera parameters and focus distances are set on ``globals`` by the
    loader.
"""
import os

import numpy as np
import skimage
from PIL import Image, ImageOps
import cv2

import exifread
from scipy.io import loadmat

import globals
import utils

# ---------------------------------------------------------------------------
# NYUv2
# ---------------------------------------------------------------------------

def load_NYUv2_dpt(path_to_file, resize_frac=2):
    """Load a NYUv2 depth image and convert to metres.

    Pixel values are stored in 0.1 mm units; dividing by 1e4 converts to
    metres.  Depth is clipped to [0.1, 10.0] m after scaling.

    Parameters
    ----------
    path_to_file : str
        Path to the depth image (.tiff or .png).
    resize_frac : int, optional
        Downsampling factor (default 2 for half resolution).

    Returns
    -------
    dpt : ndarray, shape (width, height)
        Depth map in metres, dtype float32.
    """
    dpt = skimage.io.imread(path_to_file).astype(np.float32)
    width, height = dpt.shape
    dpt /= 1e4
    dpt = np.clip(dpt, 0.1, 10.0)
    if resize_frac != 1:
        dpt = skimage.transform.resize(
            dpt,
            output_shape=(width // resize_frac, height // resize_frac),
            order=1, # bilinear interpolation
            anti_aliasing=True,
            preserve_range=True,
        )
    return dpt

def load_NYUv2_aif(path_to_file, resize_frac=2):
    """Load a NYUv2 RGB image and normalise to [0, 1].

    Parameters
    ----------
    path_to_file : str
        Path to the RGB image (.png).
    resize_frac : int, optional
        Downsampling factor (default 2 for half resolution).

    Returns
    -------
    aif : ndarray, shape (width, height, 3)
        All-in-focus image [0, 255], dtype float32.
    """
    aif = skimage.io.imread(path_to_file).astype(np.float32) #/ 255.0
    width, height, _ = aif.shape
    if resize_frac != 1:
        aif = skimage.transform.resize(
            aif,
            output_shape=(width // resize_frac, height // resize_frac),
            order=1, # bilinear interpolation
            anti_aliasing=True,
            preserve_range=True, # preserves input range
        )
    return aif

def load_single_sample_NYUv2(data_dir='data', sample='0045', set='train', res='half'):
    """Load a single NYUv2 RGB-D sample.

    If fs=10, sets globals.Df to the 10-plane focus distance array as a
    side effect.

    Parameters
    ----------
    data_dir : str
        Root data directory (relative to cwd), expected to contain 'NYUv2/'.
    sample : str
        Sample name without extension, e.g. '0045'.
    set : str
        Dataset split: 'train' or 'test'.
    res : str
        Resolution: 'half' (default) or 'full'.

    Returns
    -------
    aif : ndarray, shape (width, height, 3)
        All-in-focus image [0, 255].
    dpt : ndarray, shape (width, height)
        Depth map in metres.
    """
    assert res in ('full', 'half')

    data_path = os.path.join(os.getcwd(), data_dir, 'NYUv2')
    resize_frac = 2 if res == 'half' else 1

    dpt = load_NYUv2_dpt(os.path.join(data_path, set + '_depth', sample + '.tiff'), resize_frac=resize_frac)
    aif = load_NYUv2_aif(os.path.join(data_path, set + '_rgb', sample + '.png'), resize_frac=resize_frac)

    return aif, dpt




# ---------------------------------------------------------------------------
# Make3D
# ---------------------------------------------------------------------------

def _load_make3d_camera_params(img_filename):
    """Read EXIF metadata from a Make3D image; sets globals.f and globals.D.

    Parameters
    ----------
    img_filename : str
        Path to the Make3D JPEG image.
    """
    with open(img_filename, 'rb') as f:
        tags = exifread.process_file(f)
    focal_length_mm = utils.exif_to_float(tags.get("EXIF FocalLength"))
    globals.f = focal_length_mm * 1e-3
    f_number = utils.exif_to_float(tags.get("EXIF FNumber"))
    globals.D = globals.f / f_number


def _load_make3d_image(img_filename):
    """Load and resize a Make3D RGB image to (460, 345); sets globals.ps.

    The target resolution of (460, 345) follows Saxena et al.  Pixel size
    (globals.ps) is derived from the Canon PowerShot S40 sensor width and
    the ratio of original to resized pixel count.

    Parameters
    ----------
    img_filename : str
        Path to the Make3D JPEG image.

    Returns
    -------
    aif : ndarray, shape (460, 345, 3)
        RGB image [0, 255], dtype float32.
    """
    # one image in the dataset requires EXIF-based rotation correction
    if os.path.basename(img_filename) == "img-op29-p-295t000.jpg":
        im = Image.open(img_filename)
        im = ImageOps.exif_transpose(im)
        aif = np.array(im.convert("RGB"), dtype=np.float32) #/ 255.
    else:
        aif = np.array(Image.open(img_filename), dtype=np.float32) #/ 255.

    original_width = aif.shape[0]

    # resize as recommended by Saxena et al. and Gur (460 x 345)
    aif = skimage.transform.resize(
        aif,
        output_shape=(460, 345),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    )

    # loosely estimate based on 
    # Canon PowerShot S40: 1/1.8" sensor (~7.11 x 5.33 mm)
    sensor_width_m = 7.11e-3
    globals.ps = sensor_width_m / original_width * (original_width / aif.shape[0])
    globals.ps *= 0.01 # scaling factor that performed well in practice

    return aif


def _load_make3d_depth(dpt_filename):
    """Load Make3D depth from a .mat file.

    The .mat file contains 'Position3DGrid' with shape (H, W, 4); index 3
    is the Z (depth) channel in metres.

    Parameters
    ----------
    dpt_filename : str
        Path to the depth .mat file.

    Returns
    -------
    dpt : ndarray, shape (H, W)
        Depth map in metres, dtype float32.
    """
    data = loadmat(dpt_filename)
    dpt = np.array(data["Position3DGrid"], dtype=np.float32)
    return dpt[:, :, 3]


def load_single_sample_Make3D(img_name="img-math7-p-282t0.jpg", split='train', data_dir="data"):
    """Load a single Make3D RGB image and its ground-truth depth map.

    Sets globals.f, globals.D, and globals.ps from EXIF data and image
    dimensions as a side effect.

    Parameters
    ----------
    img_name : str
        JPEG filename of the image (e.g. 'img-math7-p-282t0.jpg').
    data_dir : str
        Root data directory containing the 'Make3D/' subdirectory.
    split : str
        Dataset split: 'train' or 'test'.

    Returns
    -------
    aif : ndarray,
        RGB image [0, 255].
    dpt : ndarray,
        Depth map in metres.
    """
    assert split in ('train', 'test')
    img_subdir = 'Test134Img' if split == 'test' else 'Train400Img'
    dpt_subdir = 'Test134Depth' if split == 'test' else 'Train400Depth'

    img_filename = os.path.join(data_dir, 'Make3D', img_subdir, img_name)

    _load_make3d_camera_params(img_filename)
    aif = _load_make3d_image(img_filename) #* 255.

    part = img_name.split("img-")[1].split(".jpg")[0]
    dpt_filename = os.path.join(data_dir, 'Make3D', dpt_subdir, "depth_sph_corr-" + part + ".mat")
    dpt = _load_make3d_depth(dpt_filename)

    dpt = skimage.transform.resize(
        dpt,
        output_shape=(460, 345),
        order=1,                  # bilinear interpolation
        anti_aliasing=True,
        preserve_range=True
    )

    return aif, dpt



# ---------------------------------------------------------------------------
# MobileDepth
# ---------------------------------------------------------------------------

# Examples without calibration data (not loadable): "bucket", "kitchen"
_MOBILE_DEPTH_VALID_EXAMPLES = [
    "keyboard", "bottles", "fruits", "metals", "plants",
    "telephone", "window", "largemotion", "smallmotion", "zeromotion", "balls",
]

# Calibration folder names differ from example names for some scenes
_MOBILE_DEPTH_CALIB_NAME_MAP = {
    "largemotion": "GTLarge",
    "smallmotion": "GTSmall",
    "zeromotion":  "GT",
    "metals":      "metal",
}


def _find_mobile_depth_example_dir(focal_stack_dir, example_name):
    """Search for a subdirectory named example_name within focal_stack_dir.

    Parameters
    ----------
    focal_stack_dir : str
        Root directory containing per-scene subdirectories.
    example_name : str
        Name of the example scene to find.

    Returns
    -------
    str
        Path to the matching example directory.

    Raises
    ------
    FileNotFoundError
        If no matching directory is found.
    """
    for name in os.listdir(focal_stack_dir):
        subdir = os.path.join(focal_stack_dir, name)
        if os.path.isdir(subdir):
            candidate = os.path.join(subdir, example_name)
            if os.path.isdir(candidate):
                print("Found at:", os.path.abspath(candidate))
                return candidate
    raise FileNotFoundError(f"Example '{example_name}' not found in {focal_stack_dir}")


def _load_mobile_depth_focal_stack(example_dir, resize_frac):
    """Load and stack aligned focal stack JPEG images from a directory.

    Filenames starting with 'a' and ending with '.jpg' are collected and
    sorted alphabetically to establish focal plane order.

    Parameters
    ----------
    example_dir : str
        Path to the directory containing the aligned focal stack images.
    resize_frac : int
        Downsampling factor; 1 for full resolution, 2 for half.

    Returns
    -------
    defocus_stack : ndarray, shape (fs, width, height, 3)
        Focal stack normalised to [0, 1], dtype float32.
    """
    paths = sorted(
        os.path.join(example_dir, fname)
        for fname in os.listdir(example_dir)
        if fname.startswith("a") and fname.endswith(".jpg")
    )
    frames = []
    for path in paths:
        frame = np.array(Image.open(path), dtype=np.float32) #/ 255.
        if resize_frac != 1:
            width, height, _ = frame.shape
            frame = skimage.transform.resize(
                frame,
                output_shape=(width // resize_frac, height // resize_frac),
                order=1, # bilinear interpolation
                anti_aliasing=True,
                preserve_range=True,
            )
        frames.append(frame)
    return np.stack(frames, 0)


def _load_mobile_depth_calibration(example_name, data_path):
    """Read the per-scene calibration file; sets globals.Df, globals.f, globals.D.

    The calibration file lists one ``focal_depth aperture`` pair per line,
    with the focal length as the final line.

    Parameters
    ----------
    example_name : str
        Scene name (remapped to the calibration folder name where needed).
    data_path : str
        Root MobileDepth data directory.

    Returns
    -------
    calib_dir : str
        Path to the resolved calibration directory, used to locate the depth
        and scale result files.
    """
    calib_name = _MOBILE_DEPTH_CALIB_NAME_MAP.get(example_name, example_name)
    calib_dir = os.path.join(data_path, 'photos-calibration-results', 'calibration', calib_name)

    with open(os.path.join(calib_dir, "calibrated.txt"), "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # last line is focal length; preceding lines are focal_depth / aperture pairs
    focal_length = float(lines[-1])
    focal_depths, apertures = [], []
    for line in lines[:-1]:
        parts = line.split()
        if len(parts) >= 2:
            focal_depths.append(float(parts[0]))
            apertures.append(float(parts[1]))

    assert len(set(apertures)) == 1, "Expected consistent aperture across all focal planes"

    globals.Df = np.array(focal_depths, dtype=np.float32)  # unitless
    globals.f = focal_length                                # unitless
    globals.D = set(apertures).pop()                        # unitless, confirmed by Supasorn

    return calib_dir


def load_single_sample_MobileDepth(example_name="keyboard", res="half", data_dir="data"):
    """Load a MobileDepth focal stack with calibration data and depth result.

    Sets globals.Df, globals.f, and globals.D from the per-scene calibration
    file as a side effect.

    Parameters
    ----------
    example_name : str
        Scene name.  Must be one of: keyboard, bottles, fruits, metals,
        plants, telephone, window, largemotion, smallmotion, zeromotion, balls.
    res : str
        Resolution: 'full' or 'half' (default).

    Returns
    -------
    defocus_stack : ndarray, shape (fs, width, height, 3)
        Aligned focal stack normalised to [0, 1].
    dpt_result : ndarray, shape (H, W)
        Pre-computed depth estimate from the MobileDepth calibration,
        dtype float32.
    scale_mat : ndarray
        Scale matrix from the calibration data.
    """
    assert example_name in _MOBILE_DEPTH_VALID_EXAMPLES
    assert res in ('full', 'half')

    data_path = os.path.join(os.getcwd(), data_dir, 'MobileDepth')
    focal_stack_dir = os.path.join(data_path, 'aligned-focus-stack', 'Aligned')

    example_dir = _find_mobile_depth_example_dir(focal_stack_dir, example_name)
    resize_frac = 2 if res == 'half' else 1
    defocus_stack = _load_mobile_depth_focal_stack(example_dir, resize_frac)

    calib_dir = _load_mobile_depth_calibration(example_name, data_path)
    
    order = np.argsort(globals.Df)
    globals.Df = globals.Df[order]
    defocus_stack = defocus_stack[order]

    dpt_result = utils.read_bin_file(os.path.join(calib_dir, "depth_var.bin")).astype(np.float32)
    scale_mat = utils.read_bin_file(os.path.join(calib_dir, "scaleMatrix.bin"))

    # Rotate and resize to half
    defocus_stack = np.stack([
        np.rot90(
            skimage.transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True) if res == 'half' else img
        , k=-1)
        for img in defocus_stack
    ], axis=0)

    return defocus_stack, dpt_result, scale_mat



# ---------------------------------------------------------------------------
# Example image (NYUv2 tutorial sample)
# ---------------------------------------------------------------------------

def load_example_image(fs=5, data_dir='data', res='half'):
    """Load the fixed NYUv2 example (sample 0045) with its ground-truth defocus stack.

    Intended for tutorials and notebooks demonstrating the full pipeline.
    Unlike load_single_sample, this function also returns the pre-rendered
    ground-truth focal stack stored in the NYUv2_single<fs> directory.

    If fs=10, sets globals.Df to the 10-plane focus distance array as a
    side effect.

    Parameters
    ----------
    fs : int
        Focal stack size: 5 or 10.
    data_dir : str
        Root data directory (relative to cwd), expected to contain
        'NYUv2_single<fs>/'.
    res : str
        Resolution: 'half' (default) or 'full'.

    Returns
    -------
    aif : ndarray, shape (width, height, 3)
        All-in-focus image in [0, 255] range.
    dpt : ndarray, shape (width, height)
        Depth map in metres.
    gt_defocus_stack : list of ndarray
        Ground-truth defocus stack; each frame has shape (width, height, 3)
        with pixel values in [0, 255].
    """
    assert fs in (5, 10)
    assert res in ('full', 'half')

    ext = str(fs)
    if fs == 10:
        globals.Df = np.array([0.5, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6], dtype=np.float32)  # m
    if res == 'half':
        ext += '_halfres'

    data_path = os.path.join(os.getcwd(), data_dir, 'NYUv2_single' + str(fs))
    img_name = '0045.png'
    resize_frac = 2 if res == 'half' else 1

    dpt = load_NYUv2_dpt(os.path.join(data_path, 'test_depth', img_name), resize_frac=resize_frac)
    aif = load_NYUv2_aif(os.path.join(data_path, 'test_rgb', img_name), resize_frac=resize_frac)

    files = sorted(os.listdir(os.path.join(data_path, 'test_fs' + ext)))
    gt_defocus_stack = [cv2.cvtColor(cv2.imread(os.path.join(data_path, 'test_fs' + ext, f)), cv2.COLOR_BGR2RGB) for f in files]
    assert len(gt_defocus_stack) == fs

    return aif, dpt, gt_defocus_stack
