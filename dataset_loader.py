"""Dataset loaders for depth-from-defocus experiments.

Three datasets are supported:

NYUv2
    Indoor RGB-D images.  Camera parameters are fixed in ``DatasetParams.for_NYUv2()``.

Make3D
    Outdoor images with LIDAR depth.  Camera parameters (focal length,
    aperture, pixel size) are read from EXIF and returned by the loader.

MobileDepth (mobile phone focal stacks)
    Phone-captured focal stacks with per-scene calibration files.
    Camera parameters and focus distances are returned by the loader.
"""
import os

import numpy as np
import skimage
from PIL import Image, ImageOps
import cv2

import exifread
from scipy.io import loadmat

import utils

# Resolve relative data_dir paths against this module's directory (dfd/)
# so that data_dir='data' always means dfd/data/ regardless of caller cwd.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_dir(data_dir):
    """Return *data_dir* as-is if absolute, else resolve relative to dfd/."""
    if os.path.isabs(data_dir):
        return data_dir
    return os.path.join(_MODULE_DIR, data_dir)


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

    data_path = os.path.join(_resolve_data_dir(data_dir), 'NYUv2')
    resize_frac = 2 if res == 'half' else 1

    dpt = load_NYUv2_dpt(os.path.join(data_path, set + '_depth', sample + '.tiff'), resize_frac=resize_frac)
    aif = load_NYUv2_aif(os.path.join(data_path, set + '_rgb', sample + '.png'), resize_frac=resize_frac)

    return aif, dpt




# ---------------------------------------------------------------------------
# Make3D
# ---------------------------------------------------------------------------

def _load_make3d_camera_params(img_filename):
    """Read EXIF metadata from a Make3D image.

    Parameters
    ----------
    img_filename : str
        Path to the Make3D JPEG image.

    Returns
    -------
    f : float
        Focal length in metres.
    D : float
        Aperture diameter in metres.
    """
    with open(img_filename, 'rb') as fh:
        tags = exifread.process_file(fh)
    focal_length_mm = utils.exif_to_float(tags.get("EXIF FocalLength"))
    f = focal_length_mm * 1e-3
    f_number = utils.exif_to_float(tags.get("EXIF FNumber"))
    D = f / f_number
    return f, D


def _load_make3d_image(img_filename):
    """Load and resize a Make3D RGB image to (460, 345).

    The target resolution of (460, 345) follows Saxena et al.  Pixel size
    is derived from the Canon PowerShot S40 sensor width and the ratio of
    original to resized pixel count.

    Parameters
    ----------
    img_filename : str
        Path to the Make3D JPEG image.

    Returns
    -------
    aif : ndarray, shape (460, 345, 3)
        RGB image [0, 255], dtype float32.
    ps : float
        Estimated pixel size in metres.
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
    ps = sensor_width_m / original_width * (original_width / aif.shape[0])
    ps *= 0.01 # scaling factor that performed well in practice

    return aif, ps


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


def load_single_sample_Make3D(img_name, dataset_params, split='train', data_dir="data"):
    """Load a single Make3D RGB image and its ground-truth depth map.

    Camera parameters (f, D, ps) are read from EXIF and image dimensions
    and set on *dataset_params* in place.

    Parameters
    ----------
    img_name : str
        JPEG filename of the image (e.g. 'img-math7-p-282t0.jpg').
    dataset_params : DatasetParams
        Camera/scene parameters; ``f``, ``D``, and ``ps`` are populated
        from EXIF data and image dimensions.
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

    resolved = _resolve_data_dir(data_dir)
    img_filename = os.path.join(resolved, 'Make3D', img_subdir, img_name)

    dataset_params.f, dataset_params.D = _load_make3d_camera_params(img_filename)
    aif, dataset_params.ps = _load_make3d_image(img_filename) #* 255.

    part = img_name.split("img-")[1].split(".jpg")[0]
    dpt_filename = os.path.join(resolved, 'Make3D', dpt_subdir, "depth_sph_corr-" + part + ".mat")
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
    """Read the per-scene calibration file.

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
    Df : ndarray
        Focus distances.
    f : float
        Focal length.
    D : float
        Aperture diameter.
    """
    calib_name = _MOBILE_DEPTH_CALIB_NAME_MAP.get(example_name, example_name)
    calib_dir = os.path.join(data_path, 'photos-calibration-results', 'calibration', calib_name)

    with open(os.path.join(calib_dir, "calibrated.txt"), "r") as fh:
        lines = [line.strip() for line in fh if line.strip()]

    # last line is focal length; preceding lines are focal_depth / aperture pairs
    focal_length = float(lines[-1])
    focal_depths, apertures = [], []
    for line in lines[:-1]:
        parts = line.split()
        if len(parts) >= 2:
            focal_depths.append(float(parts[0]))
            apertures.append(float(parts[1]))

    assert len(set(apertures)) == 1, "Expected consistent aperture across all focal planes"

    Df = np.array(focal_depths, dtype=np.float32)  # unitless
    f = focal_length                                # unitless
    D = set(apertures).pop()                        # unitless, confirmed by Supasorn

    return calib_dir, Df, f, D


def load_single_sample_MobileDepth(example_name, dataset_params, res="half", data_dir="data"):
    """Load a MobileDepth focal stack with calibration data and depth result.

    Camera parameters (Df, f, D) from the per-scene calibration file are
    set on *dataset_params* in place.

    Parameters
    ----------
    example_name : str
        Scene name.  Must be one of: keyboard, bottles, fruits, metals,
        plants, telephone, window, largemotion, smallmotion, zeromotion, balls.
    dataset_params : DatasetParams
        Camera/scene parameters; ``Df``, ``f``, and ``D`` are populated
        from calibration data.
    res : str
        Resolution: 'full' or 'half' (default).
    data_dir : str
        Root data directory.

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

    data_path = os.path.join(_resolve_data_dir(data_dir), 'MobileDepth')
    focal_stack_dir = os.path.join(data_path, 'aligned-focus-stack', 'Aligned')

    example_dir = _find_mobile_depth_example_dir(focal_stack_dir, example_name)
    resize_frac = 2 if res == 'half' else 1
    defocus_stack = _load_mobile_depth_focal_stack(example_dir, resize_frac)

    calib_dir, Df, f, D = _load_mobile_depth_calibration(example_name, data_path)

    order = np.argsort(Df)
    dataset_params.Df = Df[order]
    dataset_params.f = f
    dataset_params.D = D
    defocus_stack = defocus_stack[order]

    dpt_result = utils.read_bin_file(os.path.join(calib_dir, "depth_var.bin")).astype(np.float32)
    dpt_result = np.rot90(
        skimage.transform.resize(dpt_result, (dpt_result.shape[0] // 2, dpt_result.shape[1] // 2), anti_aliasing=True) if res == 'half' else dpt_result
    , k=-1)
    scale_mat = utils.read_bin_file(os.path.join(calib_dir, "scaleMatrix.bin"))

    # Rotate and resize to half
    defocus_stack = np.stack([
        np.rot90(img, k=-1)
        for img in defocus_stack
    ], axis=0)

    return defocus_stack, dpt_result, scale_mat



# ---------------------------------------------------------------------------
# Example image (NYUv2 tutorial sample)
# ---------------------------------------------------------------------------

def load_example_image(data_dir='data', res='half'):
    """Load the fixed NYUv2 example.

    Intended for tutorials and notebooks demonstrating the full pipeline.

    Parameters
    ----------
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
        Depth map in meters.
    """
    assert res in ('full', 'half')

    data_path = os.path.join(_resolve_data_dir(data_dir), 'example')
    img_name = '0045.png'
    resize_frac = 2 if res == 'half' else 1

    dpt = load_NYUv2_dpt(os.path.join(data_path, 'depth', img_name), resize_frac=resize_frac)
    aif = load_NYUv2_aif(os.path.join(data_path, 'rgb', img_name), resize_frac=resize_frac)

    return aif, dpt
