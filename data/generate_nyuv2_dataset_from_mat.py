"""
Extract RGB images and depth maps from the NYU Depth V2 .mat file and save them
as PNG and TIFF files, respectively, split into train/test directories.

Expected input files (download from the NYU Depth V2 project page):
  - nyu_depth_v2_labeled.mat  (HDF5 format)
  - splits.mat                (official 795/654 train/test split)

Output layout:
  NYUv2/
    train_rgb/<index>.png
    train_depth/<index>.tiff
    test_rgb/<index>.png
    test_depth/<index>.tiff
"""

import os

import h5py
import numpy as np
import tifffile as tiff
from PIL import Image
from scipy.io import loadmat


def find_datasets(f, name_substring):
    """Return full HDF5 paths of datasets whose *basename* contains substring."""
    matches = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            base = name.split("/")[-1]
            if name_substring in base:
                matches.append(name)

    f.visititems(visitor)
    return matches


def load_datasets(path, substrings=("depths", "images")):
    """Load matching HDF5 datasets into memory as numpy arrays, keyed by path."""
    out = {}
    with h5py.File(path, "r") as f:
        for sub in substrings:
            paths = find_datasets(f, sub)
            out[sub] = {p: f[p][()] for p in paths}
    return out


def load_split(path):
    """Load the official NYU Depth V2 train/test split and return 0-based indices."""
    m = loadmat(path)
    train = np.array(m["trainNdxs"]).squeeze().astype(np.int64)  # (795,)
    test  = np.array(m["testNdxs"]).squeeze().astype(np.int64)   # (654,)

    # Convert to 0-based indices if the .mat file uses 1-based (MATLAB convention)
    is_one_based = (train.min() >= 1) and (test.min() >= 1)
    train0 = train - 1 if is_one_based else train
    test0  = test  - 1 if is_one_based else test

    return train0, test0


if __name__ == "__main__":
    # Load the full labeled dataset (1449 RGBD samples)
    nyuv2_labeled_datafile = "nyu_depth_v2_labeled.mat"
    data = load_datasets(nyuv2_labeled_datafile)

    depths = data["depths"].get("depths")  # shape (1449, 640, 480)
    images = data["images"].get("images")  # shape (1449, 3, 640, 480)

    print(depths.shape, images.shape)

    # Load the official 795/654 train/test split
    split_file = "splits.mat"
    train_split, test_split = load_split(split_file)

    # Write images and depth maps into split subdirectories
    data_dir = "NYUv2"
    os.mkdir(data_dir)
    for label, split in [("train", train_split), ("test", test_split)]:
        print("Processing", label, "set...")
        os.mkdir(os.path.join(data_dir, label + "_depth"))
        os.mkdir(os.path.join(data_dir, label + "_rgb"))

        for n in split:
            fn = f"{n:04d}"
            rgb_fn = os.path.join(data_dir, label + "_rgb", fn)    # no extension yet
            dpt_fn = os.path.join(data_dir, label + "_depth", fn)

            dpt = depths[n]
            rgb = images[n]

            # The HDF5 arrays are stored transposed
            dpt = np.transpose(dpt, (1, 0))
            dpt *= 1e4  # scale to preserve sub-millimeter precision in float32
            tiff.imwrite(dpt_fn + ".tiff", dpt.astype(np.float32))

            # Reorder channel-first to channel-last 
            rgb = np.asarray(rgb)
            if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
                rgb = np.transpose(rgb, (2, 1, 0))
            Image.fromarray(rgb).save(rgb_fn + ".png")

        print("...done")