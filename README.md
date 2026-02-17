# Depth from Defocus

Joint estimation of depth maps and all-in-focus (AIF) images from focal stacks using alternating minimization.

## Overview

Given a stack of images captured at different focus distances, this algorithm jointly recovers:
- A per-pixel **depth map** of the scene
- An **all-in-focus image** free of defocus blur

The method models defocus blur as a depth-dependent Gaussian convolution and directly optimizes the inverse problem by alternating minimization, minimizing the MSE between the predicted and input focal stacks. It requires camera parameters (focal length, aperture diameter, focus distances) and a specified depth range.

**Initialization.** The AIF image is initialized using the MRF-based stitching algorithm of Suwajanakorn et al., which minimizes a two-term energy via graph cuts: a sharpness term (Sobel gradient magnitude over Gaussian-weighted patches) and a smoothness term penalizing jumps between focal stack layers.

**Alternating minimization.** The optimization alternates between two steps for a fixed number of epochs:

1. **Depth step** — with the AIF image fixed, perform a grid search over candidate depths using a precomputed blur stack, optionally with windowed MSE for local smoothness. Refine with golden-section search around the best grid point. A pixel-wise comparison with the previous depth map guarantees monotonically decreasing loss.
2. **AIF step** — with the depth map fixed, solve for the AIF image using FISTA (Beck and Teboulle), with step size set via the Lipschitz constant (approximated by power iterations) and pixel values clipped to a valid range. The number of FISTA iterations increases progressively across alternating minimization epochs.

**Post-processing (optional).** An outlier removal step can be applied to the final depth map to reduce artifacts in low-texture regions. Artifact regions are detected by thresholding the total variation of the depth map, and affected pixels are replaced with the mean of valid neighbors.

## Project Structure

```
├── run_alternating_minimization.py   # CLI entry point
├── src/
│   ├── alternating_minimization.py   # Main alternating minimization optimization loop
│   ├── forward_model.py              # Forward model for defocus blur
│   ├── initialization.py             # MRF + graph cuts AIF initialization (based on Suwajanakorn et al. AIF stitching)
│   ├── nesterov.py                   # Bounded FISTA solver (for AIF image, given dpeth map)
│   ├── section_search.py             # Grid search + golden-section refinement (for depth map, given AIF image)
│   ├── dataset_loader.py             # Data loaders for all datasets (NYUv2, Make3D, mobile phone focal stacks)
│   ├── dataset_params.py             # Camera and dataset parameter dataclass
│   ├── config.py                     # Per-dataset hyperparameters
│   ├── utils.py                      # Metrics, I/O, plotting
│   └── outlier_removal.py            # Post-processing outlier detection
├── notebooks/                        # Jupyter notebooks (tutorial, experiments)
├── tests/                            # Pytest test suite
├── data/                             # Dataset storage
├── experiments/                      # Output results
└── pyGCO/                            # Graph cuts library (gco-wrapper) for AIF initialization
```

## Setup

Requires Python 3.8+ and a C++ compiler (for `gco-wrapper`).

```bash
pip install -r requirements.txt
```

We recommend using `uv` for package management.

## Usage

```bash
python run_alternating_minimization.py <dataset> <args> [--verbose] [--show_plots] [--save_plots]
```

### NYUv2

```bash
python run_alternating_minimization.py nyuv2 <split> <image_number>
# Example:
python run_alternating_minimization.py nyuv2 test 0045 --save_plots
```

### Make3D

```bash
python run_alternating_minimization.py make3d <split> <img_name>
# Example:
python run_alternating_minimization.py make3d train img-math7-p-282t0.jpg --verbose
```

### Mobile phone focal stacks

```bash
python run_alternating_minimization.py mobiledepth <example_name>
# Example:
python run_alternating_minimization.py mobiledepth keyboard --verbose
```

Results are saved to `experiments/<dataset>/<experiment_name>_<timestamp>/` containing the estimated depth map (`dpt.npy`), AIF image (`aif.png`), and accuracy metrics when ground truth is available.  If `--save_plots` is enabled, diagnostic plots are saved for each iteration of the optimization.

## Datasets

### NYUv2 (Silberman et al. 2012)

Download the following files to `data/` and then run `generate_nyuv2_dataset_from_mat.py` in the dataset folder:

1. [nyu_depth_v2_labeled.mat](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) — labeled NYU Depth v2 dataset
2. [splits.mat](http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat) — official train/test split

### Make3D (Saxena et al. 2005, 2009)

Download Dataset 1 (images and depths for training and test) from the [Make3D project page](http://make3d.cs.cornell.edu/data.html).

### MobileDepth (Suwajanakorn et al. 2015)

Download the mobile phone focal stacks from the [DFF download page](https://www.supasorn.com/dffdownload.html).

## Testing

```bash
pytest -v
```
