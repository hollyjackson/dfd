# Depth from Defocus

Joint estimation of depth maps and all-in-focus (AIF) images from focal stacks using alternating minimization.

## Overview

Given a stack of images captured at different focus distances, this algorithm jointly recovers:
- A per-pixel **depth map** of the scene
- An **all-in-focus image** free of defocus blur

The method models defocus blur as a depth-dependent Gaussian convolution (circle of confusion) and solves the inverse problem by alternating between two sub-problems:

1. **AIF step** — given the current depth estimate, recover the all-in-focus image via bounded FISTA (Nesterov-accelerated proximal gradient descent)
2. **Depth step** — given the AIF image, find per-pixel depth via coarse grid search followed by golden-section refinement

An MRF-based initialization (graph cuts) provides a spatially smooth starting estimate for the AIF image.

## Project Structure

```
├── run_alternating_minimization.py   # CLI entry point
├── src/
│   ├── alternating_minimization.py   # Main optimization loop
│   ├── forward_model.py              # Sparse blur operators (circle of confusion)
│   ├── initialization.py             # MRF + graph cuts AIF initialization
│   ├── nesterov.py                   # Bounded FISTA solver
│   ├── section_search.py             # Grid search + golden-section refinement
│   ├── dataset_loader.py             # Data loaders for all datasets
│   ├── dataset_params.py             # Camera parameter dataclass
│   ├── config.py                     # Per-dataset hyperparameters
│   ├── utils.py                      # Metrics, I/O, plotting
│   └── outlier_removal.py            # Post-processing outlier detection
├── notebooks/                        # Jupyter notebooks (tutorial, experiments)
├── tests/                            # Pytest test suite
├── data/                             # Dataset storage
├── experiments/                      # Output results
└── pyGCO/                            # Graph cuts library (gco-wrapper)
```

## Setup

Requires Python 3.8+ and a C++ compiler (for `gco-wrapper`).

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_alternating_minimization.py <dataset> <args> [--verbose] [--show_plots] [--save_plots]
```

### Make3D

```bash
python run_alternating_minimization.py make3d <split> <img_name>
# Example:
python run_alternating_minimization.py make3d train img-math7-p-282t0.jpg --verbose
```

### MobileDepth

```bash
python run_alternating_minimization.py mobiledepth <example_name>
# Example:
python run_alternating_minimization.py mobiledepth keyboard --verbose
```

### NYUv2

```bash
python run_alternating_minimization.py nyuv2 <split> <image_number>
# Example:
python run_alternating_minimization.py nyuv2 test 123 --save_plots
```

Results are saved to `experiments/<dataset>/<experiment_name>_<timestamp>/` containing the estimated depth map (`dpt.npy`), AIF image (`aif.png`), and accuracy metrics when ground truth is available.

## Datasets

| Dataset | Type | Source |
|---------|------|--------|
| **NYUv2** | Indoor RGB-D (synthetic defocus from ground truth) | Depth camera |
| **Make3D** | Outdoor scenes (synthetic defocus from LIDAR depth) | LIDAR + EXIF metadata |
| **MobileDepth** | Real focal stacks captured with a smartphone | Phone camera with per-scene calibration |

## Testing

```bash
pytest
```
