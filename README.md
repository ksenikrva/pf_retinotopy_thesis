# Phase-Field PDEs for the Simulation of Retinotopic Pattern Preservation in Drosophila

This repository contains code developed for the Master's thesis "Phase-Field PDEs for the Simulation of Retinotopic Pattern Preservation in Drosophila" at Freie Universität Berlin, 2026.

## Overview

In this project, [retinotopic pattern preservation in Drosophila](https://www.cell.com/current-biology/abstract/S0960-9822(26)00007-2) is modeled through [phase-field PDEs](https://www.nature.com/articles/srep09172) and subsequently analysed in regards to the effect of deformability.
This repository contains:
- Simulation code
- Analysis code
- Rendered frames
- Videos
### Repository structure
.
├── 1. simulation/
│   ├── ver1/                  # Version 1 simulations
│   ├── ver2/                  # Version 2 simulations
│   ├── ver2b/                 # Version 2b simulations, including additions such as Lcells and extra advection
│   ├── unity end-state/       # End-state  simulations
│   ├── *.sh                   # HPC scripts
│   └── *.csv                  # Sampeled parameters
├── 2. analysis/
│   ├── mean_shape_and_bundles/    # Mean-shape and bundles per run
│   ├── relation_ana/              # Relationship analyses
│   ├── run_metrics/               # Calculated simulation metrics
│   ├── run_metrics_nob/           # Calculated simulation metrics (inner bundles only)
│   ├── *.ipynb                    # Analysis and visualization notebooks
│   └── *.csv                      # Biological reference and derived data
├── 3. frames/               # Rendered simulation frames (all)
├── 4. videos/               # Rendered simulation videos (subset)

## Required software

The simulations require NVIDIA CUDA, including:

- cuFFT
- cuBLAS
- CUDA-capable NVIDIA GPU
- C++ compiler

Analysis code is provided as `.ipynb` notebooks, using:

- Python 3.12.7
- NumPy 2.4.6
- pandas 2.2.3
- Matplotlib 3.10.0
- SciPy 1.17.1
- Pillow 10.3.0
- ImageIO 2.37.3

## Usage

Example compile and call for the simulations:

```bash
nvcc --std=c++17 ver2b.cu -lcuda -lcufft -lcublas -O3 -o ver2b

ver2b cool_run 3 6 5 2 7 0.7
```
Argument structure: `folder` `rows` `bundles (per row)` `lambda` `kappaFMI` `kappaSDK` `D`

For running singular runs / parameter sweeps on [Curta HPC](https://refubium.fu-berlin.de/handle/fub188/26993), example bash scripts `test.sh` and `sim_run.sh` are given