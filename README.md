# Phase-Field PDEs for the Simulation of Retinotopic Pattern Preservation in Drosophila

This repository contains code developed for the Master's thesis "Phase-Field PDEs for the Simulation of Retinotopic Pattern Preservation in Drosophila" at Freie Universität Berlin, 2026.

## Overview

In this project, [retinotopic pattern preservation in Drosophila](https://www.cell.com/current-biology/abstract/S0960-9822(26)00007-2) is modeled through [phase-field PDEs](https://www.nature.com/articles/srep09172) and subsequently analysed in regards to the effect of deformability.
This repository contains:
- Simulation code
- Analysis code
- Rendered frames
- Videos
## Required software

The simulations require:

- NVIDIA CUDA, including:

	- cuFFT
	- cuBLAS
	- CUDA-capable NVIDIA GPU
	- C++ compiler
Example compile and call:
```bash
nvcc --std=c++17 ver2b.cu -lcuda -lcufft -lcublas -O3 -o ver2b

ver2b cool_run 3 6 5 2 7 0.7
```
Argument structure: *folder* *rows* *bundles (per row)* *lambda* *kappaFMI* *kappaSDK* *D*

## Usage

```bash
stuff
```
