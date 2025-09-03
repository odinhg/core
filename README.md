# Core Bifiltration [[arXiv](https://arxiv.org/abs/2405.01214)]

This repository contains an implementation of Delaunay core (also known as alpha-core) and core Čech persistent homology, together with Jupyter notebooks demonstrating its application to noisy point cloud data.

**Update February 2025:** The Delaunay core bifiltration is now part of the [`multipers`](https://github.com/DavidLapous/multipers) library. The implementation in `multipers` computes the entire bifiltration in contrast to this implementation which only computes persistence along a line (slice).

## Get Started

Install dependencies by running `pip install -r requirements.txt` if needed.

Run the notebook `example_usage.ipynb` for an demonstration of the application of core Čech and alpha-core persistent homology to noisy point clouds. This notebook contains examples for computing persistence along a line and for a fixed $k$.

To reproduce the experiments presented in the paper with $\beta=1$, run the notebook `run_experiments.ipynb`. To reproduce the experiments examining different values of $\beta$, run the notebook `run_experiments_beta.ipynb`.

## Other Files

The code for constructing the filtered nerves is contained in `core.py`. The functions `core_cech` and `core_alpha` returns a simplex tree (a `gudhi.SimplexTree` instance) representing the core Čech and alpha-core filtered nerves, respectively. Different point cloud dataset generators used in the notebooks can be found in `datasets.py`.

## Multipersistence Plots

To reproduce the multipersistence module approximation and Hilbert function plots from the paper, run the notebook `multipersistence_plots.ipynb`. This notebook requires [`multipers`](https://github.com/DavidLapous/multipers), [`function_delaunay`](https://bitbucket.org/mkerber/function_delaunay) and [`mpfree`](https://bitbucket.org/mkerber/mpfree).

**Installing external dependencies using Anaconda:**

1. Create a new conda environment with Python 3.13 with `conda create -n multipers_env python=3.13`.
2. Activate the environment with `conda activate multipers_env` and install `multipers` in the environment with `conda install -c conda-forge multipers`.
3. In the environment, run `./install_external_libraries.sh` to compile and install `mpfree` and `function_delaunay` into your environment.

You might want to install the Python packages `shapely` and `pykeops` too.

## Comparing to the multicover bifiltration

For running size benchmarks and comparing Hilbert functions of the rhomboid tiling bifiltration (equivalent to the multicover bifiltration) and the Delaunay core bifiltration, see the repository [odinhg/multipersistence_nix](https://github.com/odinhg/multipersistence_nix).
