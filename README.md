# Core Bifiltration [[arXiv](https://arxiv.org/abs/2405.01214)]

This repository contains an implementation of alpha-core and core Čech persistent homology, together with Jupyter notebooks demonstrating its application to noisy point cloud data.

## Get Started

Install dependencies by running `pip install -r requirements.txt` if needed.

Run the notebook `example_usage.ipynb` for an demonstration of the application of core Čech and alpha-core persistent homology to noisy point clouds. This notebook contains examples for computing persistence along a line and for a fixed $k$.

To reproduce the experiments presented in the paper with $\beta=1$, run the notebook `run_experiments.ipynb`. To reproduce the experiments examining different values of $\beta$, run the notebook `run_experiments_beta.ipynb`.

## Other Files

The code for constructing the filtered nerves is contained in `core.py`. The functions `core_cech` and `core_alpha` returns a simplex tree (a `gudhi.SimplexTree` instance) representing the core Čech and alpha-core filtered nerves, respectively. Different point cloud dataset generators used in the notebooks can be found in `datasets.py`.

