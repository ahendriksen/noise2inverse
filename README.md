# Noise2Inverse

This code accompanies

- Hendriksen, A. A., Pelt, D. M., & Batenburg,
  K. J. (2020). Noise2Inverse: self-supervised deep convolutional
  denoising for linear inverse problems in imaging. CoRR, (), .

## Installation

### On Linux

Create a conda environment with:
``` bash
conda env create -f environment.yaml
conda activate noise2inverse
# Install noise2inverse package in environment
pip install -e .
```

Please be sure to use the exact ASTRA Toolbox version, since newer
versions can result in different pixel intensities.

### On Windows

There is no windows build of the MSD network available for Windows. We
have added an environment file for Windows that does not include the
MSD network.

Create a conda environment with:
``` bash
conda env create -f environment_windows.yaml
conda activate noise2inverse
# Install noise2inverse package in environment
pip install -e .
```

In the training and evaluation notebooks, make sure to select UNet or
DnCNN as the network. The `"msd"` option is not available. The
repository does not contain a trained network weights for UNet and
DnCNN. To evaluate the results, you must first run training.


## How to use

The notebooks describe how to use the package:

- 01_generate_projections.ipynb: Generates clean and noisy projections of the foam phantom;
- 02_reconstruct.ipynb: Contains code for reconstruction;
- 03_train.ipynb: Trains the network;
- 04_evaluate.ipynb: Applies the trained network to the noisy reconstructions to obtain a denoised output.
- 05_metrics.ipynb: Describes metric calculation.

Metrics and results may slightly differ from reported results in
paper: the network was retrained using this cleaned up code.

## Data

Examples of intermediate results are added to the git
repository. Committing all data files and intermediate results is not
possible due to space constraints on github. Nonetheless, using these
files, it should be possible to reproduce the results in the paper.
