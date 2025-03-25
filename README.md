<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/MaDDG.svg?branch=main)](https://cirrus-ci.com/github/<USER>/MaDDG)
[![ReadTheDocs](https://readthedocs.org/projects/MaDDG/badge/?version=latest)](https://MaDDG.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/MaDDG/main.svg)](https://coveralls.io/r/<USER>/MaDDG)
[![PyPI-Server](https://img.shields.io/pypi/v/MaDDG.svg)](https://pypi.org/project/MaDDG/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/MaDDG.svg)](https://anaconda.org/conda-forge/MaDDG)
[![Monthly Downloads](https://pepy.tech/badge/MaDDG/month)](https://pepy.tech/project/MaDDG)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/MaDDG)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# MaDDG (Maneuver Detection Data Generation)

<p align="center">A library for simulating high-fidelity observations of satellite trajectories with configurable maneuvers and custom sensor networks.</p>

MaDDG provides a simple interface for modeling complex observation scenarios. It allows you to create a satellite in any geocentric orbit, propagate its motion with a robust physical model, and track its position through optical sensors with customizable locations, observing limits, and noise parameters.

Through its use of [hydra-zen](https://github.com/mit-ll-responsible-ai/hydra-zen) and the [submitit plugin](https://hydra.cc/docs/plugins/submitit_launcher/), MaDDG can easily configure an array of simulation scenarios and distribute them in a SLURM cluster, empowering users to create large-scale, realistic datasets for training reliable maneuver detection and characterization models.

## Installation

MaDDG is available on PyPI:

```console
pip install MaDDG
```

To install from source, clone this repository and run the following
command from its top-level directory:

```console
pip install -e .
```

If you want to modify the orbit propagation physics behind MaDDG, you
will likely need to edit the [AstroForge](https://github.com/mit-ll/AstroForge) library, as well. AstroForge is
an open-source astrodynamics library and a key requirement of MaDDG. See
the [AstroForge documentation](https://astroforge.readthedocs.io/en/latest/) for installation instructions.

## Usage

For details on how to use the various features of MaDDG, we recommend following the Jupyter notebooks in the `examples/` directory.

## Citation

Please use this DOI number reference, published on [Zenodo](https://zenodo.org), when citing the software:

[![DOI](https://zenodo.org/badge/921266858.svg)](https://doi.org/10.5281/zenodo.15080638)

## Post Processing Scripts

### Weather Event (Cloud Dropout)

Use `dropout.py` to apply pseudo weather-based data dropout to dataset .csv files created with `hz_launcher.py`.

```
$ python scripts/dropout.py --help
usage: dropout.py [-h] --path PATH [--cloud_prob CLOUD_PROB] [--cloud_duration_mean CLOUD_DURATION_MEAN] [--cloud_duration_std CLOUD_DURATION_STD] [--num_runs NUM_RUNS]
                  [--save_copy_of_original] [--save_plots] [--submitit]

Script description

options:
  -h, --help            show this help message and exit
  --path PATH           The path to the input data file (.csv) (default: None)
  --cloud_prob CLOUD_PROB
                        Probability of a cloud event blocking the sky during 
                        any nighttime observable window for each sensor (default: 0.5)
  --cloud_duration_mean CLOUD_DURATION_MEAN
                        Mean duration of a cloud event (seconds) (default: 10800.0)
  --cloud_duration_std CLOUD_DURATION_STD
                        Standard deviation of a cloud event (seconds) (default: 3600.0)
  --num_runs NUM_RUNS   Number of dropout datasets to generate (default: 1)
  --save_copy_of_original
                        Raise this flag to save copy of the original input data along 
                        side of the modified dataset with dropouts (default: False)
  --save_plots          Raise this flag to generate and save plots (default: False)
  --submitit            Raise this flag to use submitit to launch jobs across multiple 
                        nodes in parallel (default: False)
```

Example vscode launch.json entry:

```json
{
    "name": "Launch Dropout",
    "type": "python",
    "request": "launch",
    "program": "scripts/dropout.py",
    "console": "integratedTerminal",
    "cwd": "/path/to/MaDDG",
    "args": [
        "--path=/path/to/complete.csv",
        "--cloud_prob=1.0",
        "--cloud_duration_mean=0,3600,7200,10800,14400,18000,21600,25200,28800,32400,36000,39600,43200,46800,50400,54000,57600",
        "--cloud_duration_std=0.0",
        "--num_runs=10",
        "--save_copy_of_original",
        "--submitit",
    ]
}
```

## Disclaimer

DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

Research was sponsored by the United States Air Force Research Laboratory and the United
States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative
Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document
are those of the authors and should not be interpreted as representing the official
policies, eitherexpressed or implied, of the United States Air Force or the U.S.
Government. The U.S.Government is authorized to reproduce and distribute reprints
for Government purposes notwithstanding any copyright notation herein.

© 2024 Massachusetts Institute of Technology.
