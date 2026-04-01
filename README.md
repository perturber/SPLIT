# SPLIT (Sliced Posteriors for Long-Inspiral Trajectories)
SPLIT is a Python package designed for Semi-Coherent EMRI (Extreme-Mass-Ratio Inspiral) Parameter Estimation on Multi-GPUs.

SPLIT exploits the physical hierarchy of the system by decomposing the parameter space into **static** and **evolving** parameters to construct a natural semi-coherent inference framework. The primary motivation for this hierarchical approach is to ensure robustness against non-stationarities in the detector data and potential waveform modeling inaccuracies. In purely coherent inference, these effects accumulate over the long inspiral duration, leading to severe signal dephasing and strong parameter biases. SPLIT mitigates this by slicing the data into `Nblocks` blocks and inferring the joint parameter space semi-coherently with a more forgiving joint prior and likelihood across all blocks.

## Features

* **Multi-GPU Orchestration**: Utilizes CuPy and the `multiprocessing` library to distribute parallel likelihood evaluations across dedicated GPUs.
* **Eryn MCMC Sampler**: Hosts the core inference pipeline, natively utilizing Eryn's "branches" (enabling static vs. evolving parameter decomposition) and "leaves" (handling multiple independent blocks), which significantly simplifies the semi-coherent inference architecture.
* **Custom MCMC Moves**: Implements customized block updating moves like `SequentialAdaptiveBlockedGibbsGaussianMove` and `SequentialBlockedGibbsStretchMove` to efficiently search the "static" and "evolving" parameter branches. 
* **Markovian Student-t Prior**: A heavy-tailed Student-t transition probability between consecutive blocks, penalizing excessive deviations from theoretical vacuum-GR trajectories, but still allowing flexibility to account for genuine waveform inaccuracies.
* **Student-t Block Likelihood**: A block-independent heavy-tailed Student-t likelihood for robustness against non-stationarities in the data such as noise glitches.
* **Automated Diagnostics**: Tracks Gelmman-Rubin convergence and autocorrelation times natively, and automatically produces comprehensive diagnostic plots for the block-level evolving parameters, static parameters, and the joint parameter set projected at t=0.

## Compatibility
Currently implemented only for extreme-mass-ration inspirals (EMRIs) in the LISA band using the `FastEMRIWaveforms(FEW)` package. Also only for devices with GPUs and with CUDA drivers. Not compatible with CPUs; CPU evaluation of N blocks will likely be prohibitively expensive anyway.

## Installation Guide
0. **It is advisable to work in a conda environment:**
   ```bash
   conda create -n split_env jupyter matplotlib tqdm
   conda activate split_env
1. **Clone the repository:**
   ```bash
   git clone https://github.com/perturber/SPLIT.git
   cd SPLIT
2. **Install in editable mode:**
   ```bash
   pip install -e .

## A Note on Dependecies
1. **Eryn**: Please use [this version of `Eryn`](https://github.com/perturber/Eryn/tree/main) with a more robust `key_order` check for loading from backends. See [here](https://github.com/perturber/Eryn/commit/738527991616eb66d046d84f13b2db791e908a72) for the fix.
2. **fastlisaresponse**: Please use a version of `fastlisaresponse` which has the option to specify `t0` ([see definition](https://github.com/mikekatz04/lisa-on-gpu/blob/v1.2.1a0/src/fastlisaresponse/response.py#L700)) and `t_buffer` ([see definition](https://github.com/mikekatz04/lisa-on-gpu/blob/v1.2.1a0/src/fastlisaresponse/response.py#L701)) separately. This package has been validated for the tag [v1.2.1a0](https://github.com/mikekatz04/lisa-on-gpu/tree/v1.2.1a0). It can be pip installed for cuda12x as:
   ```bash
   pip install --pre fastlisaresponse-cuda12x==1.2.1a0
3. **lisaanalysistools**: This package has been validated for the tag [v1.2.8](https://github.com/mikekatz04/LISAanalysistools/tree/v1.2.8). `lisaanalysistools` will try auto-installing a suitable version of cupy. So it is better to install it before any other dependencies. It can be pip installed for cuda12x as:
   ```bash
   pip install lisaanalysistools-cuda12x==1.2.8
4. **FastEMRIWaveforms**: This package has been validated for the tag [v2.0.0](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/v2.0.0). It can be installed from source as:
   ```bash
   git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
   cd FastEMRIWaveforms
   git checkout v2.0.0
   pip install -e '.[dev, testing]'

## Configuration Setup

The SPLIT orchestrator runs by ingesting two distinct JSON configuration files:

1. `emri_config.json`: Controls the physical properties and the signal injection parameters. 
  * Defines fundamental parameters including `m1`, `m2`, `a`, `p0`, `e0`, and starting orientations, phases. 
  * Sets up the trajectory slice settings, dictating the total time `T` in years, number of blocks `Nblocks`, desired network SNR scaling, frequency bounds, and prior tolerance allowances `sigma_prior`. 
  * **Model Specification:** Controls the waveform generator used for data injection (`data_model`) and the template used for inference (`analysis_model`). These natively accept standard FEW string identifiers (e.g., `"FastKerrEccentricEquatorialFlux"`). You can also inject purely custom environmental/glitch waveforms via code while recovering with vacuum GR. The user can also supply additional arguments for the custom waveform model using `"add_args"`. 
  * **Response Wrapper:** Allows toggling the LISA instrument response simulation via the `"response"` boolean flag. If `false`, the pipeline defaults to evaluating raw strain using the Long-Wavelength Approximation (LWA) sensitivity curve.
2. `sample_config.json`: Manage the Eryn MCMC sampler mechanics.
  * Assigns static and evolving parameters that should be inferred through `static_params` and `evolving_params`, respectively.
  * Assigns the set of fixed evolving parameters through `fixed_evolving` and automatically assigns any remaining model parameters to `fixed_static`. 
  * Configures Eryn's `EnsembleSampler` method with `nwalkers`, `ntemps`, `nsteps`, probability thresholds for the custom moves, etc. Allows for dynamically linked resuming output chains via `new_filename`/`old_filename`.

## Usage Quickstart

1. Simply run the `run_job.py` script directly from your terminal inside the conda environment. This script automatically handles the CUDA multiprocessing initialization and executes the main stages of the pipeline sequentially:
   ```bash
   # after installation, you can run SPLIT using the python -m flag
   python -m split
   # or explicitly call the execution file
   python run_job.py

2. You can specify custom configuration files and a custom output directory using command-line flags:
   ```bash
   # using the python -m flag
   python -m split --emri emri_config.json --samp sample_config.json --out SPLIT_Outputs
   # or explicitly
   python run_job.py --emri emri_config.json --samp sample_config.json --out SPLIT_Outputs

3. View all available command-line options and their descriptions using the help flag:
   ```bash
   python -m split -h
   
4. To run a fully-coherent analysis, simply set `'Nblocks': 1` in `emri_config.json`.
