<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logos/github-readme-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logos/github-readme-light.svg">
    <img alt="SPLIT: Sliced Posteriors for Long Inspiral Trajectories" src="docs/assets/logos/github-readme-light.svg" width="98%">
  </picture>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-%E2%89%A53.9-blue"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://github.com/perturber/SPLIT"><img alt="Status" src="https://img.shields.io/badge/status-alpha-orange"></a>
  <a href="https://doi.org/10.5281/zenodo.20290209"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.20290209.svg"></a>
</p>

<!-- docs-start -->

# SPLIT (Sliced Posteriors for Long-Inspiral Trajectories)

SPLIT is a Python package designed for Semi-Coherent EMRI (Extreme-Mass-Ratio Inspiral) Parameter Estimation on Multi-GPUs.

SPLIT exploits the physical hierarchy of the system by decomposing the parameter space into **static** and **evolving** parameters to construct a natural semi-coherent inference framework. The primary motivation for this hierarchical approach is to ensure robustness against non-stationarities in the detector data and potential waveform modeling inaccuracies. In purely coherent inference, these effects accumulate over the long inspiral duration, leading to severe signal dephasing and strong parameter biases. SPLIT mitigates this by slicing the data into `Nblocks` blocks and inferring the joint parameter space semi-coherently with a more forgiving joint prior and likelihood across all blocks.

<!-- features-start -->
## Features

* **Multi-GPU Orchestration**: Utilizes CuPy and the `multiprocessing` library to distribute parallel likelihood evaluations across dedicated GPUs.
* **Eryn MCMC Sampler**: Hosts the core inference pipeline, natively utilizing Eryn's "branches" (enabling static vs. evolving parameter decomposition) and "leaves" (handling multiple independent blocks), which significantly simplifies the semi-coherent inference architecture.
* **Custom MCMC Moves**: Implements customized block updating moves like `SequentialAdaptiveBlockedGibbsGaussianMove` and `SequentialBlockedGibbsStretchMove` to efficiently search the "static" and "evolving" parameter branches.
* **Markovian Student-t Prior**: A heavy-tailed Student-t transition probability between consecutive blocks, penalizing excessive deviations from theoretical vacuum-GR trajectories, but still allowing flexibility to account for genuine waveform inaccuracies.
* **Student-t Block Likelihood**: A block-independent heavy-tailed Student-t likelihood for robustness against non-stationarities in the data such as noise glitches.
* **Automated Diagnostics**: Tracks Gelman-Rubin convergence and autocorrelation times natively, and automatically produces comprehensive diagnostic plots for the block-level evolving parameters, static parameters, and the joint parameter set projected at t=0.
<!-- features-end -->

<!-- compatibility-start -->
## Compatibility

> **Note:** SPLIT is currently GPU-only. It is implemented for extreme-mass-ratio inspirals (EMRIs) in the LISA band using the [`FastEMRIWaveforms` (FEW)](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) package and requires NVIDIA GPUs with CUDA 12.x drivers. CPU-only evaluation of `N` blocks is not supported and would be prohibitively expensive in any case.
<!-- compatibility-end -->

<!-- docs-end -->

<!-- installation-start -->
## Installation Guide

It is advisable to work in a fresh `conda` environment:

```bash
conda create -n split_env python=3.10
conda activate split_env
```

Then clone the repository and install in editable mode:

```bash
git clone https://github.com/perturber/SPLIT.git
cd SPLIT
pip install -e .
```

> **Warning:** `pip install -e .[gpu]` will pull `eryn` and `fastemriwaveforms-cuda12x` from PyPI at unconstrained versions, which is **not** what SPLIT expects. Install the pinned dependencies below manually before (or instead of) using the `[gpu]` extras.
<!-- installation-end -->

<!-- dependencies-start -->
## A Note on Dependencies

1. **Eryn**: Please use [this version of `Eryn`](https://github.com/perturber/Eryn/tree/main) with a more robust `key_order` check for loading from backends. See [here](https://github.com/perturber/Eryn/commit/738527991616eb66d046d84f13b2db791e908a72) for the fix. You can install this version of `Eryn` from source using the following commands:
   ```bash
   git clone https://github.com/perturber/Eryn.git
   cd Eryn
   pip install -e .
   ```
2. **fastlisaresponse**: Please use a version of `fastlisaresponse` which has the option to specify `t0` ([see definition](https://github.com/mikekatz04/lisa-on-gpu/blob/v1.2.1a0/src/fastlisaresponse/response.py#L700)) and `t_buffer` ([see definition](https://github.com/mikekatz04/lisa-on-gpu/blob/v1.2.1a0/src/fastlisaresponse/response.py#L701)) separately. This package has been validated for the tag [v1.2.1a0](https://github.com/mikekatz04/lisa-on-gpu/tree/v1.2.1a0). It can be pip installed for cuda12x as:
   ```bash
   pip install --pre fastlisaresponse-cuda12x==1.2.1a0
   ```
3. **lisaanalysistools**: This package has been validated for the tag [v1.2.8](https://github.com/mikekatz04/LISAanalysistools/tree/v1.2.8). `lisaanalysistools` will try auto-installing a suitable version of cupy. So it is better to install it before any other dependencies. It can be pip installed for cuda12x as:
   ```bash
   pip install lisaanalysistools-cuda12x==1.2.8
   ```
4. **FastEMRIWaveforms**: This package has been validated for the tag [v2.0.0](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/v2.0.0). It can be installed from source as:
   ```bash
   git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
   cd FastEMRIWaveforms
   git checkout v2.0.0
   pip install -e '.[dev, testing]'
   ```
<!-- dependencies-end -->

<!-- configuration-start -->
## Configuration Setup

The SPLIT orchestrator runs by ingesting two distinct JSON configuration files:

1. `emri_config.json`: Controls the physical properties and the signal injection parameters.
   * Defines fundamental parameters including `m1`, `m2`, `a`, `p0`, `e0`, and starting orientations, phases.
   * Sets up the trajectory slice settings, dictating the total time `T` in years, number of blocks `Nblocks`, desired network SNR scaling, frequency bounds, and prior tolerance allowances `sigma_prior`.
   * **Model Specification:** Controls the waveform generator used for data injection (`data_model`) and the template used for inference (`analysis_model`). These natively accept standard FEW string identifiers (e.g., `"FastKerrEccentricEquatorialFlux"`). You can also inject purely custom environmental/glitch waveforms via code while recovering with vacuum GR. The user can also supply additional arguments for the custom waveform model using `"add_args"`.
   * **Response Wrapper:** Allows toggling the LISA instrument response simulation via the `"response"` boolean flag. If `false`, the pipeline defaults to evaluating raw strain using the Long-Wavelength Approximation (LWA) sensitivity curve.
2. `sample_config.json`: Manages the Eryn MCMC sampler mechanics.
   * Assigns static and evolving parameters that should be inferred through `static_params` and `evolving_params`, respectively.
   * Assigns the set of fixed evolving parameters through `fixed_evolving` and automatically assigns any remaining model parameters to `fixed_static`.
   * Configures Eryn's `EnsembleSampler` method with `nwalkers`, `ntemps`, `nsteps`, probability thresholds for the custom moves, etc. Allows for dynamically linked resuming output chains via `new_filename`/`old_filename`.

See [`emri_config.json`](https://github.com/perturber/SPLIT/blob/main/emri_config.json) and [`sample_config.json`](https://github.com/perturber/SPLIT/blob/main/sample_config.json) for working examples.
<!-- configuration-end -->

<!-- quickstart-start -->
## Usage Quickstart

1. After installation, you can run SPLIT using the `python -m` flag:
   ```bash
   python -m split
   # or explicitly call the execution file
   python run_job.py
   ```

2. You can specify custom configuration files and a custom output directory using command-line flags:
   ```bash
   # using the python -m flag
   python -m split --emri emri_config.json --samp sample_config.json --out SPLIT_Outputs
   # or explicitly
   python run_job.py --emri emri_config.json --samp sample_config.json --out SPLIT_Outputs
   ```

3. View all available command-line options and their descriptions using the help flag:
   ```bash
   python -m split -h
   ```

4. To run a fully-coherent analysis, simply set `'Nblocks': 1` in `emri_config.json`.
<!-- quickstart-end -->

<!-- troubleshooting-start -->
## Troubleshooting

### GPU / CUDA Issues

**`RuntimeError: CUDA driver error: initialization error`**
Ensure your CUDA drivers are compatible with CUDA 12.x. Verify with:
```bash
nvidia-smi
nvcc --version
```

**`cupy.cuda.runtime.CUDARuntimeError: cudaErrorNoDevice`**
No GPU was detected. SPLIT requires at least one NVIDIA GPU. Confirm with `nvidia-smi`.

**`OutOfMemoryError` during `run_sampler`**
Each worker process allocates GPU memory for waveform generation and data arrays. Try reducing the number of blocks (`Nblocks` in `emri_config.json`) or the number of walkers (`nwalkers` in `sample_config.json`).

### Dependency Issues

**`ImportError: No module named 'eryn'`**
Install the patched Eryn fork required by SPLIT:
```bash
pip install git+https://github.com/perturber/Eryn.git@main
```

**`AttributeError: 'HDFBackend' object has no attribute 'key_order'`**
You have the upstream `eryn` installed instead of the patched fork. See [A Note on Dependencies](https://github.com/perturber/SPLIT#a-note-on-dependencies).

**`fastlisaresponse` version mismatch**
SPLIT requires `fastlisaresponse >= 1.2.1a0` for separate `t0`/`t_buffer` arguments:
```bash
pip install --pre fastlisaresponse-cuda12x==1.2.1a0
```

### Waveform / Trajectory Issues

**Sampler produces `-inf` log-likelihood for all walkers**
- Check that `fmin`/`fmax` in `emri_config.json` are within the signal's frequency range.
- Verify `Nblocks` is not so large that individual blocks are shorter than one orbital cycle.
- Ensure `p0`, `e0`, and `a` satisfy the separatrix condition for the chosen spin value.

**Custom waveform model not found**
Pass it directly via `custom_injection_func` / `custom_analysis_func` arguments to `SPLIT`, or register it in the `named_models` property in `split.py`.

### MCMC / Convergence Issues

**Walkers initialized outside prior bounds**
Reduce `jitter` in `sample_config.json` or tighten the prior bounds via the `custom_bounds` key.

**Sampler never converges**
- Increase `nsteps` or `nwalkers` (minimum recommended: `4 × ndim`).
- Tune `sigma_prior` tolerances in `emri_config.json` — values that are too tight trap walkers near the true solution.
- Use a dedicated `burn` phase first to let the adaptive moves (`BlockAdaptGaussian`) tune their covariance before the main run.
<!-- troubleshooting-end -->

## Citation

If you use SPLIT in your research, please cite it via its Zenodo DOI:
[10.5281/zenodo.20290209](https://doi.org/10.5281/zenodo.20290209).

A ready-to-use BibTeX entry:

```bibtex
@software{kejriwal_2026_20290209,
  author       = {Kejriwal, Shubham},
  title        = {SPLIT - Sliced Posteriors for Long-Inspiral
                   Trajectories
                  },
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.0.1alpha},
  doi          = {10.5281/zenodo.20290209},
  url          = {https://doi.org/10.5281/zenodo.20290209},
}
```

The same metadata is also provided in machine-readable form in
[`CITATION.cff`](CITATION.cff); GitHub renders this as a
**"Cite this repository"** button in the repo sidebar, with BibTeX
and APA exports available with one click.

## License

SPLIT is distributed under the terms of the [MIT License](LICENSE).
