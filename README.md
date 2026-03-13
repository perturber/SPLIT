# SPLIT
Sliced Posteriors for Long-Inspiral Trajectories.

## Compatibility
Currently available only for devices with GPUs and with CUDA drivers. Not compatible with CPUs; CPU evaluation of N blocks will likely be prohibitively expensive anyway.

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
