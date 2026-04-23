# Quickstart

1. After installation, you can run SPLIT using the `python -m` flag
   ```bash
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