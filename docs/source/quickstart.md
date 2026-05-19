# Quickstart

```{include} ../../README.md
:start-after: <!-- quickstart-start -->
:end-before: <!-- quickstart-end -->
```

## Outputs

By default, a SPLIT run writes the following into the directory specified by `--out` (default: `SPLIT_Outputs/`):

- An Eryn HDF5 backend file containing the full chain, log-probabilities, and acceptance statistics.
- Diagnostic plots for the block-level evolving parameters, static parameters, and the joint parameter set projected at `t=0`.
- A copy of the resolved `emri_config.json` and `sample_config.json` used for the run, for reproducibility.

Resume a previous run by setting `old_filename` (and a new `new_filename`) in `sample_config.json`.
