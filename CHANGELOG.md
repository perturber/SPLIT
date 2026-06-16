# Changelog

All notable changes to SPLIT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres (best-effort) to [PEP 440](https://peps.python.org/pep-0440/)
versioning.

## [v0.0.1] - 2026-05-19

First public alpha of SPLIT — Sliced Posteriors for Long-Inspiral Trajectories.

### Added
- Multi-GPU semi-coherent EMRI parameter-estimation pipeline (`split.split.SPLIT`).
- Custom Eryn moves: `SequentialAdaptiveBlockedGibbsGaussianMove` and
  `SequentialBlockedGibbsStretchMove` (`split.moves`).
- Markovian Student-t transition prior across blocks (`split.priors`).
- Block-independent Student-t likelihood for robustness against
  non-stationarities in the data.
- Automated Gelman-Rubin convergence and autocorrelation-time diagnostics
  with corner-plot output (`split.diagnostics`).
- JSON-driven configuration via `emri_config.json` and `sample_config.json`,
  including dynamic resume via `new_filename`/`old_filename`.
- Two reference custom waveform models in `split.customEMRIs`:
  `AccEccEqPn5AAK` and `FastKerrEccentricEquatorialAccretionFlux`.
- Sphinx + Read the Docs documentation with installation, configuration,
  quickstart, troubleshooting, contributing, and API-reference pages.

### Known limitations
- GPU-only; requires NVIDIA hardware with CUDA 12.x drivers.
- Depends on a patched fork of `Eryn` and pinned versions of
  `fastlisaresponse-cuda12x`, `lisaanalysistools-cuda12x`, and
  `FastEMRIWaveforms` — see the README for installation details.

[v0.0.1alpha]: https://github.com/perturber/SPLIT/releases/tag/v0.0.1alpha
