# Changelog

All notable changes to SPLIT will be documented on this page.

## 0.0.1a — Initial alpha release

- Multi-GPU semi-coherent EMRI parameter estimation pipeline.
- Custom Eryn moves: `SequentialAdaptiveBlockedGibbsGaussianMove`, `SequentialBlockedGibbsStretchMove`.
- Markovian Student-t prior across blocks.
- Student-t block likelihood for non-stationary data.
- Automated Gelman-Rubin and autocorrelation-time diagnostics with corner-plot output.
- JSON-driven configuration via `emri_config.json` and `sample_config.json`.
