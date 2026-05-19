# Configuration

```{include} ../../README.md
:start-after: <!-- configuration-start -->
:end-before: <!-- configuration-end -->
```

## `emri_config.json` reference

| Key | Type | Description |
|---|---|---|
| `m1`, `m2` | float | Primary and secondary masses (solar masses). |
| `a` | float | Dimensionless spin of the primary. |
| `p0`, `e0` | float | Initial semi-latus rectum and eccentricity. |
| `xI0` | float | Initial cosine of the orbital inclination. |
| `dist` | float | Luminosity distance (Gpc). |
| `qS`, `phiS` | float | Sky-location polar angles (radians). |
| `qK`, `phiK` | float | Spin-orientation polar angles (radians). |
| `Phi_phi0`, `Phi_r0`, `Phi_theta0` | float | Initial orbital phases. |
| `T` | float | Total observation time (years). |
| `Nblocks` | int | Number of semi-coherent blocks. Set to `1` for a fully coherent analysis. |
| `dt` | float | Sampling cadence (seconds). |
| `fmin`, `fmax` | float | Frequency-band edges used for likelihood (Hz). |
| `SNR_target` | float | Target network SNR; the injection is rescaled to match. |
| `sigma_prior` | float \| list | Tolerance(s) controlling the per-block Student-t prior width. |
| `data_model` | str | FEW string identifier of the waveform model used for injection. |
| `analysis_model` | str | FEW string identifier of the template used for inference. |
| `add_args` | dict | Extra keyword arguments forwarded to a custom waveform model. |
| `response` | bool | If `true`, apply the LISA `ResponseWrapper`; if `false`, use the LWA sensitivity curve on raw strain. |

## `sample_config.json` reference

| Key | Type | Description |
|---|---|---|
| `static_params` | list[str] | Parameter names to be sampled as *static* (shared across all blocks). |
| `evolving_params` | list[str] | Parameter names to be sampled as *evolving* (per-block). |
| `fixed_evolving` | list[str] | Evolving parameters held fixed at their injected values. |
| `nwalkers` | int | Number of MCMC walkers per temperature. Minimum recommended: `4 × ndim`. |
| `ntemps` | int | Number of parallel-tempering temperatures. |
| `nsteps` | int | Number of sampler iterations. |
| `burn` | int | Number of burn-in iterations before the main run. |
| `jitter` | float | Gaussian scatter applied when initializing walkers around the truth. |
| `custom_bounds` | dict | Optional per-parameter `(low, high)` overrides of the default prior bounds. |
| `move_probs` | dict | Selection probabilities for each custom MCMC move. |
| `new_filename` | str | Output HDF5 backend filename. |
| `old_filename` | str \| null | If non-null, resume from this backend instead of starting fresh. |

> **Tip:** Any model parameter listed in neither `static_params`, `evolving_params`, nor `fixed_evolving` is automatically appended to `fixed_static` and held at its injected value.
