# Contributing

Contributions are welcome! The notes below describe the conventions used in this repository.

## Development install

```bash
git clone https://github.com/perturber/SPLIT.git
cd SPLIT
pip install -e .
```

Then install the pinned GPU dependencies described in [Installation](installation.md).

## Building the docs locally

```bash
pip install -e ".[docs]"
cd docs
make html
```

The rendered HTML is written to `docs/_build/html/index.html`.

## Docstring style

SPLIT uses **NumPy-style** docstrings, parsed by Sphinx via the `napoleon` extension. A typical function docstring looks like:

```python
def compute_snr(waveform, psd, dt):
    """Compute the matched-filter SNR of a waveform.

    Parameters
    ----------
    waveform : array_like
        Complex frequency-domain waveform.
    psd : array_like
        One-sided power spectral density evaluated on the same grid.
    dt : float
        Sampling cadence in seconds.

    Returns
    -------
    float
        The matched-filter SNR.
    """
```

Please add or update docstrings for any public function, class, or method you touch.

## Pull request checklist

- [ ] Docstrings updated for changed public APIs.
- [ ] User-facing changes mentioned in [`changelog.md`](changelog.md).
- [ ] `make html` builds without new warnings.
- [ ] The example run in [Quickstart](quickstart.md) still completes end-to-end.
