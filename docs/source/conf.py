import os
import sys

# Point to your src directory so autodoc can find the 'split' package
sys.path.insert(0, os.path.abspath('../../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'SPLIT'
copyright = '2026, Shubham Kejriwal'
author = 'Shubham Kejriwal'

# Pull the version dynamically from installed package metadata so the docs
# don't drift from pyproject.toml.
try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version('split')
except Exception:
    release = '0.0.1a'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # Pulls docstrings from your code
    'sphinx.ext.napoleon',     # Parses NumPy/Google style docstrings
    'sphinx.ext.viewcode',     # Adds links to highlighted source code
    'sphinx.ext.intersphinx',  # Cross-reference external project docs
    'myst_parser',             # Enables Markdown (.md) support
]

templates_path = ['_templates']
exclude_patterns = []
language = 'en'

# Tell MyST to auto-generate anchor links for H1, H2, and H3 headers
myst_heading_anchors = 3

# Read the Docs cannot import the GPU stack, so mock heavy/optional imports
# at documentation build time. Anything autodoc tries to import that lives
# in one of these packages will be replaced with a harmless mock.
autodoc_mock_imports = [
    'cupy',
    'emcee',
    'eryn',
    'few',
    'fastlisaresponse',
    'lisatools',
    'stableemrifisher',
    'corner',
    'tqdm',
]

# Cross-link standard scientific Python projects.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '../assets/logos/square-light.svg'
html_theme_options = {
    'logo_only': True,
}
