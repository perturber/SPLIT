import os
import sys
# Point to your src directory so autodoc can find the 'split' package
sys.path.insert(0, os.path.abspath('../../src/split'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SPLIT'
copyright = '2026, Shubham Kejriwal'
author = 'Shubham Kejriwal'
release = '0.0.1a'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Pulls docstrings from your code
    'sphinx.ext.napoleon',     # Parses NumPy/Google style docstrings
    'sphinx.ext.viewcode',     # Adds links to highlighted source code
    'myst_parser',             # Enables Markdown (.md) support
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
