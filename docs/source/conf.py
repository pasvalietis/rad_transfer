# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, '/home/gabriel/rad_transfer')
sys.path.insert(0, '/home/gabriel/rad_transfer/rushlight/user_notebooks')

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Rushlight'
copyright = f'{datetime.now().year}, Ivan Oparin, Sabastian Fernandes'
author = 'Ivan Oparin, Sabastian Fernandes'
release = '0.1.dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "classic"
# html_static_path = ['_static']
