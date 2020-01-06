# Configuration file for the Sphinx documentation builder.
#
# u This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

import climpred

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
current_year = datetime.datetime.now().year
project = 'climpred'
copyright = f'2019-{current_year}, climpred development team'
author = 'climpred development team'

version = climpred.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.napoleon',
    'sphinx.ext.imgmath',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
]

extlinks = {
    'issue': ('https://github.com/bradyrx/climpred/issues/%s', 'GH#'),
    'pr': ('https://github.com/bradyrx/climpred/pull/%s', 'GH#'),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'
source_suffix = '.rst'
master_doc = 'index'

nbsphinx_timeout = 180  # 3 minute timeout
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "alabaster"
html_theme = 'sphinx_rtd_theme'
html_logo = 'images/climpred-logo.png'
html_theme_options = {'logo_only': False, 'style_nav_header_background': '#fcfcfc'}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# Currently makes build time quite slow and doesn't add anythiing significant to docs.
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'xarray': ('https://http://xarray.pydata.org/en/stable/', None),
#     'numpy': ('https://docs.scipy.org/doc/numpy/', None),
# }

# Should only be uncommented when testing page development while notebooks
# are breaking.
# nbsphinx_allow_errors = True
