"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
http://www.sphinx-doc.org/en/master/config
"""

# -- Path setup --------------------------------------------------------------

import datetime
import os
import sys

import xarray

import climpred

xarray.DataArray.__module__ = "xarray"
xarray.Dataset.__module__ = "xarray"


sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
current_year = datetime.datetime.now().year
project = "climpred"
copyright = f"2019-{current_year}, climpred development team"
today_fmt = "%Y-%m-%d"

version = climpred.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    # "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.napoleon",
    "sphinx.ext.imgmath",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
]

# autosummary_generate = True
# autodoc_typehints = "none"


# MyST config
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "html_image"]
myst_url_schemes = ["http", "https", "mailto"]

# Cupybutton configuration
# See: https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

extlinks = {
    "issue": ("https://github.com/pangeo-data/climpred/issues/%s", "GH#"),
    "pr": ("https://github.com/pangeo-data/climpred/pull/%s", "GH#"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

pygments_style = "sphinx"
source_suffix = ".rst"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
# html_theme = "pydata_sphinx_theme"
# html_theme = "sphinx_rtd_theme"
html_logo = "images/climpred-logo.png"
html_theme_options = {"logo_only": False, "style_nav_header_background": "#fcfcfc"}


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "repository_url": "https://github.com/pangeo-data/climpred",
    "use_edit_page_button": True,
    # "navbar_end": "search-field.html",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "home_page_in_toc": False,
    "extra_navbar": "",
    "navbar_footer_text": "",
}

html_context = {
    "github_user": "pangeo-data",
    "github_repo": "climpred",
    "github_version": "main",
    "doc_path": "docs",
}


nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
You can run this notebook in a `live session <https://binder.pangeo.io/v2/gh/pangeo-data/climpred/main?urlpath=lab/tree/docs/{{docname }}>`_ |Binder| or view it `on Github <https://github.com/pangeo-data/climpred/blob/main/docs/{{ docname }}>`_.
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://binder.pangeo.io/v2/gh/pangeo-data/main?urlpath=lab/tree/docs/{{ docname }}
"""  # noqa: E501


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "xskillscore": ("https://xskillscore.readthedocs.io/en/stable", None),
    "xclim": ("https://xclim.readthedocs.io/en/latest/", None),
}

# Should only be uncommented when testing page development while notebooks
# are breaking.
# nbsphinx_allow_errors = True

# nbsphinx_kernel_name = "climpred-docs"  # doesnt work
nbsphinx_allow_errors = True
nbsphinx_timeout = 600

# Napoleon configurations
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "PredictionEnsemble": "~climpred.PredictionEnsemble",
    "HindcastEnsemble": "~climpred.HindcastEnsemble",
    "PerfectModelEnsemble": "~climpred.PerfectModelEnsemble",
}

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = today_fmt


def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # Make sure we're outputting HTML
    if app.builder.format != "html":
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered


def html_page_context(app, pagename, templatename, context, doctree):
    # Disable edit button for docstring generated pages
    if "generated" in pagename:
        context["theme_use_edit_page_button"] = False


def setup(app):
    app.connect("source-read", rstjinja)
    app.connect("html-page-context", html_page_context)
