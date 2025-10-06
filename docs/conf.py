"""
Author: Vasiliy Zdanovskiy
email: vasilyvz@gmail.com

Sphinx configuration for BHLFF documentation.

This module configures Sphinx for building the BHLFF documentation,
including extensions, theme settings, and project metadata.

Physical Meaning:
    Documentation serves as the interface between the theoretical
    7D phase field framework and practical implementation,
    ensuring proper understanding and usage of the BHLFF package.
"""

import os
import sys
from pathlib import Path

# Add the bhlff directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project information
project = "BHLFF"
copyright = "2024, Vasiliy Zdanovskiy"
author = "Vasiliy Zdanovskiy"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Templates path
templates_path = ["_templates"]

# List of patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Theme options
html_theme_options = {
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MathJax configuration
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True


# Custom CSS
def setup(app):
    """Setup function for Sphinx."""
    app.add_css_file("custom.css")
