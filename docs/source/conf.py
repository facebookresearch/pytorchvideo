# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# flake8: noqa

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import mock
import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser


# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../pytorchvideo"))
sys.path.insert(0, os.path.abspath("../../"))

DEPLOY = os.environ.get("READTHEDOCS") == "True"


try:
    import torch  # noqa
except ImportError:
    for m in [
        "torch",
        "torchvision",
        "torch.nn",
        "torch.autograd",
        "torch.autograd.function",
        "torch.nn.modules",
        "torch.nn.modules.utils",
        "torch.utils",
        "torch.utils.data",
        "torchvision",
        "torchvision.ops",
        "torchvision.datasets",
        "torchvision.datasets.folder",
        "torch.utils.data.IterableDataset",
    ]:
        sys.modules[m] = mock.Mock(name=m)


# sys.modules["iopath"] = mock.Mock(name="iopath")
# sys.modules["cv2"] = mock.Mock(name="cv2")
# sys.modules["fvcore"] = mock.Mock(name="fvcore")

# -- Project information -----------------------------------------------------

project = "PyTorchVideo"
copyright = "2021, PyTorchVideo contributors"
author = "PyTorchVideo contributors"

# The full version, including alpha/beta/rc tags
import pytorchvideo
version = pytorchvideo.__version__
# The full version, including alpha/beta/rc tags
release = version


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "3.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx_markdown_tables",
]



# -- Configurations for plugins ------------
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}
# -------------------------

source_parsers = {".md": CommonMarkParser}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "build", "README.md"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "collapse_navigation": False,  # default
    "display_version": True,  # default
    "logo_only": True,  # default = False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/img/ptv_logo.png"
html_favicon = '../../website/website/static/img/logo_no_text.svg'



# setting custom stylesheets https://stackoverflow.com/a/34420612
html_context = {"css_files": ["_static/css/pytorchvideo_theme.css"]}

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pytorchvideodoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "pytorchvideo.tex", "PyTorchVideo Documentation",\
     "pytorchvideo contributors", "manual")
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pytorchvideo", "PyTorchVideo Documentation",\
     [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "PyTorchVideo",
        "PyTorchVideo Documentation",
        author,
        "PyTorchVideo",
        "One line description of project.",
        "Miscellaneous",
    )
]


def setup(app):
    from recommonmark.transform import AutoStructify

    app.add_config_value(
        "recommonmark_config",
        {
            "auto_toc_tree_section": "Contents",
            "enable_math": True,
            "enable_inline_math": True,
            "enable_eval_rst": True,
            "enable_auto_toc_tree": True,
        },
        True,
    )
    return app