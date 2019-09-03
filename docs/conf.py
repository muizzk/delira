# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

from delira._version import get_versions
import os
import sys
import re

# source code directory, relative to this file, for sphinx-build
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))

# -- Project information -----------------------------------------------------

project = 'delira'
copyright = '2019, Justus Schock, Michael Baumgartner, Oliver Rippel, Christoph Haarburger'
author = 'Justus Schock, Michael Baumgartner, Oliver Rippel, Christoph Haarburger'


def read_file(file):
    with open(file) as f:
        content = f.read()
    return content


whole_version = get_versions()["version"]
# The short X.Y version
version = whole_version.split("+", 1)[0]
# The full version, including alpha/beta/rc tags
release = whole_version  # delira.__version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosectionlabel',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "collapse_navigation": False,
    "logo_only": True
}

html_logo = "_static/logo/delira.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
        'localtoc.html',
        'sourcelink.html',
    ]
}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'deliradoc'


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
    (master_doc, 'delira.tex', 'delira Documentation',
     author, 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'delira', 'delira Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'delira', 'delira Documentation',
     author, 'delira', 'One line description of project.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'https://docs.python.org/': None,
    'trixi': (
        'https://trixi.readthedocs.io/en/latest/',
        None),
    'torch': (
        'https://pytorch.org/docs/stable/',
        None),
    'tensorflow': (
        'https://www.tensorflow.org/api_docs/python/',
        None),
    'chainer': (
        'https://docs.chainer.org/en/stable/',
        None),
    'sklearn': (
        'https://scikit-learn.org/stable/documentation/',
        None),
    'numpy': (
        'https://docs.scipy.org/doc/numpy/reference/',
        None),
    'scipy': (
        'https://docs.scipy.org/doc/scipy/reference/'
    )
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
autoclass_content = 'both'
add_module_names = False

autodoc_default_flags = ['members',
                         'undoc-members',
                         'private-members',
                         'inherited-members',
                         'show-inheritance']

autodoc_inherit_docstrings = True

autodoc_mock_imports = [
    "numpy",
    "torchvision",
    "torch",
    "skimage",
    "sklearn",
    "jupyter",
    "flake8"
    "pytest-cov",
    "autopep8",
    "ipython",
    "joblib",
    "pillow",
    "SimpleITK",
    "pylint",
    "tqdm",
    "visdom",
    "pyyaml",
    "trixi",
    "batchgenerators",
    "psutil",
    "nested_lookup",
    "colorlover",
    "flask",
    "graphviz",
    "matplotlib",
    "seaborn",
    "scipy",
    "scipy.ndimage",
    "telegram",
    "portalocker",
    "plotly",
    "PIL",
    "umap",
    "tensorflow",
    "yaml",
    "chainer"
]

# autodoc_mock_imports = [
#         "torch.optim",
#         "torch.optim.lr_scheduler",
#         "yaml",
#         "numpy",
#         "torchvision",
#         "torchvision.datasets",
#         "torch",
#         "torch.nn",
#         "torch.nn.functional",
#         "skimage",
#         "skimage.io",
#         "skimage.transform",
#         "sklearn",
#         "sklearn.model_selection",
#         "jupyter",
#         "flake8"
#         "pytest-cov",
#         "autopep8",
#         "ipython",
#         "joblib",
#         "pillow",
#         "SimpleITK",
#         "pylint",
#         "tqdm",
#         "visdom",
#         "pyyaml",
#         "trixi",
#         "trixi.experiment",
#         "trixi.logger",
#         "trixi.util",
#         "batchgenerators",
#         "batchgenerators.dataloading",
#         "batchgenerators.dataloading.data_loader",
#         "batchgenerators.transforms",
#         "psutil",
#         "nested_lookup",
#         "colorlover",
#         "flask",
#         "graphviz",
#         "matplotlib",
#         "seaborn",
#         "scipy",
#         "scipy.ndimage",
#         "telegram",
#         "portalocker",
#         "plotly",
#         "PIL",
#         "umap",
#         "PIL.Image",
#         "tensorflow",
#         "tqdm.auto",
#         "trixi.logger.tensorboard",
#         "trixi.logger.tensorboard.tensorboardxlogger",
#         "sklearn.metrics",
# ]
