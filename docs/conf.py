# Configuration file for the Sphinx documentation builder.
project = "Klea"
copyright = "2026, NeuroML contributors"
author = "NeuroML contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.typer",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Klea"
html_show_sphinx = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
