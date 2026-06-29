#!/usr/bin/env python3
"""
Sphinx documentation configuration

File: conf.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

# Configuration file for the Sphinx documentation builder.
project = "Klea"
copyright = "2026, NeuroML contributors"
author = "NeuroML contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.typer",
]

# mcp 1.28+ and its fastmcp/langchain deps crash on import with
# pydantic 2.13 under Sphinx's autodoc importer. Mock them.
autodoc_mock_imports = [
    "mcp",
    "fastmcp",
    "langchain",
    "langchain_core",
    "langgraph",
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
