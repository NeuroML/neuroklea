Welcome to Klea
===============

Knowledge vaLidated Expert AI Assistant for Neuroscience.

Klea is a suite of AI tools for the `NeuroML <https://neuroml.org>`_
community.  It provides a generic RAG pipeline, an AI-assisted coding
workflow system, and an MCP server for NeuroML development.

Architecture
------------

The project is organised as a monorepo with four installable packages:

.. list-table::
   :header-rows: 1

   * - Directory
     - Package
     - CLI
     - Purpose
   * - ``utils_pkg``
     - ``klea_utils``
     - ``klea-vs-create``
     - Shared utilities, vector store management, base graph classes
   * - ``rag_pkg``
     - ``klea_rag``
     - ``klea-rag``, ``klea-rag-serve``
     - Generic RAG pipeline with multi-domain support
   * - ``code_pkg``
     - ``klea_code``
     - ``klea-code``
     - AI-assisted coding and workflow system
   * - ``mcp_pkg``
     - ``neuroml_mcp``
     - ``nml-mcp``
     - MCP server for NeuroML tooling

Each package is built on a shared foundation in ``klea_utils``, which
provides LLM setup, vector store abstraction, and the
:class:`~klea_utils.graph.base.BaseLangGraph` orchestrator framework.

Quick start
-----------

1. Install the packages::

      pip install -r requirements.txt

2. Set up a configuration file (see :doc:`install` for details).

3. Run the CLI::

      klea-rag --help

Quick links
-----------

* :doc:`install`
* :doc:`contributing`
* `GitHub <https://github.com/NeuroML/neuroklea>`_


.. toctree::
   :hidden:

   install
   contributing
   cli/index
   api/index
   tutorials/index
