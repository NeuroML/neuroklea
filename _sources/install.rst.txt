Installation
============

Requirements
------------

* Python 3.12 or later
* A `LangChain-compatible inference provider
  <https://docs.langchain.com/oss/python/integrations/providers/overview>`_
  for LLM access (e.g. OpenAI, Anthropic, Ollama, HuggingFace, etc.)

klea-rag and klea-utils (PyPI)
-------------------------------

The RAG and utilities packages are available on PyPI::

   pip install klea-rag

This installs ``klea_rag`` and its core dependency ``klea_utils``.
Optional extras for vector store backends and document ingestion are
listed below — add them with e.g. ``pip install klea-rag[chroma]``.

If you use `uv <https://github.com/astral-sh/uv>`_, replace ``pip`` with
``uv pip``.

klea-utils extras
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
     - Purpose
   * - ``chroma``
     - ``langchain-chroma``, ``chromadb``
     - `Chroma <https://github.com/chroma-core/chroma>`_ vector store support
   * - ``pgvector``
     - ``langchain-postgres``
     - `pgvector <https://github.com/pgvector/pgvector-python>`_ support
   * - ``qdrant``
     - ``langchain-qdrant``
     - `Qdrant <https://github.com/qdrant/qdrant>`_ vector store support
   * - ``ingest``
     - ``docling``, ``typer``, ``xxhash``
     - Document ingestion pipeline
   * - ``full``
     - All of the above
     - All optional extras

Usage::

   pip install klea_utils[chroma]

klea-rag extras
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
     - Purpose
   * - ``chroma``
     - ``klea_utils[chroma]``
     - `Chroma <https://github.com/chroma-core/chroma>`_ support for RAG
   * - ``pgvector``
     - ``klea_utils[pgvector]``
     - `pgvector <https://github.com/pgvector/pgvector-python>`_ support for RAG
   * - ``qdrant``
     - ``klea_utils[qdrant]``
     - `Qdrant <https://github.com/qdrant/qdrant>`_ support for RAG
   * - ``full``
     - All vector store extras
     - All RAG optional extras

Usage::

   pip install klea_rag[full]

klea-code and neuroml-mcp (from source)
----------------------------------------

These packages are in active development.  They are not yet on PyPI.
To install them, clone the repository and follow the
:doc:`development workflow <contributing>`.

Configuration
-------------

Both the RAG and Code packages load configuration from:

1. An env file (``k=v`` format):

   * ``KLEA_RAG_ENV_FILE`` or ``rag.env`` for the RAG system
   * ``KLEA_CODE_ENV_FILE`` or ``klea_code.env`` for the Code system

2. A JSON configuration file referenced inside the env file.

   * ``rag_pkg/klea_rag.json`` for RAG domains and vector stores
   * ``code_pkg/mcp.json`` for Code MCP server configuration

Example env file::

   KLEA_RAG_CHAT_MODEL=ollama:qwen3:0.6b
   KLEA_RAG_EMBEDDING_MODEL=ollama:bge-m3
   KLEA_RAG_APP_CONFIG_FILE=/path/to/klea_rag.json

Choosing models
~~~~~~~~~~~~~~~

Each model provider requires its corresponding LangChain integration
package to be installed.  For example, for Ollama::

   pip install langchain-ollama

See the `LangChain provider docs
<https://docs.langchain.com/oss/python/integrations/providers/overview>`_
for other providers and their package names.  ``langchain-huggingface``
is included automatically via ``klea_utils``.

Model names are prefixed according to their provider:

* ``ollama:<model_name>:<tag>`` for Ollama models
* ``huggingface:<model_id>`` for HuggingFace inference providers.
  HuggingFace models additionally require the ``HF_TOKEN`` environment
  variable to be set (see `HuggingFace tokens
  <https://huggingface.co/docs/hub/security-tokens>`_).
* Others (e.g. OpenAI, Anthropic) use their standard model names and
  environment variables as supported by LangChain.
