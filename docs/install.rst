Installation
============

Requirements
------------

* Python 3.12 or later
* A `LangChain-compatible inference provider
  <https://docs.langchain.com/oss/python/integrations/providers/overview>`_
  for LLM access (e.g. OpenAI, Anthropic, Ollama, HuggingFace, etc.)

Clone
-----

Until packages are published on PyPI, clone the repository and work
from the ``development`` branch::

   git clone https://github.com/NeuroML/neuroklea.git
   cd neuroklea
   git checkout development

.. note::

   PyPI releases coming soon.  Once published, ``pip install klea-rag``
   (and friends) will work directly and the clone step will not be
   needed.

Install all packages
--------------------

From the repository root::

   pip install -r requirements.txt

This installs all four packages (``klea_utils``, ``klea_rag``,
``klea_code``, ``neuroml_mcp``) with their dependencies.

The recommended tool for dependency management is `uv
<https://github.com/astral-sh/uv>`_.

``uv pip install -r requirements.txt`` is a faster alternative to
``pip install`` used above.

Per-package install
-------------------

Each package can be installed individually from its directory::

   cd utils_pkg && pip install .
   cd rag_pkg   && pip install .
   cd code_pkg  && pip install .
   cd mcp_pkg   && pip install .

Optional extras
----------------

``klea_utils`` extras
~~~~~~~~~~~~~~~~~~~~~~

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

``klea_rag`` extras
~~~~~~~~~~~~~~~~~~~~

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

Dev install
-----------

For development, install in editable mode with dev extras::

   pip install -r requirements-dev.txt

.. seealso::

   :doc:`contributing` for the full development workflow.

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
