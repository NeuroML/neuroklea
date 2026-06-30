Create and use a RAG system
===========================

This tutorial walks through the full lifecycle of a Klea RAG system:
preparing documents, building a vector store, configuring the RAG
pipeline, and querying it.

Overview
--------

By the end you will have:

* A Chroma vector store populated with chunks from your own documents
* A running Klea RAG server backed by that store
* Hands-on experience querying the system via the CLI and web UI

Prerequisites
-------------

* Python 3.12 or later
* :doc:`Packages installed <../install>` with Chroma and ingestion
  extras:

  .. code-block:: bash

     pip install klea_rag[chroma] klea_utils[ingest]

  .. note::

     ``klea_utils[ingest]`` pulls in `Docling <https://docling-project.github.io/docling/>`_
     and its PyTorch dependency.  The download is several hundred MB.
     On systems with a CUDA-capable GPU, PyTorch will use the GPU
     automatically for faster document processing.

* A running `Ollama <https://ollama.com/>`_ instance with the required
  models:

  .. code-block:: bash

     ollama pull qwen3:0.6b
     ollama pull llama-guard3:1b
     ollama pull bge-m3:latest

  This tutorial uses Ollama for all inference (chat, guard, and
  embeddings).  Klea supports other providers too -- see
  :doc:`../install` for HuggingFace, OpenAI, Anthropic, and other
  LangChain-compatible options.

Step 1: Prepare source documents
---------------------------------

Place the files you want to index in a single directory.  Docling
handles a wide range of formats: PDF, HTML, Markdown, DOCX, PPTX, XLSX,
images, and more (see `Docling supported formats
<https://docling-project.github.io/docling/usage/supported_formats/>`_
for the full list).

For this tutorial we will refer to this directory as
``<folder-of-files>``.

Step 2: Create a vector store
------------------------------

.. code-block:: bash

   klea-vs-create build <folder-of-files> \\
       --collection my-docs \\
       --store chroma:/path/to/my-store.db

The ``build`` command runs the full pipeline:

1. **Convert** -- every supported file is parsed with Docling into a
   structured document.
2. **Chunk** -- documents are split into token-aware chunks (450 tokens
   by default) using Docling's ``HybridChunker``.  Each chunk retains
   its heading hierarchy as metadata.
3. **Embed** -- chunks are embedded using ``bge-m3`` (or whichever
   embedding model you configure).
4. **Store** -- embeddings and text are written to a Chroma vector store
   at the path you specify.

Flags explained:

* ``--collection`` / ``-n`` -- the collection name inside the store
  (e.g. ``my-docs``).
* ``--store`` / ``-s`` -- the vector store URI; ``chroma:/path`` creates a
  persistent Chroma database at that location.
* ``--model`` / ``-m`` -- embedding model (default ``ollama:bge-m3:latest``).
* ``--max-tokens`` -- maximum tokens per chunk (default 450).
* ``--force`` / ``-f`` -- re-process all files even if previously cached.

Re-running ``klea-vs-create build`` on the same directory is safe --
it skips files whose content has not changed and skips chunks whose
hashes already exist in the store (idempotent).  Adding new files to
the source directory and re-running adds only the new content
(incremental ingestion).

The source directory will contain a ``.klea-cache/`` folder after the
first run.  This caches converted chunks so subsequent runs skip the
expensive Docling conversion.

Step 3: Configure the RAG system
---------------------------------

Create an environment file (e.g. ``my-rag.env``):

.. code-block:: ini

   KLEA_RAG_CHAT_MODEL=ollama:qwen3:0.6b
   KLEA_RAG_GUARD_MODEL=ollama:llama-guard3:1b
   KLEA_RAG_EMBEDDING_MODEL=ollama:bge-m3:latest
   KLEA_RAG_APP_CONFIG_FILE=my-config.json

Create the JSON configuration file (``my-config.json``) that wires the
vector store to a domain:

.. code-block:: json

   {
       "general": {
           "default_k": 3,
           "k_max": 5,
           "non_domain_chat": true,
           "fallback_to_training_data": true
       },
       "domains": {
           "MyDomain": {
               "description": "Documents related to my project",
               "vector_stores": [
                   {
                       "name": "my-docs",
                       "path": "chroma:/path/to/my-store.db"
                   }
               ]
           }
       }
   }

The ``general`` section controls retrieval behaviour:

* ``default_k`` -- number of documents to retrieve per query.
* ``k_max`` -- maximum ``k`` when the evaluator requests more context.
* ``non_domain_chat`` -- whether to fall back to the LLM's training data
  for questions that do not match any domain.
* ``fallback_to_training_data`` -- whether to let the LLM answer from its
  own knowledge when retrieval returns nothing useful.

Each entry under ``domains`` defines a knowledge area with one or more
vector stores.  The ``description`` helps the classifier route queries
to the right domain.

.. seealso::

   :doc:`../install` for details on HuggingFace, OpenAI, and other
   provider model naming conventions.

Step 4: Start the RAG server
-----------------------------

.. code-block:: bash

   KLEA_RAG_ENV_FILE=my-rag.env klea-rag-serve serve

The server loads the configuration, initialises the embedding model,
and compiles the LangGraph pipeline.  Once ready, check it is alive:

.. code-block:: bash

   curl http://127.0.0.1:8005/health/ready

A ``200 OK`` response means the system is ready to accept queries.

Step 5: Query the RAG
----------------------

Single-query mode is the quickest way to test:

.. code-block:: bash

   klea-rag cli --single-query "What does my collection of documents cover?"

For an interactive session:

.. code-block:: bash

   klea-rag cli

Type your questions at the prompt.  Use ``quit`` to exit.

For a graphical interface, launch the Streamlit web UI:

.. code-block:: bash

   klea-rag web

All three methods connect to the running server at ``http://127.0.0.1:8005``
by default.  Use ``--server`` to point at a different address.

Going further
--------------

Once the basic pipeline works, here are natural next steps:

**Metadata enrichment**
   Add source URLs or other metadata to retrieved chunks.  First run
   ``klea-vs-create chunk`` to generate a ``metadata-map.template.json``,
   fill in the values, then ``klea-vs-create store --metadata-map <file>``.
   See ``klea-vs-create --help`` for examples.

**Different embedding models**
   Swap ``ollama:bge-m3:latest`` for a HuggingFace embedding model
   (see :doc:`../install` for model naming conventions).

**Multiple domains**
   Add more ``domains`` entries in the JSON config, each with its own
   vector store and description.  The classifier will route queries
   automatically.

**MCP tools**
   Add ``mcp_servers`` to a domain config to give the LLM access to
   external tools (e.g. a NeuroML validation server).  See the example
   in ``rag_pkg/klea_rag.json``.

**Separate chunk-and-store workflow**
   Use ``klea-vs-create chunk`` to convert and cache without writing
   to a store, then ``klea-vs-create store`` later.  This lets you
   inspect the chunks and edit the metadata map before embedding.

Troubleshooting
---------------

**Ollama is not running**
   Start it with ``ollama serve`` or run Ollama as a system service.

**Model not found**
   Ensure you have pulled all three models (chat, guard, embedding).
   Run ``ollama list`` to see what is available.

**Server fails to start**
   Check that ``KLEA_RAG_ENV_FILE`` points to a valid env file and that
   the JSON config file path inside it is correct.  Look for JSON syntax
   errors (trailing commas, missing quotes).

**Queries return empty or irrelevant results**
   Increase ``default_k`` in the JSON config.  Verify the vector store
   path and collection name match.  Check that your source files are in
   a format Docling supports.

.. seealso::

   * :doc:`../cli/klea-vs-create` -- full CLI reference for vector store
     creation
   * :doc:`../cli/klea-rag-serve` -- server CLI reference
   * :doc:`../cli/klea-rag` -- client CLI reference
   * :class:`~klea_utils.stores.ingestion.VSBuilder` -- Python API for
     ingestion
   * :class:`~klea_utils.stores.retrieval.VSRetriever` -- Python API for
     retrieval
