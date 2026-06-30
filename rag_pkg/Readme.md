# Klea RAG

Klea is a suite of AI tools for Neuroscience.

This package provides a generic RAG (Retrieval Augmented Generation)
pipeline with multi-domain support.  It implements a
LangChain/LangGraph state machine for answering queries over your own
documents.  Features:

- multi-domain support with automatic query classification
- vector store retrieval from Chroma, Qdrant, and PGVector backends
- answer evaluation with iterative improvement loops
- MCP tool integration for live data access
- FastAPI server, CLI client, and Streamlit web UI

Documentation: https://neuroklea.org
