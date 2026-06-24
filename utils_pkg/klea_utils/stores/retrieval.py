#!/usr/bin/env python3
"""
Vector stores retrieval manager

File: klea_utils/stores/retrieval.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from langchain_core.documents import Document

from ..llm import setup_embedding
from .config import VectorStoresConfig
from .utils import instantiate_vector_store


class VectorStores:
    """Manages domain-specific vector stores.

    Loads vector stores on demand per domain and provides similarity search
    retrieval across multiple stores within a domain.

    Store paths use a URI-style scheme prefix to identify the backend:

    - ``chroma:/path/to/dir``  ---  ChromaDB (persistent, local disk)
    - ``qdrant:http://host:port``  ---  Qdrant (remote HTTP)
    - ``pgvector:postgresql://host/db``  ---  PGVector (PostgreSQL)
    """

    def __init__(self, vs_config: VectorStoresConfig, logger: logging.Logger):
        """Initialise vector stores manager.

        :param logger: Logger instance (injected from orchestrator)
        """
        self.default_k = 5
        self.k_max = 10
        self.k = self.default_k
        self.sim_thresh = 0.15
        self.embeddings = None
        self.vs_config: VectorStoresConfig = vs_config
        self.logger = logging.getLogger(f"{logger.name}.{self.__class__.__name__}")
        self.embedding_model: str | None = None

    def setup(self) -> None:
        """Load configuration and initialise embedding model."""
        self._load_config()
        self.embeddings = setup_embedding(self.embedding_model, self.logger)

        assert self.embedding_model

    def _load_config(self) -> None:
        """Load domains from the configuration file."""
        self.embedding_model = self.vs_config.embedding_model
        self.default_k = self.vs_config.default_k
        self.k_max = self.vs_config.k_max
        self.logger.debug(f"{self.vs_config =}")

        assert self.embedding_model

    def inc_k(self, inc: int = 1) -> bool:
        """Increase k by inc.

        :param inc: Amount to increase k by
        :returns: True if k was increased, False if already at max
        """
        if (self.k + inc) <= self.k_max:
            self.k += inc
            self.logger.debug(f"k increased to {self.k =}")
            return True
        return False

    def reset_k(self) -> None:
        """Reset k to default value."""
        self.k = self.default_k
        self.logger.debug(f"k reset to {self.k =}")

    def load_all_stores(self) -> None:
        """Load all vector stores for all domains."""
        for domain_name in self.domains:
            self.load(domain_name)

    @property
    def domains(self) -> list[str]:
        """Get a list of all configured domains."""
        return list(self.vs_config.domains.keys())

    def load(self, domain_name: str) -> None:
        """Load vector stores for a domain (lazy loading).

        :param domain_name: Name of the domain to load stores for
        """
        assert self.embeddings

        domain = self.vs_config.domains.get(domain_name, None)
        assert domain

        self.logger.debug(f"Got domain information: {domain}")

        stores = domain.vector_stores

        for store in stores:
            if store.loaded_object is not None:
                self.logger.debug(f"Store '{store.name}' already loaded, skipping")
                continue

            store_name = store.name
            self.logger.debug(
                f"Got store for domain {domain_name}: {store_name} ({store.path})"
            )

            store.loaded_object = self._instantiate_store(store.path, store_name)

            self.logger.debug(
                f"Finished loading vector store '{store_name}' from {store.path}"
            )

    def _instantiate_store(self, path: str, name: str):
        """Instantiate a vector store based on the URI scheme in path.

        :param path: URI-style string with scheme prefix
        :param name: Collection name for the vector store
        :returns: Instantiated LangChain VectorStore
        """
        return instantiate_vector_store(path, name, self.embeddings, self.logger)

    def retrieve(self, domain_name: str, query: str) -> list[tuple[Document, float]]:
        """Retrieve documents from vector stores for a query.

        :param domain_name: Name of the domain to search in
        :param query: User query string
        :returns: List of (document, relevance_score) tuples
        """
        self.load(domain_name)

        domain = self.vs_config.domains.get(domain_name, None)
        assert domain
        stores = domain.vector_stores

        res = []

        for store in stores:
            assert store.loaded_object
            data = store.loaded_object.similarity_search_with_relevance_scores(
                query, k=self.k, score_threshold=self.sim_thresh
            )
            self.logger.debug(f"{data =}")
            if len(data) == 0:
                self.logger.warning(
                    f"No data retrieved. Check VS is correctly populated and that "
                    f"the collection name is correct ({store.name})"
                )
            res.extend(data)

        return res
