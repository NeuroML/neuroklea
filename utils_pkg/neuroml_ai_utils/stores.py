#!/usr/bin/env python3
"""
Vector stores management for RAG systems

File: neuroml_ai_utils/stores.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel

from .llm import setup_embedding


class FallbackConfig(BaseModel):
    """Configuration for fallback to training data."""

    enabled: bool = False
    warning: str = ""


class VectorStoreInfo(BaseModel):
    """Information about a single vector store."""

    name: str
    path: str
    loaded_object: Optional[Any] = None


class PerDomainConfig(BaseModel):
    """Configuration for a single domain."""

    description: str
    vector_stores: list[VectorStoreInfo]


class VectorStoresConfig(BaseModel):
    """Top-level vector stores configuration."""

    default_k: int = 5
    k_max: int = 10
    pre_prompt: str = ""
    embedding_model: str
    domains: Dict[str, PerDomainConfig]
    fallback_to_training_data: FallbackConfig


class VectorStores:
    """Manages domain-specific ChromaDB vector stores.

    Loads vector stores on demand per domain and provides similarity search
    retrieval across multiple stores within a domain.
    """

    def __init__(
        self,
        vs_config_file: str,
        logger: logging.Logger,
    ):
        """Initialise vector stores manager.

        :param vs_config_file: Path to the JSON configuration file
        :param logger: Logger instance (injected from orchestrator)
        """
        self.default_k = 5
        self.k_max = 10
        self.k = self.default_k
        self.sim_thresh = 0.15
        self.embeddings = None
        self.vs_config_file = vs_config_file
        self.vs_config: VectorStoresConfig
        self.logger = logger

    def setup(self) -> None:
        """Load configuration and initialise embedding model."""
        self._load_config()
        self.embeddings = setup_embedding(self.embedding_model, self.logger)

        # Extract model name for collection naming
        if self.embedding_model.lower().startswith("huggingface:"):
            self.embedding_model = (
                self.embedding_model.replace("huggingface:", "")
                .replace(":cheapest", "")
                .replace(":fastest", "")
            )
            splits = self.embedding_model.split("/")
            self.embedding_model = "".join(splits[1:])
        elif self.embedding_model.lower().startswith("ollama:"):
            self.embedding_model = self.embedding_model.replace("ollama:", "")

    def _load_config(self) -> None:
        """Load domains from the configuration file."""
        self.logger.debug(f"{self.vs_config_file =}")
        with open(self.vs_config_file) as f:
            domain_info = json.load(f)
            self.vs_config = VectorStoresConfig(**domain_info)
        self.embedding_model = self.vs_config.embedding_model
        self.default_k = self.vs_config.default_k
        self.k_max = self.vs_config.k_max
        self.logger.debug(f"{self.vs_config =}")

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
        assert stores

        for store in stores:
            store_name = store.name
            store_path = Path(store.path)
            self.logger.debug(
                f"Got store for domain {domain_name}: {store_name} ({store_path})"
            )

            # If not absolute, resolve relative to cwd
            if not store_path.is_absolute():
                store_path = Path.cwd() / store_path
                self.logger.debug(
                    f"Store path made absolute relative to cwd: {store_path}"
                )

            if not store_path.is_dir():
                self.logger.error(f"Could not find folder: {store_path}")
                raise FileNotFoundError(f"Could not find folder: {store_path}")

            # Check that it is a pre-existing DB
            store_db = store_path / Path("chroma.sqlite3")
            assert store_db.is_file()

            self.logger.debug(
                f"Loading Chroma vector store '{store_name}' from path {store_path.absolute()}"
            )

            chroma_client_settings = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=str(store_path.absolute()),
                anonymized_telemetry=False,
            )
            loaded_store = Chroma(
                collection_name=store_name,
                embedding_function=self.embeddings,
                client_settings=chroma_client_settings,
            )
            store.loaded_object = loaded_store

            self.logger.debug(
                f"Finished loading Chroma vector store '{store_name}' from path {store_path.absolute()}"
            )

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
        assert stores

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
