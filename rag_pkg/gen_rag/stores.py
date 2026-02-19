#!/usr/bin/env python3
"""
Vector stores

File: gen_rag/stores.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from neuroml_ai_utils.llm import setup_embedding
from neuroml_ai_utils.logging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)
from pydantic import BaseModel

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class VectorStoreInfo(BaseModel):
    name: str
    path: str
    loaded_object: Optional[Any] = None


class PerDomainConfig(BaseModel):
    description: str
    vector_stores: list[VectorStoreInfo]


class VectorStoresConfig(BaseModel):
    default_k: int = 5
    k_max: int = 10
    embedding_model: str  #  = "ollama:bge-m3"
    domains: Dict[str, PerDomainConfig]
    answer_other_queries: bool = False
    refusal_message: str = "I am unable to answer the question based on the information provided to me. Please try another query."


class Vector_Stores(object):
    """Vector stores"""

    def __init__(
        self,
        vs_config_file: str,
        logging_level: int = logging.DEBUG,
    ):
        """Init"""
        # per store
        self.default_k = 5
        self.k_max = 10
        self.k = self.default_k
        self.sim_thresh = 0.15
        # set a default
        self.embeddings = None
        self.vs_config_file = vs_config_file
        self.vs_config: VectorStoresConfig

        self.logger = logging.getLogger("NeuroML-AI")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(logger_formatter_info)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging_level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(logger_formatter_other)
        self.logger.addHandler(stderr_handler)

    def setup(self):
        """Setup embeddings"""
        self.load_config()
        self.embeddings = setup_embedding(self.embedding_model, self.logger)
        # extract model name
        if self.embedding_model.lower().startswith("huggingface:"):
            # strip suffix/prefix
            self.embedding_model = (
                self.embedding_model.replace("huggingface:", "")
                .replace(":cheapest", "")
                .replace(":fastest", "")
            )
            # strip collection name
            splits = self.embedding_model.split("/")
            self.embedding_model = "".join(splits[1:])
        elif self.embedding_model.lower().startswith("ollama:"):
            # strip prefix
            self.embedding_model = self.embedding_model.replace("ollama:", "")

    def load_config(self):
        """Load domains this RAG is going to answer for from config file"""
        self.logger.debug(f"{self.vs_config_file =}")
        with open(self.vs_config_file) as f:
            domain_info = json.load(f)
            self.vs_config = VectorStoresConfig(**domain_info)
        self.embedding_model = self.vs_config.embedding_model
        self.default_k = self.vs_config.default_k
        self.k_max = self.vs_config.k_max
        self.logger.debug(f"{self.vs_config =}")

    def inc_k(self, inc: int = 1):
        """Increase k by inc

        :param inc: int to increase k by
        :returns: True if k was increased, False otherwise

        """
        if (self.k + inc) <= self.k_max:
            self.k += inc
            self.logger.debug(f"k increased to {self.k =}")
            return True

        return False

    def reset_k(self):
        """Reset k to default value"""
        self.k = self.default_k
        self.logger.debug(f"k reset to {self.k =}")

    def load_all_stores(self):
        """Load all vector stores"""
        for domain_name in self.domains:
            self.load(domain_name)

    @property
    def domains(self):
        """Get a list of all domains"""
        return list(self.vs_config.domains.keys())

    def load(self, domain_name: str):
        """Create/load the vector store"""
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

            # if not absolute, it must be in a data folder in the location of
            # this file
            if not store_path.is_absolute():
                store_path = Path.cwd() / store_path
                self.logger.debug(
                    f"Store path made absolute relative to cwd: {store_path}"
                )

            if not store_path.is_dir():
                self.logger.error(f"Could not find folder: {store_path}")
                raise FileNotFoundError(f"Could not find folder: {store_path}")

            # check that it is a pre-existing DB
            store_db = store_path / Path("chroma.sqlite3")
            assert store_db.is_file()

            self.logger.debug(
                f"Loading Chroma vector store '{store_name}' from path {store_path.absolute()}"
            )

            chroma_client_settings_text = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=str(store_path.absolute()),
                anonymized_telemetry=False,
            )
            # NOTE:
            # Must match the values set when the store was created
            loaded_store = Chroma(
                collection_name=store_name,
                embedding_function=self.embeddings,
                client_settings=chroma_client_settings_text,
            )
            # save it as the loaded object
            store.loaded_object = loaded_store

            self.logger.debug(
                f"Finished loading Chroma vector store '{store_name}' from path {store_path.absolute()}"
            )

    def retrieve(self, domain_name: str, query: str) -> list[tuple[Document, float]]:
        """Retrieve embeddings from documentation to answer a query

        :param domain_name: name of domain
        :type domain_name: str
        :param query: user query
        :type query: str
        :returns: list of tuples (document, score)

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
                    f"No data retrieved. Check VS is correctly populated and that the collection name is correct ({store.name})"
                )
            res.extend(data)

        return res
