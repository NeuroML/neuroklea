#!/usr/bin/env python3
"""
Script to generate NeuroML vector stores from provided data

File: nml.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import mimetypes
from glob import glob
from hashlib import sha256
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from neuroml_ai_utils.llm import setup_embedding

logging.basicConfig(level=logging.WARNING)


class NML(object):
    # limit to two header levels
    md_headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        # ("###", "Header 3"),
        # ("####", "Header 4"),
    ]

    """NeuroML vector store generator"""

    def __init__(self, embedding_model: str, logging_level: int = logging.DEBUG):
        """TODO: to be defined."""
        self.chunk_size = 600
        self.chunk_overlap = 60
        self.embedding_model = embedding_model

        my_path = Path(__file__).parent
        self.stores_sources_path = f"{my_path}/sources"

        self.logger = logging.getLogger("NeuroML-doc-embeddings")
        self.logger.setLevel(logging_level)
        self.logger.propagate = True

    def setup(self):
        """Setup embeddings"""
        self.embeddings = setup_embedding(self.embedding_model, self.logger)
        # extract model name
        if self.embedding_model.lower().startswith("huggingface:"):
            # strip suffix/prefix
            self.embedding_model = self.embedding_model.replace("huggingface:", "")
            # strip collection name
            splits = self.embedding_model.split("/")
            self.embedding_model = "".join(splits[1:])
        elif self.embedding_model.lower().startswith("ollama:"):
            # strip prefix
            self.embedding_model = self.embedding_model.replace("ollama:", "")

        self.embedding_model.replace(":cheapest", "").replace(":fastest", "")

    def create(self):
        """Create the vector store"""
        assert self.embeddings

        self.logger.debug("Setting up/loading Chroma vector store")

        self.logger.debug(f"{self.stores_sources_path =}")
        vec_store_sources = glob(f"{self.stores_sources_path}/*", recursive=False)
        self.logger.debug(f"{vec_store_sources =}")

        assert len(vec_store_sources)

        for src in vec_store_sources:
            self.logger.debug(f"Setting up vector store: {src}")
            src_path = Path(src)

            assert src_path.is_dir()

            vs_persist_dir = f"./vector-stores/{src_path.name}_{self.embedding_model.replace(':', '_')}.db"
            self.logger.debug(f"{vs_persist_dir =}")

            chroma_client_settings_text = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=vs_persist_dir,
                anonymized_telemetry=False,
            )
            store = Chroma(
                collection_name=src_path.name,
                embedding_function=self.embeddings,
                client_settings=chroma_client_settings_text,
            )

            info_files = glob(f"{src}/*.md", recursive=True)
            url_maps = glob(f"{src}/*.json", recursive=True)
            self.logger.debug(f"Loaded {len(info_files)} files from {src}")
            self.logger.debug(f"Loaded {len(url_maps)} url map files from {src}")

            # only a single url map file is allowed here
            assert len(url_maps) <= 1

            if len(url_maps) == 1:
                with open(url_maps[0], "r") as f:
                    url_map_data = json.load(f)
            else:
                url_map_data = {}

            for info_file in info_files:
                try:
                    file_type = mimetypes.guess_file_type(info_file)[0]
                except AttributeError:
                    # for py<3.13
                    file_type = mimetypes.guess_type(info_file)[0]

                if file_type:
                    if "markdown" in file_type:
                        self.add_md(store, info_file, url_map_data)
                    else:
                        self.logger.warning(
                            f"File {info_file} is of type {file_type} which is not currently supported. Skipping"
                        )
                else:
                    self.logger.warning(
                        f"Could not guess file type for file {info_file}. Skipping"
                    )

    def add_md(self, store, file, url_map):
        """Add a markdown file to the vector store

        We add the file hash as extra metadata so that we can filter on it
        later.

        TODO: Handle images referenced in the markdown file.

        For this, we need to use the same metadata for the chunks and for the
        images in those chunks when they're added to the text and image stores.
        The text chunks need to have an id each, and a list of figures too. The
        images being added will need to have the document/file id, and the
        figure ids.

        For retrieval, we will first run the similarity search on both the text
        and images. For text results, we will retrieve any linked images.

        Note that for text only LLMs, only the associated metadata of the
        obtained images (captions and so on) can be used in the context. To use
        the images too, we need to use multi-modal LLMs.
        """
        file_path = Path(file)
        file_hash = sha256(file_path.name.encode("utf-8")).hexdigest()
        already_added = store.get(where={"file_hash": file_hash})

        if already_added and already_added["ids"]:
            self.logger.debug(f"File already exists in vector store: {file_path}")
            return

        self.logger.debug(f"Adding markdown file to text vector store: {file_path}")
        with open(file, "r") as f:
            md_doc = f.read()
            self.logger.debug(f"Length of loaded file: {len(md_doc.split())}")
            md_splitter = MarkdownHeaderTextSplitter(
                self.md_headers_to_split_on, strip_headers=False
            )
            md_splits = md_splitter.split_text(md_doc)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            splits = text_splitter.split_documents(md_splits)
            for split in splits:
                # get url
                # header 1
                url = None
                if "Header 1" in split.metadata.keys():
                    url = url_map.get(split.metadata["Header 1"], None)
                # try header 2: more specific
                if "Header 2" in split.metadata.keys():
                    url = url_map.get(split.metadata["Header 2"], None)
                # fall back to default url
                if not url:
                    url = url_map.get("DEFAULT_URL")

                meta_update = {
                    "file_hash": file_hash,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "url": url,
                }
                self.logger.debug(f"{meta_update =}")

                split.metadata.update(meta_update)

            self.logger.debug(f"Length of split docs: {len(splits)}")
            _ = store.add_documents(documents=splits)


if __name__ == "__main__":
    converter = NML(embedding_model="ollama:bge-m3:latest")
    converter.setup()
    converter.create()
