#!/usr/bin/env python3
"""
Test vector store ingestion.

File: tests/test_stores_ingestion.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import tempfile
from pathlib import Path

import pytest
from ollama import ResponseError

from klea_utils.stores.ingestion import VSBuilder
from klea_utils.stores.utils import instantiate_vector_store

TEST_MD_CONTENT = """# Test Document
## Section 1
This is some content in section 1. It has enough text to produce at least
one chunk for the vector store.

## Section 2
This is content in section 2. It also has enough text to produce at least
one chunk for the vector store. We need enough text here to make sure
the chunker actually produces some chunks.
"""

TEST_MD_TWO = """# Second Document
## Overview
This is another test file. It will be used to test incremental
ingestion where new files are added to an existing store.
"""


class TestIngestion:
    """Test vector store ingestion."""

    def setup_method(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_path = Path(self.tmpdir.name)
        self.logger = logging.getLogger("test_ingestion")

    def teardown_method(self):
        self.tmpdir.cleanup()

    @pytest.mark.localonly
    def test_build_chroma(self):
        """Test building a chroma store from a source directory."""
        md_file = self.tmpdir_path / "test.md"
        md_file.write_text(TEST_MD_CONTENT)
        self.logger.info(f"Wrote test file: {md_file}")

        store_dir = self.tmpdir_path / "chroma_store"
        store_uri = f"chroma:{store_dir}"
        collection_name = "test_build"
        self.logger.info(f"Store URI: {store_uri}, collection: {collection_name}")

        try:
            builder = VSBuilder(
                embedding_model="ollama:bge-m3:latest",
                logger=self.logger,
            )
            builder.setup()
            self.logger.info("Builder set up with Ollama embeddings")
            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("Build completed")

            store = instantiate_vector_store(
                store_uri, collection_name, builder.embeddings, self.logger
            )
            result = store.get()
            self.logger.info(f"Store contains {len(result['ids'])} chunks")
            assert result["ids"], "No documents found in store"
            assert len(result["ids"]) > 0

            first_meta = result["metadatas"][0]
            assert "file_hash" in first_meta
            assert "file_name" in first_meta
            assert "source_path" in first_meta
            assert "headings" in first_meta
            assert first_meta["file_name"] == "test.md"
            self.logger.info(
                f"First chunk metadata verified: headings={first_meta['headings']}"
            )

        except ResponseError as e:
            pytest.skip(str(e))
        except ConnectionError as e:
            pytest.skip(str(e))

    @pytest.mark.localonly
    def test_build_idempotent(self):
        """Test that building twice produces the same result."""
        md_file = self.tmpdir_path / "test.md"
        md_file.write_text(TEST_MD_CONTENT)
        self.logger.info(f"Wrote test file: {md_file}")

        store_dir = self.tmpdir_path / "chroma_store"
        store_uri = f"chroma:{store_dir}"
        collection_name = "test_idempotent"

        try:
            builder = VSBuilder(
                embedding_model="ollama:bge-m3:latest",
                logger=self.logger,
            )
            builder.setup()
            self.logger.info("Builder set up with Ollama embeddings")

            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("First build completed")

            store = instantiate_vector_store(
                store_uri, collection_name, builder.embeddings, self.logger
            )
            first_result = store.get()
            first_count = len(first_result["metadatas"])
            self.logger.info(f"First build: {first_count} chunks")

            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("Second (idempotent) build completed")

            second_result = store.get()
            second_count = len(second_result["metadatas"])
            self.logger.info(f"Second build: {second_count} chunks")

            assert first_count == second_count, (
                f"Expected same count after idempotent build "
                f"({first_count} != {second_count})"
            )
            self.logger.info("Idempotency verified — counts match")

        except ResponseError as e:
            pytest.skip(str(e))
        except ConnectionError as e:
            pytest.skip(str(e))

    @pytest.mark.localonly
    def test_build_incremental(self):
        """Test adding a new file after initial build."""
        md_file = self.tmpdir_path / "test.md"
        md_file.write_text(TEST_MD_CONTENT)
        self.logger.info(f"Wrote initial test file: {md_file}")

        store_dir = self.tmpdir_path / "chroma_store"
        store_uri = f"chroma:{store_dir}"
        collection_name = "test_incremental"

        try:
            builder = VSBuilder(
                embedding_model="ollama:bge-m3:latest",
                logger=self.logger,
            )
            builder.setup()
            self.logger.info("Builder set up with Ollama embeddings")

            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("Initial build completed")

            store = instantiate_vector_store(
                store_uri, collection_name, builder.embeddings, self.logger
            )
            first_count = len(store.get()["metadatas"])
            self.logger.info(f"Initial build: {first_count} chunks")

            md_two = self.tmpdir_path / "another.md"
            md_two.write_text(TEST_MD_TWO)
            self.logger.info(f"Added new test file: {md_two}")

            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("Incremental build completed")

            second_count = len(store.get()["metadatas"])
            self.logger.info(f"Incremental build: {second_count} chunks")

            assert second_count > first_count, (
                f"Expected more chunks after incremental build "
                f"({second_count} <= {first_count})"
            )
            self.logger.info(
                f"Incremental verified — added {second_count - first_count} more chunks"
            )

        except ResponseError as e:
            pytest.skip(str(e))
        except ConnectionError as e:
            pytest.skip(str(e))


if __name__ == "__main__":
    pytest.main()
