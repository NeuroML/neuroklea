#!/usr/bin/env python3
"""
Vector store ingestion -- convert documents, chunk, embed, and write to store

File: klea_utils/stores/ingestion.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
from pathlib import Path
from typing import Any

import xxhash
from langchain_core.documents import Document

from ..llm import setup_embedding
from .utils import instantiate_vector_store


class VSBuilder:
    """Build vector stores from a directory of source documents.

    Uses Docling for document conversion and token-aware chunking, then
    embeds chunks and writes them to a vector store backend.
    """

    DEFAULT_MAX_TOKENS = 450
    DEFAULT_MERGE_PEERS = True
    DEFAULT_TOKENIZER_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        embedding_model: str,
        logger: logging.Logger,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        merge_peers: bool = DEFAULT_MERGE_PEERS,
        tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    ):
        """Initialise the builder.

        :param embedding_model: Embedding model identifier
            (e.g. ``"ollama:bge-m3:latest"``)
        :param logger: Logger instance
        :param max_tokens: Maximum tokens per chunk
        :param merge_peers: Whether the chunker should merge peer
            elements (e.g. consecutive paragraphs)
        :param tokenizer_model: HuggingFace tokenizer model used for
            token-aware chunking
        """
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(f"{logger.name}.{self.__class__.__name__}")
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.tokenizer_model = tokenizer_model

        self.embeddings = None
        self._converter = None
        self._chunker = None

    def setup(self) -> None:
        """Initialise the embedding model and ensure chunker tokenizer is available."""
        self.embeddings = setup_embedding(self.embedding_model, self.logger)
        self._ensure_tokenizer()

    def build(
        self,
        source_dir: str,
        store_uri: str,
        collection_name: str,
        force: bool = False,
        metadata_map_path: str | None = None,
    ) -> None:
        """Convert documents in ``source_dir``, chunk, embed, and store.

        Files are hashed with xxhash to skip already-indexed content
        unless ``force`` is ``True``.

        When a metadata map is provided, each chunk's metadata is
        enriched by matching its most specific heading against map keys.
        The ``DEFAULT`` key provides a fallback when no heading matches.

        :param source_dir: Path to a directory containing source documents
        :param store_uri: Vector store URI (``chroma:/path``, etc.)
        :param collection_name: Collection name for the store
        :param force: Re-process all files even if they have not changed
        :param metadata_map_path: Optional path to a JSON file mapping
            heading text to dicts of metadata key-value pairs
        :raises FileNotFoundError: If ``source_dir`` or
            ``metadata_map_path`` does not exist
        :raises ValueError: If the metadata map is malformed
        """
        assert self.embeddings

        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        metadata_map = None
        if metadata_map_path:
            metadata_map = self._load_metadata_map(metadata_map_path)

        store = instantiate_vector_store(
            store_uri, collection_name, self.embeddings, self.logger, create=True
        )

        files = self._find_files(source_path)
        self.logger.info(f"Found {len(files)} ingestible files in {source_path}")

        for file_path in files:
            file_hash = _hash_file(file_path)

            if not force:
                existing = store.get(where={"file_hash": file_hash})
                if existing and existing["ids"]:
                    self.logger.debug(
                        f"Skipping already indexed file: {file_path.name}"
                    )
                    continue

            self.logger.info(f"Processing: {file_path.name}")
            try:
                docs = self._convert_and_chunk(file_path)

                for doc in docs:
                    doc.metadata.update(
                        {
                            "file_hash": file_hash,
                            "file_name": file_path.name,
                            "source_path": str(file_path),
                        }
                    )

                if metadata_map:
                    for doc in docs:
                        meta = self._resolve_metadata(
                            doc.metadata.get("headings"), metadata_map
                        )
                        if meta:
                            doc.metadata.update(meta)

                store.add_documents(docs)
                self.logger.info(f"Added {len(docs)} chunks from {file_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")

        self.logger.info(
            f"Ingestion complete for collection '{collection_name}' "
            f"({len(files)} files)"
        )

    # ------------------------------------------------------------------
    # Metadata map helpers
    # ------------------------------------------------------------------

    def _load_metadata_map(self, metadata_map_path: str) -> dict[str, dict[str, Any]]:
        """Load and validate a metadata map JSON file.

        The file must contain a JSON object with string keys (heading
        text) and dict values of metadata key-value pairs.  An optional
        ``DEFAULT`` key provides a fallback when no heading matches.

        :param metadata_map_path: Path to the JSON file
        :returns: Mapping of heading text to metadata dicts
        :raises FileNotFoundError: If the path does not exist
        :raises ValueError: If the JSON is not well-formed
        """
        path = Path(metadata_map_path)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata map file not found: {path}")
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Metadata map must be a JSON object (dict), got {type(data).__name__}"
            )
        for k, v in data.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"Metadata map keys must be strings, got {type(k).__name__}"
                )
            if not isinstance(v, dict):
                raise ValueError(
                    f"Values in metadata map must be dicts, "
                    f"got {type(v).__name__} for key {k!r}"
                )
        self.logger.info(f"Loaded metadata map with {len(data)} entries from {path}")
        return data

    def _resolve_metadata(
        self,
        headings: list[str] | None,
        metadata_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Resolve a metadata dict for a chunk by matching its heading chain.

        Iterates the heading chain from most specific to least specific
        and returns the first match.  Falls back to ``DEFAULT`` if no
        heading matches.

        :param headings: Heading hierarchy for the chunk (most specific
            last), or ``None``
        :param metadata_map: Mapping of heading text to metadata dicts
        :returns: Matched metadata dict, or ``None``
        """
        if headings:
            for heading in reversed(headings):
                if heading in metadata_map:
                    return metadata_map[heading]
        return metadata_map.get("DEFAULT")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_files(self, source_dir: Path) -> list[Path]:
        """Walk ``source_dir`` and return files whose extensions are in
        docling's :attr:`~docling.datamodel.base_models.FormatToExtensions`.

        Files with unsupported extensions are logged as a warning and skipped.

        :param source_dir: Directory to walk recursively
        :returns: Sorted list of files with supported extensions
        """
        from docling.datamodel.base_models import FormatToExtensions

        all_exts: set[str] = set()
        for exts in FormatToExtensions.values():
            all_exts.update(exts)

        supported: list[Path] = []
        for f in sorted(source_dir.rglob("*")):
            if not f.is_file():
                continue
            suffix = f.suffix.lstrip(".").lower()
            if suffix in all_exts:
                supported.append(f)
            else:
                self.logger.warning(f"Skipping unsupported file: {f.name}")

        return supported

    def _ensure_tokenizer(self) -> None:
        """Download the HuggingFace tokenizer used for token-aware chunking
        if it is not already cached locally.

        ..  TODO:: Allow overriding ``tokenizer_model`` via an environment
            variable (e.g. ``KLEA_INGEST_TOKENIZER_MODEL``) or a local
            filesystem path so that air-gapped deployments can point at
            pre-downloaded tokenizer files.
        """
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(self.tokenizer_model)

    def _get_converter(self):
        """Lazily initialise and return the Docling
        :class:`~docling.document_converter.DocumentConverter` singleton.

        :returns: Shared :class:`~docling.document_converter.DocumentConverter`
            instance
        """
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter

    def _get_chunker(self):
        """Lazily initialise and return the
        :class:`~docling.chunking.HybridChunker`
        configured with the instance tokenizer and chunking parameters.

        :returns: Configured :class:`~docling.chunking.HybridChunker` instance
        """
        if self._chunker is None:
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from transformers import AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
            tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer, max_tokens=self.max_tokens
            )
            self._chunker = HybridChunker(
                tokenizer=tokenizer, merge_peers=self.merge_peers
            )
        return self._chunker

    def _convert_and_chunk(self, file_path: Path) -> list[Document]:
        """Convert ``file_path`` with Docling, chunk with the
        :class:`~docling.chunking.HybridChunker`,
        and return :class:`~langchain_core.documents.Document` objects.

        Each document's metadata includes a ``headings`` list (the heading
        hierarchy for the chunk).

        :param file_path: Path to the source document file
        :returns: List of chunked :class:`~langchain_core.documents.Document`
            objects ready for embedding
        """
        converter = self._get_converter()
        chunker = self._get_chunker()

        result = converter.convert(str(file_path))
        dl_doc = result.document

        docs: list[Document] = []
        for chunk in chunker.chunk(dl_doc=dl_doc):
            chunk_text = chunker.contextualize(chunk=chunk)
            meta = chunk.meta.model_dump()

            doc = Document(
                page_content=chunk_text,
                metadata={"headings": meta.get("headings", [])},
            )
            docs.append(doc)

        return docs


def _hash_file(file_path: Path) -> str:
    """Return an xxhash hex digest of a file's contents.

    :param file_path: Path to the file to hash
    :returns: Hex digest string prefixed with ``"xxh64:"``
    """
    h = xxhash.xxh64()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"xxh64:{h.hexdigest()}"
