#!/usr/bin/env python3
"""
Vector store ingestion -- convert documents, chunk, embed, and write to store

File: klea_utils/stores/ingestion.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import xxhash
from langchain_core.documents import Document

from ..llm import setup_embedding
from .utils import instantiate_vector_store

CACHE_DIR_NAME = ".klea-cache"
TEMPLATE_FILE_NAME = "metadata-map.template.json"


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
            (e.g. ``"ollama:bge-m3:latest"``).  Only needed when
            :meth:`store_all` will be called.
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

    def build(
        self,
        source_dir: str,
        store_uri: str,
        collection_name: str,
        force: bool = False,
        metadata_map_path: str | None = None,
    ) -> None:
        """Full pipeline: chunk documents and write them to a vector store.

        Convenience wrapper around :meth:`chunk_all` + :meth:`store_all`.

        :param source_dir: Path to a directory containing source documents
        :param store_uri: Vector store URI (e.g. ``chroma:/path``)
        :param collection_name: Collection name for the store
        :param force: Re-process all files even if unchanged
        :param metadata_map_path: Optional path to a metadata map JSON file
        """
        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        metadata_map = None
        if metadata_map_path:
            metadata_map = self._load_metadata_map(metadata_map_path)

        results, _ = self.chunk_all(source_path, metadata_map, force)
        self.store_all(results, store_uri, collection_name, force)
        self.logger.info(f"Ingestion complete for collection '{collection_name}'")

    def chunk_all(
        self,
        source_path: Path,
        metadata_map: dict[str, dict[str, Any]] | None = None,
        force: bool = False,
    ) -> tuple[list[tuple[str, list[Document], Path]], set[str]]:
        """Convert, chunk, cache, and enrich metadata for all files.

        Skips converting files whose cache entry exists (unless
        ``force`` is ``True``).  Always caches newly-converted chunks.
        Heading chains are collected across all files regardless of mode.

        :param source_path: Resolved source directory path
        :param metadata_map: Metadata map for heading-based enrichment,
            or ``None``
        :param force: Re-process all files even if cached
        :returns: ``(results, all_heading_chains)`` where *results* is a
            list of ``(file_hash, docs, file_path)`` tuples and
            *all_heading_chains* is a set of unique heading chains
        """
        self._ensure_tokenizer()

        files = self._find_files(source_path)
        self.logger.info(f"Found {len(files)} ingestible files in {source_path}")

        results: list[tuple[str, list[Document], Path]] = []
        all_heading_chains: set[str] = set()
        total = len(files)

        for ctr, file_path in enumerate(files, 1):
            file_hash = _hash_file(file_path)

            docs = None
            if not force:
                docs = self._load_from_cache(source_path, file_hash)

            if docs is None:
                self.logger.info(f"Processing: {file_path.name} ({ctr}/{total})")
                try:
                    docs = self._convert_and_chunk(file_path)
                    self._save_to_cache(docs, source_path, file_hash)
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path.name}: {e}")
                    continue
            else:
                self.logger.debug(
                    f"Using cached chunks for: {file_path.name} ({ctr}/{total})"
                )

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

            for doc in docs:
                headings = doc.metadata.get("headings", [])
                if headings:
                    all_heading_chains.add(" > ".join(headings))

            results.append((file_hash, docs, file_path))

        return results, all_heading_chains

    def store_all(
        self,
        results: list[tuple[str, list[Document], Path]],
        store_uri: str,
        collection_name: str,
        force: bool = False,
    ) -> None:
        """Write chunked documents to a vector store.

        Initialises the embedding model on first call if not already
        done.  Skips files whose hash is already present in the store
        (unless ``force`` is ``True``).

        :param results: List of ``(file_hash, docs, file_path)`` tuples
            from :meth:`chunk_all`
        :param store_uri: Vector store URI
        :param collection_name: Collection name for the store
        :param force: Re-store all files even if already indexed
        """
        if self.embeddings is None:
            self.embeddings = setup_embedding(self.embedding_model, self.logger)
        assert store_uri and collection_name

        store = instantiate_vector_store(
            store_uri,
            collection_name,
            self.embeddings,
            self.logger,
            create=True,
        )

        for file_hash, docs, file_path in results:
            if not force:
                existing = store.get(where={"file_hash": file_hash})
                if existing and existing["ids"]:
                    self.logger.debug(
                        f"Skipping already indexed file: {file_path.name}"
                    )
                    continue

            store.add_documents(docs)
            self.logger.info(f"Added {len(docs)} chunks from {file_path.name}")

    def write_heading_template(
        self, heading_chains: set[str], source_dir: Path
    ) -> None:
        """Write a metadata-map template JSON file with empty placeholder
        dicts for every unique heading chain found across all files.

        The user fills in the ``{}`` with their metadata key-value pairs
        and passes the file to ``klea-vs-create store --metadata-map``.

        :param heading_chains: Unique heading chains collected across all
            processed files
        :param source_dir: Resolved source directory path (template is
            written alongside it)
        """
        template: dict[str, dict[str, Any]] = {"DEFAULT": {}}
        for chain in sorted(heading_chains):
            template[chain] = {}

        out_path = source_dir / TEMPLATE_FILE_NAME
        with open(out_path, "w") as f:
            json.dump(template, f, indent=4)
            f.write("\n")
        self.logger.info(
            f"Metadata map template written to {out_path} "
            f"({len(template) - 1} heading chains)"
        )

    def _cache_dir(self, source_dir: Path) -> Path:
        """Return the cache directory path inside *source_dir*.

        :param source_dir: Resolved source directory path
        :returns: Path to ``<source_dir>/.klea-cache/``
        """
        return source_dir / CACHE_DIR_NAME

    def _cache_path(self, source_dir: Path, file_hash: str) -> Path:
        """Return the cache file path for a given file hash.

        The ``:`` in the hash is replaced with ``_`` for filesystem
        safety (``:`` is allowed in most Linux filesystems but is
        problematic on Windows and some networked FSes).

        :param source_dir: Resolved source directory path
        :param file_hash: xxhash digest of the source file
        :returns: Path to ``<cache_dir>/<file_hash>.pkl``
        """
        safe_hash = file_hash.replace(":", "_")
        return self._cache_dir(source_dir) / f"{safe_hash}.pkl"

    def _save_to_cache(
        self, docs: list[Document], source_dir: Path, file_hash: str
    ) -> None:
        """Pickle *docs* to the cache directory.

        Creates the cache directory if it does not exist.

        :param docs: List of chunked documents to cache
        :param source_dir: Resolved source directory path
        :param file_hash: xxhash digest of the source file
        """
        cache_dir = self._cache_dir(source_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(source_dir, file_hash)
        with open(path, "wb") as f:
            pickle.dump(docs, f)
        self.logger.debug(f"Cached {len(docs)} chunks to {path}")

    def _load_from_cache(
        self, source_dir: Path, file_hash: str
    ) -> list[Document] | None:
        """Load pickled chunks from the cache.

        To inspect cached chunks from a Python shell::

            import pickle
            from pathlib import Path
            for p in Path("<source_dir>/.klea-cache/").glob("*.pkl"):
                docs = pickle.load(open(p, "rb"))
                print(p.stem, docs[0].metadata.get("headings"))

        :param source_dir: Resolved source directory path
        :param file_hash: xxhash digest of the source file
        :returns: List of :class:`~langchain_core.documents.Document`, or
            ``None`` if the cache file does not exist
        """
        path = self._cache_path(source_dir, file_hash)
        if not path.is_file():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

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
