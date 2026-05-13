#!/usr/bin/env python3
"""
Base class for LangGraph-based orchestrators

File: neuroml_ai_utils/graph/base.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from textwrap import dedent
from typing import Any, List, Literal, Type, final

from fastmcp import Client
from fastmcp.mcp_config import MCPConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from mcp.types import Tool
from pydantic import BaseModel, create_model

from neuroml_ai_utils.stores import VectorStores


class BaseLangGraph(ABC):
    """Abstract base class for LangGraph-based orchestrators.

    Provides common infrastructure for:
    - Configuration loading from env files
    - MCP client creation from JSON config
    - LLM model setup (delegated to subclasses)
    - LangGraph compilation and execution
    - Session checkpointing
    - Dual-stream logging

    Subclasses must implement:
    - :meth:`_setup_models`: Create LLM model instances
    - :meth:`_create_graph`: Build and compile the LangGraph
    - Set ``config_class`` to the appropriate Pydantic settings class
    """

    #: Pydantic BaseSettings class for configuration loading.
    #: Subclasses must set this to their AppConfig class.
    config_class: Type[BaseModel]

    #: Name of the environment variable that controls the config file path.
    config_env_var: str = "CONFIG_FILE"

    #: Default config file name if the environment variable is not set.
    config_file_default: str = "config.env"

    #: Logger name for this orchestrator.
    logger_name: str = "BaseLangGraph"

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        memory: bool = True,
    ):
        """Initialise the base orchestrator.

        :param logging_level: Logging level for the orchestrator
        :param memory: Whether to enable checkpoint-based session memory
        """
        self.c_model = None
        self.mcp_client: Client | None = None

        self.memory = memory
        self.checkpointer: InMemorySaver | None = InMemorySaver() if memory else None

        self.config_file = os.getenv(self.config_env_var, self.config_file_default)
        self.config: BaseModel

        self.graph: CompiledStateGraph | None = None

        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        self.mcp_tools: list[Tool] | None = None
        self.stores: VectorStores | None = None
        self.QueryDomainSchema: Type[BaseModel] | None = None

        self._setup_logging(logging_level)

    def _setup_logging(self, level: int) -> None:
        """Set up dual-stream logging (INFO to stdout, rest to stderr).

        :param level: Logging level for stderr handler
        """
        from neuroml_ai_utils.logging import (
            LoggerInfoFilter,
            LoggerNotInfoFilter,
            logger_formatter_info,
            logger_formatter_other,
        )

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(logger_formatter_info)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(logger_formatter_other)
        self.logger.addHandler(stderr_handler)

    def _load_config(self) -> None:
        """Load configuration from the env file.

        Uses ``self.config_class`` and ``self.config_file`` to locate and parse
        the configuration file. Raises FileNotFoundError if the file does not exist.
        """
        cfg_path = Path(self.config_file)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Could not find config file: {self.config_file}")

        self.config = self.config_class(_env_file=self.config_file)
        assert self.config
        self.logger.debug(f"Config file: {self.config_file}")
        self.logger.debug(f"Config: {self.config}")

    def _create_mcp_client(self) -> None:
        """Create MCP client from the JSON config file.

        Reads the MCP server configurations from ``self.config.mcp_config_file``
        and creates a ``fastmcp.Client`` instance.
        """
        if self.config.mcp_config_file:  # type: ignore
            mcp_config_text = ""
            with open(self.config.mcp_config_file, "r") as f:
                mcp_config_text = json.load(f)
                self.logger.debug(f"{mcp_config_text = }")

            self.mcp_config = MCPConfig(**mcp_config_text)
            self.logger.debug(f"{self.mcp_config = }")
            self.mcp_client = Client(self.mcp_config)
            assert self.mcp_client
        else:
            self.logger.warning("No MCP server configured.")
            self.mcp_client = None

    async def _get_mcp_tools(self) -> None:
        """List MCP tools and optionally set up vector stores."""
        if self.mcp_client:
            async with self.mcp_client:
                self.mcp_tools = await self.mcp_client.list_tools()
            self.logger.debug(f"{self.mcp_tools =}")

    async def _get_vector_stores(self) -> None:
        """Get vector stores"""
        if self.config.vs_config_file:  # type: ignore
            self.stores = VectorStores(
                vs_config_file=self.config.vs_config_file, logger=self.logger
            )
            self.stores.setup()
            self.logger.info(
                f"Vector stores loaded from {self.config.vs_config_file}: {self.stores.domains}"
            )

            # dynamically generate schema for domains
            all_domains = self.stores.domains.copy()
            all_domains.append("undefined")

            self.QueryDomainSchema = create_model(
                "QueryDomainSchema",
                query_domains=(List[Literal[tuple(all_domains)]], "undefined"),
            )
        else:
            self.logger.warning("No vector stores configured.")

    def _export_graph_png(self, filename: str) -> None:
        """Export the LangGraph as a Mermaid PNG diagram.

        Skipped when running inside Docker (``RUNNING_IN_DOCKER`` env var set).

        :param filename: Output file path for the PNG
        """
        if os.environ.get("RUNNING_IN_DOCKER", 0):
            return
        try:
            assert self.graph
            self.graph.get_graph().draw_mermaid_png(output_file_path=filename)
        except BaseException as e:
            self.logger.error("Something went wrong generating lang graph png")
            self.logger.error(e)

    @cached_property
    def tools_description(self) -> str:
        """Get formatted descriptions of available MCP tools.

        :returns: Formatted string with tool names, descriptions, and parameters
        """
        if not self.mcp_tools:
            return ""
        description = ""
        ctr = 0
        for t in self.mcp_tools:
            if "dummy" in t.name:
                continue
            ctr += 1
            description += dedent(
                f"""
                ## {ctr}.  {t.name}

                ### Description

                {t.description}

                """
            )
            if t.inputSchema:
                description += dedent(
                    f"""
                    ### Parameters

                    {t.inputSchema.get("properties")}

                    """
                )
        return description

    # ------------------------------------------------------------------
    # Abstract methods -- subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup_models(self) -> None:
        """Set up LLM model instances.

        Subclasses should assign model instances to ``self.c_model`` (and
        optionally ``self.r_model`` for reasoning models).
        """
        ...

    @abstractmethod
    async def _create_graph(self) -> None:
        """Build and compile the LangGraph, storing it in ``self.graph``.

        This is where subclasses define their nodes, edges, and conditional
        routing logic.
        """
        ...

    # ------------------------------------------------------------------
    # Hook methods -- override for pre/post setup work
    # ------------------------------------------------------------------

    def _pre_setup(self) -> None:
        """Hook called before the standard setup sequence.

        Override to perform subclass-specific initialisation before
        config loading and model setup.
        """
        pass

    def _post_setup(self) -> None:
        """Hook called after the standard setup sequence.

        Override to perform subclass-specific finalisation after
        the LangGraph has been compiled.
        """
        pass

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    async def _pre_graph(self) -> None:
        """Hook called after MCP client setup but before graph compilation.

        Override to perform subclass-specific initialisation that depends
        on config and MCP client but must happen before the LangGraph is built.
        """
        pass

    @final
    async def setup(self) -> None:
        """Set up the orchestrator.

        Calls hooks and template methods in this order:
        1. ``_pre_setup()``
        2. ``_load_config()``
        3. ``_setup_models()``
        4. ``_create_mcp_client()``
        5. ``_pre_graph()``
        6. ``_create_graph()``
        7. ``_post_setup()``
        """
        self._pre_setup()
        self._load_config()
        self._setup_models()
        self._create_mcp_client()
        await self._get_mcp_tools()
        await self._get_vector_stores()
        await self._pre_graph()
        await self._create_graph()
        self._post_setup()

    # ------------------------------------------------------------------
    # Execution methods -- identical across all implementations
    # ------------------------------------------------------------------

    async def run_graph_invoke_state(
        self, state: dict, thread_id: str = "default_thread"
    ) -> dict:
        """Run the graph, accepting and returning full state dicts.

        :param state: Initial graph state (must contain ``query`` key)
        :param thread_id: Session/thread identifier for checkpointing
        :returns: Final graph state
        """
        config = {"configurable": {"thread_id": thread_id}}

        if "query" not in state:
            self.logger.error(f"Provided state should include the key 'query': {state}")
            sys.exit(-1)

        final_state = await self.graph.ainvoke(state, config=config)
        self.logger.debug(final_state)
        return final_state

    # TODO: fields to be extracted from the final state to be returned should
    # be configurable with a schema
    async def run_graph_invoke(
        self, query: str, thread_id: str = "default_thread"
    ) -> str:
        """Run the graph with a simple string query.

        :param query: User query string
        :param thread_id: Session/thread identifier for checkpointing
        :returns: The ``message_for_user`` field from the final state
        """
        config = {"configurable": {"thread_id": thread_id}}

        final_state = await self.graph.ainvoke({"query": query}, config=config)

        self.logger.debug(f"{final_state =}")
        if message := final_state.get("message_for_user", None):
            return message
        else:
            return "I was unable to answer"

    async def run_graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph and yield intermediate ``message_for_user`` values.

        :param query: User query string
        :param thread_id: Session/thread identifier for checkpointing
        :yields: ``message_for_user`` strings from each node
        """
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.graph.astream({"query": query}, config=config):
            for node, state in chunk.items():
                self.logger.debug(f"{node}: {repr(state)}")
                if message := state.get("message_for_user", None):
                    self.logger.info(f"User message: {message}")
                    yield message
                else:
                    self.logger.debug(f"Working in node: {node}")

    async def graph_stream(self, query: str, thread_id: str = "default_thread") -> Any:
        """Run the graph and return the raw astream result.

        :param query: User query string
        :param thread_id: Session/thread identifier for checkpointing
        :returns: Raw async generator from ``graph.astream()``
        """
        config = {"configurable": {"thread_id": thread_id}}

        res = await self.graph.astream({"query": query}, config=config)
        return res
