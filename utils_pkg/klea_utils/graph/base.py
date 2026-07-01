#!/usr/bin/env python3
"""
Base class for LangGraph-based orchestrators

File: klea_utils/graph/base.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from textwrap import dedent
from typing import Any, List, Literal, Type, final

from fastmcp import Client
from fastmcp.mcp_config import MCPConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import StreamTransformer
from langgraph.types import RunnableConfig
from mcp.types import Tool
from pydantic import BaseModel, create_model

from klea_utils.stores.config import VectorStoresConfig
from klea_utils.stores.retrieval import VSRetriever


class _CustomChannelEnabler(StreamTransformer):
    """Enables the ``custom`` channel in LangGraph v3 event streams.

    Non-LLM nodes use ``get_stream_writer()`` to emit progress events on the
    custom channel.  LangGraph requires a ``StreamTransformer`` declaring
    ``required_stream_modes = ("custom",)`` for that channel to be enabled.
    This no-op transformer satisfies that requirement.
    """

    required_stream_modes = ("custom",)

    def __init__(self, scope=()):
        super().__init__(scope)

    def init(self):
        return {}

    def process(self, event):
        return True


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
    - Set ``env_class`` to the appropriate Pydantic settings class
    """

    #: Pydantic BaseSettings class for env loading.
    #: Subclasses must set this to their AppEnv class.
    env_class: Type[BaseModel]

    #: Pydantic BaseModel class for configuration loading.
    #: Subclasses must set this to their AppConfig class.
    config_class: Type[BaseModel]

    #: Name of the environment variable that controls the env file path.
    env_var: str = "ENV_FILE"

    #: Default config file name if the environment variable is not set.
    env_file_default: str = "config.env"

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
        self.env_file = os.getenv(self.env_var, self.env_file_default)
        self.app_env: BaseModel

        self.c_model = None

        self.memory = memory

        self.tools_description: dict[str, str] = {}
        self.domain_mcp_configs: dict[str, MCPConfig] = {}
        self.checkpointer: InMemorySaver | None = InMemorySaver() if memory else None

        self.config_dict: dict[str, Any]

        self.graph: CompiledStateGraph | None = None

        self.mcp_config: MCPConfig | None = None
        self.mcp_client: Client | None = None
        self.mcp_tools: list[Tool] | None = None

        self.stores_config: VectorStoresConfig | None = None
        self.stores: VSRetriever | None = None
        self.embedding_model: str | None = None
        self.default_k: int = 5
        self.k_max: int = 10

        self.QueryDomainSchema: Type[BaseModel] | None = None

        from klea_utils.plogging import setup_logger

        self.logger = setup_logger(self.logger_name, stderr_level=logging_level)

    def _load_env(self) -> None:
        """Load env file, and configuration

        Uses ``self.env_class`` and ``self.env_file`` to locate and parse
        the configuration file. Raises FileNotFoundError if the file does not exist.
        """
        env_file_path = Path(self.env_file)
        if not env_file_path.exists():
            raise FileNotFoundError(
                f"""Could not find env file: {self.env_file}. You can use the {self.env_var} environment variable to specify the env file."""
            )

        self.app_env = self.env_class(_env_file=self.env_file)
        assert self.app_env
        self.logger.debug(f"env file: {self.env_file}")
        self.logger.debug(f"env: {self.app_env}")

        if "app_config_file" in self.env_class.model_fields:
            config_file = Path(self.app_env.app_config_file)
            if not config_file.exists():
                raise FileNotFoundError(f"Could not find config file: {config_file}")
            else:
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                    self.logger.debug(f"{config_dict = }")
                    self.app_config = self.config_class(**config_dict)
                    self.logger.debug(f"{self.app_config = }")
        else:
            raise FileNotFoundError(
                f"No config file provided. Please provide one in the env file ({self.env_file})."
                + f"You can use the {self.env_var} environment variable to specify the env file."
            )

    def _create_mcp_client(self) -> None:
        """Create MCP client from the JSON config file.

        Reads the MCP server configurations from ``self.app_env.mcp_config_file``
        and creates a ``fastmcp.Client`` instance.
        """
        if self.mcp_config and self.mcp_config.mcpServers:
            self.logger.debug(f"{self.mcp_config = }")
            self.mcp_client = Client(self.mcp_config)
        else:
            self.logger.warning("No MCP server configured.")
            self.mcp_client = None

    async def _get_mcp_tools(self) -> None:
        """Get MCP tools."""
        if self.mcp_client:
            async with self.mcp_client:
                self.mcp_tools = await self.mcp_client.list_tools()
            self.logger.debug(f"{self.mcp_tools =}")
            self._build_tools_description()

    def _build_tools_description(self) -> None:
        """Build per-domain tool descriptions from fetched MCP tools."""
        self.tools_description = {}
        if not self.mcp_tools or not self.domain_mcp_configs:
            return

        # map server names to domains
        domain_servers: dict[str, list[str]] = {}
        num_servers = 0
        for domain, config in self.domain_mcp_configs.items():
            if config.mcpServers:
                domain_servers[domain] = list(config.mcpServers.keys())
                num_servers += len(list(config.mcpServers.keys()))

        for domain, server_names in domain_servers.items():
            desc = ""
            ctr = 0
            for t in self.mcp_tools:
                if "dummy" in t.name:
                    continue
                # tools will be prefixed with server names
                if num_servers > 1:
                    if not any(t.name.startswith(s + "_") for s in server_names):
                        continue
                # otherwise, there's only one server
                ctr += 1
                desc += dedent(f"""
                    ## {ctr}.  {t.name}

                    ### Description

                    {t.description}

                    """)
                if t.inputSchema:
                    desc += dedent(f"""
                        ### Parameters

                        {t.inputSchema.get("properties")}

                        """)
            self.tools_description[domain] = desc

    async def _get_vector_stores(self) -> None:
        """Get vector stores"""
        if self.stores_config and self.embedding_model:
            self.stores = VSRetriever(
                vs_config=self.stores_config,
                logger=self.logger,
                embedding_model=self.embedding_model,
                default_k=self.default_k,
                k_max=self.k_max,
            )
            self.stores.setup()
            self.logger.info(f"Vector stores loaded: {self.stores.domains}")

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

    # ------------------------------------------------------------------
    # Abstract methods -- subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _configure_resources(self) -> None:
        """Configure vector stores and MCP servers

        Subclasses should implement this to populate ``self.stores_config``,
        ``self.mcp_config``, and ``self.domain_mcp_configs``, which will be used
        to create the vector store class, mcp client, and per-domain tool descriptions.
        """
        ...

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
        2. ``_load_env()``
        3. ``_setup_models()``
        4. ``_create_mcp_client()``
        5. ``_pre_graph()``
        6. ``_create_graph()``
        7. ``_post_setup()``
        """
        self._pre_setup()
        self._load_env()
        self._configure_resources()
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
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

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
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

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
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

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
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        res = await self.graph.astream({"query": query}, config=config)
        return res

    async def run_graph_astream_events(
        self, query: str, thread_id: str = "default_thread"
    ):
        """Run the graph and yield structured streaming events.

        Yields dicts with:
        - ``{"type": "progress", "node": "<label>"}`` -- when the graph enters
          a new node (LLM or custom-event non-LLM)
        - ``{"type": "token", "content": "<chunk>", "node": "<label>"}`` -- LLM
          token chunk from the current node
        - ``{"type": "complete", "message_for_user": "<answer>"}`` -- final
          answer from the completed graph

        Uses LangGraph's ``astream_events`` v3 protocol.  LLM output is read
        from the ``messages`` channel; non-LLM nodes emit ``custom`` events
        via ``get_stream_writer()``.  A ``StreamTransformer`` enables the
        custom channel so those events flow through.

        Reference: https://docs.langchain.com/oss/python/langgraph/event-streaming

        :param query: User query string
        :param thread_id: Session/thread identifier for checkpointing
        :yields: Structured event dicts
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        assert self.graph, "Graph not compiled. Call setup() first."

        stream = await self.graph.astream_events(
            {"query": query},
            config=config,
            version="v3",
            transformers=[_CustomChannelEnabler],
        )

        current_node = ""
        last_values: dict = {}

        async for event in stream:
            method = event["method"]

            if method == "custom":
                data = event["params"]["data"]
                if (
                    isinstance(data, dict)
                    and data.get("type") == "progress"
                    and data.get("node")
                ):
                    node = data["node"]
                    if node != current_node:
                        current_node = node
                        self.logger.debug(f"Progress: {current_node}")
                        yield {"type": "progress", "node": current_node}

            elif method == "messages":
                data = event["params"]["data"]
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    node = item.get("langgraph_node", "")
                    if node and node != current_node:
                        current_node = node
                        self.logger.debug(f"Progress: {current_node}")
                        yield {"type": "progress", "node": current_node}

                    if item.get("event") == "content-block-delta":
                        delta = item.get("delta", {})
                        if "text" in delta:
                            yield {
                                "type": "token",
                                "content": delta["text"],
                                "node": current_node,
                            }

            elif method == "values":
                last_values = event["params"]["data"]

        message = (last_values or {}).get("message_for_user", "")
        yield {"type": "complete", "message_for_user": message}
