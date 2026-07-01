#!/usr/bin/env python3
"""
General RAG implementation

File: rag.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import final, override

from fastmcp.mcp_config import MCPConfig
from klea_utils.graph.base import BaseLangGraph
from klea_utils.llm import setup_llm
from klea_utils.nodes.answer_general import AnswerGeneral, FallbackConfig
from klea_utils.nodes.fixed_answer import FixedAnswer
from klea_utils.nodes.guard import GuardNode
from klea_utils.nodes.guard_router import GuardRouterNode
from klea_utils.nodes.summarise_memory import SummariseMemoryNode
from klea_utils.stores.config import VectorStoresConfig
from langgraph.graph import END, START, StateGraph

from .config import AppConfig, AppEnv
from .nodes.answer_from_context import AnswerFromContext
from .nodes.answer_user import AnswerUser
from .nodes.classify_question import ClassifyQuestion
from .nodes.evaluator import Evaluator
from .nodes.generate_retrieval_query import GenerateRetrievalQuery
from .nodes.init_rag import InitRAGState
from .nodes.retrieve_info import RetrieveInfoNode
from .nodes.route_evaluator import RouteEvaluator
from .nodes.route_query import RouteQuery
from .nodes.tools_caller import ToolsCaller
from .nodes.tools_picker import ToolsPicker
from .schemas import RAGState

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


@final
class RAG(BaseLangGraph):
    """General RAG implementation"""

    env_class = AppEnv
    env_var = "KLEA_RAG_ENV_FILE"
    env_file_default = "rag.env"
    config_class = AppConfig
    logger_name = "RAG"

    # type hints
    app_env: AppEnv
    app_config: AppConfig

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        memory: bool = True,
    ):
        """Initialise"""
        super().__init__(logging_level=logging_level, memory=memory)

        self.g_model = None

        # total number of reference documents
        self.num_refs_max = 10

    @override
    def _setup_models(self) -> None:
        """Set up the LLM chat model"""
        self.c_model = setup_llm(self.app_env.chat_model, self.logger)
        self.g_model = setup_llm(self.app_env.guard_model, self.logger)

    async def get_graph(self):
        """Setup and get compiled graph"""
        await self.setup()
        return self.graph

    @override
    async def _pre_graph(self):
        "Set up bits required before graph is compiled"
        # for refusal node
        self.refusal_message = "Sorry. I cannot answer this query as it does not fall into my permitted domains. Available domains are:\n"
        self.refusal_message += "\n- ".join([""] + list(self.app_config.domains))
        self.refusal_message += "\n\n\nPlease try another query."

        # for clarification node
        self.clarification_message = "Apologies. I could not answer that question. Can you please ask another one or try to reword it and I will try again?"

    def _splitter_node(self, state: RAGState):
        return {}

    @override
    def _configure_resources(self):
        """Configure resources"""
        assert self.app_config
        domains = self.app_config.domains
        domain_vs = {}
        domain_ms = {}
        for d, inf in domains.items():
            domain_vs[d] = inf.model_dump(include={"vector_stores", "description"})

            # flat config for mcp client initialization
            domain_ms.update(inf.model_dump(include={"mcp_servers"})["mcp_servers"])

        self.logger.debug(f"{domain_vs = }")
        self.logger.debug(f"{domain_ms = }")

        # set up configs
        self.stores_config = VectorStoresConfig(domains=domain_vs)
        self.embedding_model = self.app_env.embedding_model
        self.default_k = self.app_config.general.default_k
        self.k_max = self.app_config.general.k_max
        self.mcp_config = MCPConfig(mcpServers=domain_ms)

        # store per-domain MCP configs for domain-aware tool descriptions
        self.domain_mcp_configs = {}
        for d, inf in domains.items():
            domain_ms = inf.model_dump(include={"mcp_servers"}).get("mcp_servers", {})
            if domain_ms:
                self.domain_mcp_configs[d] = MCPConfig(mcpServers=domain_ms)

    @override
    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(RAGState)  # ty: ignore[invalid-assignment]

        # TODO: should be a check that gives user an error
        assert self.stores is not None or self.mcp_client is not None
        assert self.QueryDomainSchema is not None

        # Guard nodes
        self._guard_node = GuardNode(
            logger=self.logger,
            label="Checking safety",
            model=self.g_model,
            temperature=0.3,
            memory=self.memory,
        )
        self.workflow.add_node(self._guard_node.label, self._guard_node.execute)

        self._guard_router_node = GuardRouterNode(
            logger=self.logger, label="Routing safety"
        )

        self._decline_to_answer_node = FixedAnswer(
            logger=self.logger,
            label="Declining query",
            state_attr="message_for_user",
            message="I cannot respond to this query. Please try another.",
        )
        self.workflow.add_node(
            self._decline_to_answer_node.label, self._decline_to_answer_node.execute
        )

        self._init_rag_state_node = InitRAGState(
            logger=self.logger, label="Initializing"
        )
        self.workflow.add_node(
            self._init_rag_state_node.label, self._init_rag_state_node.execute
        )
        self._classify_question_node = ClassifyQuestion(
            logger=self.logger,
            label="Classifying question",
            model=self.c_model,
            output_schema=self.QueryDomainSchema,
            temperature=0.3,
            memory=self.memory,
            domains={
                d: info.description for d, info in self.app_config.domains.items()
            },
            pre_prompt=self.app_config.general.pre_prompt,
        )
        self.workflow.add_node(
            self._classify_question_node.label, self._classify_question_node.execute
        )

        self._route_query_domain_node = RouteQuery(
            logger=self.logger,
            label="Routing question",
            non_domain_chat=self.app_config.general.non_domain_chat,
        )

        self._generate_retrieval_query_node = GenerateRetrievalQuery(
            logger=self.logger,
            label="Generating search",
            model=self.c_model,
            temperature=0.3,
        )
        self.workflow.add_node(
            self._generate_retrieval_query_node.label,
            self._generate_retrieval_query_node.execute,
        )
        self._tools_picker_node = ToolsPicker(
            logger=self.logger,
            label="Selecting tools",
            model=self.c_model,
            temperature=0.01,
            domain_tools_description=self.tools_description,
        )
        self.workflow.add_node(
            self._tools_picker_node.label, self._tools_picker_node.execute
        )

        self._tools_caller_node = ToolsCaller(
            logger=self.logger,
            label="Running tools",
            mcp_client=self.mcp_client,
        )
        self.workflow.add_node(
            self._tools_caller_node.label, self._tools_caller_node.execute
        )

        self._answer_general_node = AnswerGeneral(
            logger=self.logger,
            label="Answering generally",
            model=self.c_model,
            temperature=0.3,
            memory=self.memory,
            fallback_config=FallbackConfig(
                enabled=self.app_config.general.fallback_to_training_data,
                warning=self.app_config.general.fallback_warning,
            ),
        )
        self.workflow.add_node(
            self._answer_general_node.label, self._answer_general_node.execute
        )

        self._refuse_answer_node = FixedAnswer(
            logger=self.logger,
            label="Refusing query",
            state_attr="message_for_user",
            message=self.refusal_message,
        )
        self.workflow.add_node(
            self._refuse_answer_node.label, self._refuse_answer_node.execute
        )

        self._retrieve_info_node = RetrieveInfoNode(
            logger=self.logger,
            label="Retrieving information",
            stores=self.stores,
            num_refs_max=self.num_refs_max,
        )
        self.workflow.add_node(
            self._retrieve_info_node.label, self._retrieve_info_node.execute
        )
        self._generate_answer_from_context_node = AnswerFromContext(
            logger=self.logger,
            label="Generating answer",
            model=self.c_model,
            temperature=0.3,
            memory=False,
        )
        self.workflow.add_node(
            self._generate_answer_from_context_node.label,
            self._generate_answer_from_context_node.execute,
        )
        self._evaluate_answer_node = Evaluator(
            logger=self.logger,
            label="Evaluating answer",
            model=self.c_model,
            temperature=0.0,
        )
        self.workflow.add_node(
            self._evaluate_answer_node.label, self._evaluate_answer_node.execute
        )

        self._route_evaluator_node = RouteEvaluator(
            logger=self.logger,
            label="Routing evaluation",
            stores=self.stores,
            max_retrieval_attempts=self.app_config.general.max_retrieval_attempts,
            max_rewrite_attempts=self.app_config.general.max_rewrite_attempts,
            fallback_to_training_data=self.app_config.general.fallback_to_training_data,
        )

        self._answer_user_node = AnswerUser(
            logger=self.logger, label="Preparing response"
        )
        self.workflow.add_node(
            self._answer_user_node.label, self._answer_user_node.execute
        )

        self._ask_user_for_clarification_node = FixedAnswer(
            logger=self.logger,
            label="Requesting clarification",
            state_attr="message_for_user",
            message=self.clarification_message,
        )
        self.workflow.add_node(
            self._ask_user_for_clarification_node.label,
            self._ask_user_for_clarification_node.execute,
        )

        if self.memory:
            self._summarise_history_node = SummariseMemoryNode(
                logger=self.logger,
                label="Summarizing history",
                model=self.c_model,
                temperature=0.3,
                summarisation_threshold=10,
            )
            self.workflow.add_node(
                self._summarise_history_node.label,
                self._summarise_history_node.execute,
            )

        self._splitter_label = "Splitting"

        self.workflow.add_edge(START, self._init_rag_state_node.label)
        self.workflow.add_edge(self._init_rag_state_node.label, self._guard_node.label)
        self.workflow.add_conditional_edges(
            self._guard_node.label,
            self._guard_router_node.execute,
            {
                "safe": self._classify_question_node.label,
                "unsafe": self._decline_to_answer_node.label,
            },
        )

        self.workflow.add_node(self._splitter_label, self._splitter_node)

        self.workflow.add_conditional_edges(
            self._classify_question_node.label,
            self._route_query_domain_node.execute,
            {
                "domain_query": self._splitter_label,
                "non_domain_query": self._answer_general_node.label,
                "non_domain_refuse": self._refuse_answer_node.label,
            },
        )
        self.workflow.add_edge(
            self._splitter_label, self._generate_retrieval_query_node.label
        )
        self.workflow.add_edge(self._splitter_label, self._tools_picker_node.label)
        self.workflow.add_edge(
            self._tools_picker_node.label, self._tools_caller_node.label
        )
        self.workflow.add_edge(
            self._generate_retrieval_query_node.label,
            self._retrieve_info_node.label,
        )
        self.workflow.add_edge(
            self._retrieve_info_node.label,
            self._generate_answer_from_context_node.label,
        )
        self.workflow.add_edge(
            self._tools_caller_node.label,
            self._generate_answer_from_context_node.label,
        )
        self.workflow.add_edge(
            self._generate_answer_from_context_node.label,
            self._evaluate_answer_node.label,
        )

        self.workflow.add_conditional_edges(
            self._evaluate_answer_node.label,
            self._route_evaluator_node.execute,
            {
                "continue": self._answer_user_node.label,
                "retrieve_more_info": self._retrieve_info_node.label,
                "rewrite_answer": self._generate_answer_from_context_node.label,
                "modify_query": self._generate_retrieval_query_node.label,
                "fallback": self._answer_general_node.label,
                "undefined": self._ask_user_for_clarification_node.label,
            },
        )

        if self.memory:
            self.workflow.add_edge(
                self._answer_user_node.label,
                self._summarise_history_node.label,
            )
            self.workflow.add_edge(
                self._ask_user_for_clarification_node.label,
                self._summarise_history_node.label,
            )
            self.workflow.add_edge(
                self._answer_general_node.label,
                self._summarise_history_node.label,
            )
            self.workflow.add_edge(self._summarise_history_node.label, END)
        else:
            self.workflow.add_edge(self._answer_user_node.label, END)
            self.workflow.add_edge(self._ask_user_for_clarification_node.label, END)
            self.workflow.add_edge(self._answer_general_node.label, END)

        self.workflow.add_edge(self._decline_to_answer_node.label, END)
        self.workflow.add_edge(self._refuse_answer_node.label, END)

        if self.checkpointer:
            self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        else:
            self.graph = self.workflow.compile()

        self._export_graph_png("rag-lang-graph.png")
