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
from klea_utils.stores import VectorStoresConfig
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
    env_var = "KLEA_RAG_CONFIG_FILE"
    env_file_default = "rag.env"
    config_class = AppConfig
    logger_name = "RAG"

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
        self.refusal_message += "\n- ".join([""] + self.stores.domains)
        self.refusal_message += "\n\n\nPlease try another query."

        # for clarification node
        self.clarification_message = "Apologies. I could not answer that question. Can you please ask another one or try to reword it and I will try again?"

    def _splitter_node(self, state: RAGState):
        return {}

    @override
    def _configure_resources(self):
        """Configure resources"""
        assert self.app_config
        general_settings = self.app_config.general.model_dump()
        general_settings["embedding_model"] = self.app_env.embedding_model
        domains = self.app_config.domains
        domain_vs = {}
        domain_ms = {}
        for d, inf in domains.items():
            domain_vs[d] = inf.model_dump(include={"vector_stores", "description"})

            domain_ms.update(inf.model_dump(include={"mcp_servers"})["mcp_servers"])

        self.logger.debug(f"{domain_vs = }")
        self.logger.debug(f"{domain_ms = }")

        self.stores_config = VectorStoresConfig(domains=domain_vs, **general_settings)
        self.mcp_config = MCPConfig(mcpServers=domain_ms)

    @override
    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(RAGState)

        # Guard nodes
        self._guard_node = GuardNode(
            logger=self.logger,
            model=self.g_model,
            temperature=0.3,
            memory=self.memory,
        )
        self.workflow.add_node("guard", self._guard_node.execute)

        self._guard_router_node = GuardRouterNode(logger=self.logger)

        self._decline_to_answer_node = FixedAnswer(
            logger=self.logger,
            state_attr="message_for_user",
            message="I cannot respond to this query. Please try another.",
        )
        self.workflow.add_node(
            "decline_to_answer", self._decline_to_answer_node.execute
        )

        self._init_rag_state_node = InitRAGState(logger=self.logger)
        self.workflow.add_node("init_rag_state", self._init_rag_state_node.execute)
        self._classify_question_node = ClassifyQuestion(
            logger=self.logger,
            model=self.c_model,
            output_schema=self.QueryDomainSchema,
            temperature=0.3,
            memory=self.memory,
            stores=self.stores,
        )
        self.workflow.add_node(
            "classify_question_domain", self._classify_question_node.execute
        )

        self._route_query_domain_node = RouteQuery(
            logger=self.logger,
            non_domain_chat=self.app_config.general.non_domain_chat,
        )

        self._generate_retrieval_query_node = GenerateRetrievalQuery(
            logger=self.logger, model=self.c_model, temperature=0.3
        )
        self.workflow.add_node(
            "generate_retrieval_query", self._generate_retrieval_query_node.execute
        )
        self._tools_picker_node = ToolsPicker(
            logger=self.logger,
            model=self.c_model,
            temperature=0.01,
            tools_description=self.tools_description,
        )
        self.workflow.add_node("tools_picker", self._tools_picker_node.execute)

        self._tools_caller_node = ToolsCaller(
            logger=self.logger,
            mcp_client=self.mcp_client,
        )
        self.workflow.add_node("tools_caller", self._tools_caller_node.execute)

        self._answer_general_node = AnswerGeneral(
            logger=self.logger,
            model=self.c_model,
            temperature=0.3,
            memory=self.memory,
            fallback_config=FallbackConfig(
                enabled=self.app_config.general.fallback_to_training_data,
                warning=self.app_config.general.fallback_warning,
            ),
        )
        self.workflow.add_node(
            "answer_general_question", self._answer_general_node.execute
        )

        self._refuse_answer_node = FixedAnswer(
            logger=self.logger,
            state_attr="message_for_user",
            message=self.refusal_message,
        )
        self.workflow.add_node("refuse_to_answer", self._refuse_answer_node.execute)

        self._retrieve_info_node = RetrieveInfoNode(
            logger=self.logger,
            stores=self.stores,
            num_refs_max=self.num_refs_max,
        )
        self.workflow.add_node("retrieve_info", self._retrieve_info_node.execute)
        self._generate_answer_from_context_node = AnswerFromContext(
            logger=self.logger,
            model=self.c_model,
            temperature=0.3,
            memory=False,
        )
        self.workflow.add_node(
            "generate_answer_from_context",
            self._generate_answer_from_context_node.execute,
        )
        self._evaluate_answer_node = Evaluator(
            logger=self.logger, model=self.c_model, temperature=0.0
        )
        self.workflow.add_node("evaluate_answer", self._evaluate_answer_node.execute)

        self._route_evaluator_node = RouteEvaluator(
            logger=self.logger, stores=self.stores
        )

        self._answer_user_node = AnswerUser(logger=self.logger)
        self.workflow.add_node(
            "give_domain_answer_to_user", self._answer_user_node.execute
        )

        self._ask_user_for_clarification_node = FixedAnswer(
            logger=self.logger,
            state_attr="message_for_user",
            message=self.clarification_message,
        )
        self.workflow.add_node(
            "ask_user_for_clarification", self._ask_user_for_clarification_node.execute
        )

        if self.memory:
            self._summarise_history_node = SummariseMemoryNode(
                logger=self.logger,
                model=self.c_model,
                temperature=0.3,
                summarisation_threshold=10,
            )
            self.workflow.add_node(
                "summarise_history", self._summarise_history_node.execute
            )

        self.workflow.add_edge(START, "init_rag_state")
        self.workflow.add_edge("init_rag_state", "guard")
        self.workflow.add_conditional_edges(
            "guard",
            self._guard_router_node.execute,
            {
                "safe": "classify_question_domain",
                "unsafe": "decline_to_answer",
            },
        )

        self.workflow.add_node("splitter", self._splitter_node)

        self.workflow.add_conditional_edges(
            "classify_question_domain",
            self._route_query_domain_node.execute,
            {
                "domain_query": "splitter",
                "non_domain_query": "answer_general_question",
                "non_domain_refuse": "refuse_to_answer",
            },
        )
        self.workflow.add_edge("splitter", "generate_retrieval_query")
        self.workflow.add_edge("splitter", "tools_picker")
        self.workflow.add_edge("tools_picker", "tools_caller")
        self.workflow.add_edge("generate_retrieval_query", "retrieve_info")
        self.workflow.add_edge("retrieve_info", "generate_answer_from_context")
        self.workflow.add_edge("tools_caller", "generate_answer_from_context")
        self.workflow.add_edge("generate_answer_from_context", "evaluate_answer")

        self.workflow.add_conditional_edges(
            "evaluate_answer",
            self._route_evaluator_node.execute,
            {
                "continue": "give_domain_answer_to_user",
                "retrieve_more_info": "retrieve_info",  # more info
                "rewrite_answer": "generate_answer_from_context",
                "modify_query": "generate_retrieval_query",  # new query
                "fallback": "answer_general_question",
                "undefined": "ask_user_for_clarification",
            },
        )

        if self.memory:
            self.workflow.add_edge("give_domain_answer_to_user", "summarise_history")
            self.workflow.add_edge("ask_user_for_clarification", "summarise_history")
            self.workflow.add_edge("answer_general_question", "summarise_history")
            self.workflow.add_edge("summarise_history", END)
        else:
            self.workflow.add_edge("give_domain_answer_to_user", END)
            self.workflow.add_edge("ask_user_for_clarification", END)
            self.workflow.add_edge("answer_general_question", END)

        self.workflow.add_edge("decline_to_answer", END)
        self.workflow.add_edge("refuse_to_answer", END)

        if self.checkpointer:
            self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        else:
            self.graph = self.workflow.compile()

        self._export_graph_png("rag-lang-graph.png")
