#!/usr/bin/env python3
"""
General RAG implementation

File: rag.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from textwrap import dedent
from typing import override

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.graph import BaseLangGraph
from neuroml_ai_utils.llm import setup_llm, split_output_by_section
from neuroml_ai_utils.nodes.answer_general import AnswerGeneral
from neuroml_ai_utils.nodes.fixed_answer import FixedAnswer
from neuroml_ai_utils.nodes.summarise_memory import SummariseMemoryNode
from neuroml_ai_utils.stores import serialize_reference

from .config import AppConfig
from .nodes.answer_user import AnswerUser
from .nodes.classify_question import ClassifyQuestion
from .nodes.evaluator import Evaluator
from .nodes.generate_retrieval_query import GenerateRetrievalQuery
from .nodes.init_rag import InitRAGState
from .nodes.retrieve_info import RetrieveInfoNode
from .nodes.route_evaluator import RouteEvaluator
from .nodes.route_query import RouteQuery
from .schemas import RAGState

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class RAG(BaseLangGraph):
    """General RAG implementation"""

    config_class = AppConfig
    config_env_var = "GEN_RAG_CONFIG_FILE"
    config_file_default = "rag.env"
    logger_name = "RAG"

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        memory: bool = True,
    ):
        """Initialise"""
        super().__init__(logging_level=logging_level, memory=memory)

        # total number of reference documents
        self.num_refs_max = 10

    @override
    def _setup_models(self) -> None:
        """Set up the LLM chat model"""
        self.c_model = setup_llm(self.config.chat_model, self.logger)

    async def get_graph(self):
        """Setup and get compiled graph"""
        await self.setup()
        return self.graph

    def _generate_answer_from_context_node(self, state: RAGState) -> dict:
        """Generate the answer"""
        assert self.c_model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        You are a query assistant. Your goal is to use the provided context to
        answer the user's query.

        If the context is missing some information to answer the question,
        say so.

        # Core Directives:

        - Limit yourself to facts from the provided context only, avoid using
          knowledge from your general training.
        - Use concise, formal language appropriate for neurosience and
          computational modelling.
        - Write the answer as a self contained explanation that does not assume
          access to the context.
        - Do not mention "context", "reference material", "documents" or
          "retrieval".
        - Do not include inline links.
        - Always include a section called "References" at the end of your answer.
            - In this section, list the reference URLs of the documents
              from the provided context that you used to generate the
              current answer.
            - Only include each reference URL ONCE in the list

        # Context (reference material not visible to the user, ordered from most relevant to least relevant):

        {reference_material}

        """)

        generate_answer_template = ChatPromptTemplate(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Question: {question}",
                ),
            ]
        )
        question = state.query
        reference_material = state.reference_material
        reference_material_text = serialize_reference(reference_material)

        prompt = generate_answer_template.invoke(
            {"question": question, "reference_material": reference_material_text}
        )
        self.logger.debug(f"{prompt =}")
        output = self.c_model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        thought, answer = split_output_by_section(output.content, "<think>", "</think>")

        # remove redundant references and format
        references = ""
        answer_text = ""
        ref_list = []
        # could use re. but using split function already, so keeping it simple
        for rf in ["\nreference", "\nReference", "\nReferences", "\nreferences"]:
            if rf in answer:
                answer_text, references = split_output_by_section(answer, rf)
                ref_list = list(set(references.split()))
                break

        if ref_list:
            answer_text += "\nReferences:" + "\n- ".join(ref_list)

        output.content = answer
        self.logger.debug(output.pretty_repr())

        messages = state.messages
        messages.append(output)

        return {"messages": messages, "reference_material": reference_material}

    @override
    async def _pre_graph(self):
        "Set up bits required before graph is compiled"
        # for refusal node
        self.refusal_message = "Sorry. I cannot answer this query as it does not fall into my permitted domains. Available domains are:\n"
        self.refusal_message += "\n- ".join([""] + self.stores.domains)
        self.refusal_message += "\n\n\nPlease try another query."

        # for clarification node
        self.clarification_message = "Apologies. I could not answer that question. Can you please ask another one or try to reword it and I will try again?"

    @override
    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(RAGState)
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
            stores=self.stores,
            non_domain_chat=self.config.non_domain_chat,
        )

        self._generate_retrieval_query_node = GenerateRetrievalQuery(
            logger=self.logger, model=self.c_model, temperature=0.3
        )
        self.workflow.add_node(
            "generate_retrieval_query", self._generate_retrieval_query_node.execute
        )
        self._answer_general_node = AnswerGeneral(
            logger=self.logger,
            model=self.c_model,
            temperature=0.3,
            memory=self.memory,
            fallback_config=self.stores.vs_config.fallback_to_training_data,
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
        self.workflow.add_node(
            "generate_answer_from_context", self._generate_answer_from_context_node
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
        self.workflow.add_edge("init_rag_state", "classify_question_domain")

        self.workflow.add_conditional_edges(
            "classify_question_domain",
            self._route_query_domain_node.execute,
            {
                "domain_query": "generate_retrieval_query",
                "non_domain_query": "answer_general_question",
                "non_domain_refuse": "refuse_to_answer",
            },
        )
        self.workflow.add_edge("generate_retrieval_query", "retrieve_info")
        self.workflow.add_edge("retrieve_info", "generate_answer_from_context")
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

        if self.checkpointer:
            self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        else:
            self.graph = self.workflow.compile()

        self._export_graph_png("rag-lang-graph.png")
