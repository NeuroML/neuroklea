#!/usr/bin/env python3
"""
General RAG implementation

File: rag.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from textwrap import dedent

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.graph import BaseLangGraph
from neuroml_ai_utils.llm import (
    setup_llm,
    split_output_by_section,
)
from neuroml_ai_utils.nodes.answer_general import AnswerGeneral
from neuroml_ai_utils.nodes.summarise_memory import SummariseMemoryNode
from neuroml_ai_utils.stores import serialize_reference

from .config import AppConfig
from .nodes.answer_user import AnswerUser
from .nodes.classify_question import ClassifyQuestion
from .nodes.evaluator import Evaluator
from .nodes.generate_retrieval_query import GenerateRetrievalQuery
from .nodes.init_rag import InitRAGState
from .nodes.refuse_answer import RefuseAnswer
from .nodes.retrieve_info import RetrieveInfoNode
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

    def _route_evaluator_node(self, state: RAGState) -> str:
        """Route depending on evaluation of answer"""
        self.logger.debug(f"{state =}")
        resp = state.text_response_eval
        next_step = resp.next_step

        if next_step == "continue" and (
            resp.coverage >= 0.5
            and resp.confidence >= 0.5
            and resp.relevance >= 0.5
            and resp.groundedness >= 0.5
            and resp.coherence >= 0.5
            and resp.conciseness >= 0.5
        ):
            self.stores.reset_k()
            self.logger.debug("returning: continue")
            return "continue"
        elif not state.query_modified and (
            next_step == "modify_query" or resp.coverage < 0.3
        ):
            self.logger.debug("returning: modify_query")
            return "modify_query"
        elif next_step == "retrieve_more_info" or (
            resp.coverage >= 0.5 and resp.confidence < 0.5
        ):
            # limit what max k we can have, otherwise, we end up pulling the
            # whole store..
            if self.stores.inc_k():
                self.logger.debug("returning: retrieve_more_info")
                return "retrieve_more_info"
            else:
                # we are already at max context, so we need to modify the query
                # to get a better result if possible
                if not state.query_modified:
                    self.logger.debug("returning: modify_query")
                    return "modify_query"
                # if we've already modified query, fallback to training data if
                # possible, otherwise ask for clarification
                else:
                    if self.stores.vs_config.fallback_to_training_data:
                        self.logger.debug("returning: fallback")
                        return "fallback"
                    else:
                        self.logger.debug("returning: undefined")
                        return "undefined"
        elif next_step == "rewrite_answer" or (
            resp.coverage >= 0.5
            and resp.confidence >= 0.5
            and (
                resp.relevance < 0.5
                and resp.groundedness < 0.5
                and resp.coherence < 0.5
                and resp.conciseness < 0.5
            )
        ):
            self.logger.debug("returning: rewrite_answer")
            return "rewrite_answer"
        # all other cases: fallback to training data if enabled, otherwise ask for clarification
        else:
            if self.stores.vs_config.fallback_to_training_data:
                self.logger.debug("returning: fallback")
                return "fallback"
            else:
                self.logger.debug("returning: undefined")
                return "undefined"

    def _route_query_domain_node(self, state: RAGState) -> str:
        """Route the query depending on LLM's result"""
        self.logger.debug(f"{state =}")
        query_domain = state.query_domain

        if query_domain in self.stores.domains and query_domain != "undefined":
            return "domain_query"
        else:
            if self.config.non_domain_chat:
                return "non_domain_query"
            else:
                return "non_domain_refuse"

    def _ask_user_for_clarification_node(self, state: RAGState) -> dict:
        """Ask the user for clarification or a different question"""
        self.logger.debug(f"{state =}")

        answer = AIMessage(
            "Apologies. I could not answer that question. Can you please ask another one or try to reword it and I will try again?"
        )

        self.logger.info(f"Asking user for clarification: {answer.content}")

        return {"message_for_user": answer.content}

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
            num_history_messages=10,
        )
        self.workflow.add_node(
            "classify_question_domain", self._classify_question_node.execute
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
            num_history_messages=10,
            fallback_config=self.stores.vs_config.fallback_to_training_data,
        )
        self.workflow.add_node(
            "answer_general_question", self._answer_general_node.execute
        )
        self._refuse_answer_node = RefuseAnswer(logger=self.logger, stores=self.stores)
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
        self._answer_user_node = AnswerUser(logger=self.logger)
        self.workflow.add_node(
            "give_domain_answer_to_user", self._answer_user_node.execute
        )
        self.workflow.add_node(
            "ask_user_for_clarification", self._ask_user_for_clarification_node
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
            self._route_query_domain_node,
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
            self._route_evaluator_node,
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
