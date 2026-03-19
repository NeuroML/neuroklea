#!/usr/bin/env python3
"""
General RAG implementation

File: rag.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.llm import (
    add_memory_to_prompt,
    get_history_summary_prompt,
    get_last_n_conversations,
    parse_output_with_thought,
    setup_llm,
    split_output_by_section,
)
from neuroml_ai_utils.logging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)
from pydantic import create_model

from .config import AppConfig
from .schemas import EvaluateAnswerSchema, RAGState
from .stores import Vector_Stores

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class RAG(object):
    """General RAG implementation"""

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        memory: bool = True,
    ):
        """Initialise"""
        self.c_model = None
        self.config_file = os.getenv("GEN_RAG_CONFIG_FILE", "rag.env")
        self.config: AppConfig

        # total number of reference documents
        self.num_refs_max = 10

        # Whether this graph should manage it's own memory/checkpoints
        # Can be turned off when this RAG is being included in another
        # graph/app
        self.memory = memory
        if self.memory:
            self.checkpointer: Optional[InMemorySaver] = InMemorySaver()
        else:
            self.checkpointer = None

        # toggle for answer generator
        self.modify_query = False

        # number of conversations after which to summarise
        # no need to summarise after each
        # 5 rounds: 10 messages
        self.num_recent_messages = 10

        self.logger = logging.getLogger("RAG")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(logger_formatter_info)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging_level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(logger_formatter_other)
        self.logger.addHandler(stderr_handler)

    def _load_config(self):
        """Load configuration from file"""
        cfg_path = Path(self.config_file)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Could not find config file: {self.config_file}")

        self.config = AppConfig(_env_file=self.config_file)
        self.logger.debug(f"Config file:\n{self.config_file}")
        self.logger.debug(f"Config:\n{self.config}")

    async def setup(self):
        """Set up basics."""

        self._load_config()
        self.stores = Vector_Stores(vs_config_file=self.config.vs_config_file)
        self._setup_chat_model()
        self.stores.setup()

        # dynamically generate schema for domains
        all_domains = self.stores.domains.copy()
        all_domains.append("undefined")

        self.QueryDomainSchema = create_model(
            "QueryDomainSchema", query_domain=(Literal[tuple(all_domains)], "undefined")
        )

        await self._create_graph()

    async def get_graph(self):
        """Setup and get compiled graph"""
        await self.setup()
        return self.graph

    def _setup_chat_model(self):
        """Set up the LLM chat model"""
        self.c_model = setup_llm(self.config.chat_model, self.logger)

    def _summarise_history_node(self, state: RAGState) -> dict:
        """Clean ups after every round of conversation"""
        assert self.c_model
        conversation, human_messages, ai_messages = get_last_n_conversations(
            state.messages, state.summarised_till, None
        )
        conversations_num = len(human_messages) + len(ai_messages)

        if conversations_num < self.num_recent_messages:
            self.logger.debug(
                f"Not enough conversations to summarise yet: {conversations_num}/{self.num_recent_messages}"
            )
            return {}

        prompt = get_history_summary_prompt(
            conversation=conversation,
            logger=self.logger,
            current_summary=state.context_summary,
        )
        output = self.c_model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        self.logger.debug(f"Current history summary is:\n{output.content}")
        thought, answer = split_output_by_section(output.content, "<think>", "</think>")

        # Do not update messages here, since we don't want this to be noted as
        # an AI response to a user query
        return {
            "context_summary": answer,
            "summarised_till": len(state.messages),
        }

    def _init_rag_state_node(self, state: RAGState) -> dict:
        """Initialise, reset state before next iteration"""
        return {
            "query_domain": "undefined",
            "text_response_eval": EvaluateAnswerSchema(),
            "message_for_user": "",
            "reference_material": {},
        }

    def _classify_question_domain(self, state: RAGState) -> dict:
        """Ask LLM to figure out the domain of the query"""
        assert self.c_model
        self.logger.debug(f"{state =}")

        messages = state.messages
        messages.append(HumanMessage(content=state.query))

        domain_info = self.stores.vs_config.domains

        domain_str = ""
        domain_str += self.stores.vs_config.pre_prompt
        domain_str += "\n\nCategories:\n\n"

        for d, info in domain_info.items():
            desc = info.description
            if not desc or len(desc) == 0:
                desc = f"if the question is about {d}"
            else:
                desc = f"if the question is about {desc}"
            domain_str += f"\n- {d}: {desc}"

        system_prompt = dedent("""
            You are an expert query classifier.
            Reason about the user's query to classify it into one of the given
            categories.

            """)
        system_prompt += domain_str + "\n- undefined: otherwise\n\n"
        system_prompt += dedent("""
            Rules:

            - Choose exactly ONE category
            - Base your decision on semantic intent
            - Do not explain your reasoning
            - Do not include any other additional text
            - Provide your answer ONLY as a JSON object matching the requested
              schema:
              {{
                "query_domain": "..."
              }}
            - Take past conversation history and context into account.

        """)

        if self.memory:
            system_prompt += add_memory_to_prompt(
                messages=state.messages,
                context_summary=state.context_summary,
                num_recent_messages=self.num_recent_messages,
            )

        self.logger.debug(f"{system_prompt = }")

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        query_node_llm = self.c_model.with_structured_output(
            self.QueryDomainSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke({"query": state.query})

        self.logger.debug(f"{prompt = }")

        output = query_node_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )

        self.logger.debug(f"{output = }")

        if output["parsing_error"]:
            query_domain_result = parse_output_with_thought(
                output["raw"], self.QueryDomainSchema
            )
        else:
            query_domain_result = output["parsed"]
            if isinstance(query_domain_result, str):
                query_domain_result = self.QueryDomainSchema(
                    query_domain=query_domain_result
                )
            elif isinstance(query_domain_result, dict):
                query_domain_result = self.QueryDomainSchema(**query_domain_result)
            else:
                if not isinstance(query_domain_result, self.QueryDomainSchema):
                    self.logger.critical(
                        f"Received unexpected query classification: {query_domain_result =}"
                    )
                    query_domain_result = self.QueryDomainSchema(
                        query_domain="undefined"
                    )

        self.logger.debug(f"{query_domain_result =}")
        return {
            "query_domain": query_domain_result.query_domain,
            "messages": messages,
        }

    def _refuse_to_answer_node(self, state: RAGState) -> dict:
        return {
            "message_for_user": "I cannot answer this query as it does not fall into my allowed domains. Please try another query."
        }

    def _answer_general_question_node(self, state: RAGState) -> dict:
        """Answer a general question"""
        assert self.c_model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        You are a warm, easy-going conversational assistant.
        Engage with the user and answer questions to the best of your ability.
        Reflect their tone, acknowledge what they say, and continue the conversation naturally.

        ## Core directives

        - Do not assume this question is related to any particular domain.
        - Only provide information you are confident about. If you are unsuare, clearly say so.
        - Avoid inventing facts. If a fact is not known or uncertain, respond with "I was unable to find factual information about this query".
        - Keep answers clear, concise, and user-friendly.
        - Respond in a formal, academic style.

        Examples:
        User: Thank you.
        Assistant: You are welcome.
        User: I like cats.
        Assistant: That's great, I like cats too. I also like dogs.

        """)
        if self.memory:
            system_prompt += add_memory_to_prompt(
                messages=state.messages,
                context_summary=state.context_summary,
                num_recent_messages=self.num_recent_messages,
            )

        question_prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )
        self.logger.debug(f"{question_prompt_template =}")
        prompt = question_prompt_template.invoke({"query": state.query})

        output = self.c_model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        self.logger.debug(f"{output =}")

        answer = ""
        if self.stores.answer_non_domain.enabled:
            answer += "\n\n" + self.stores.answer_non_domain.warning + "\n\n"

        thought, answer_text = split_output_by_section(
            output.content, "<think>", "</think>"
        )
        answer += answer_text

        messages = state.messages
        output.content = answer
        messages.append(output)

        return {"messages": messages, "message_for_user": output.content}

    def _generate_retrieval_query_node(self, state: RAGState) -> dict:
        """Generate a retrieval query"""
        assert self.c_model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        Generate a concise retrieval query from the user's question.  Think
        about the user's intent step by step.

        Directives:
        - a concept is a single technical entity or noun phrase
        - extract all concepts from the query
        - split multiple concepts that are joined by 'and', commas and other
          conjunctions into separate, individual concepts
        - generate a query for EXACTLY one concept

        For the rewritten query:
        - only include content words (nouns, verbs, adjectives)
        - do NOT include stop words: a, an, the, in, of, for, on, at, and
        - limit yourself to 3-8 words
        - no sentences
        - no explanations
        - ignore sentency fluency, only use keywords

        Only return the rewritten query.
        """)

        if self.memory:
            system_prompt += add_memory_to_prompt(
                messages=state.messages,
                context_summary=state.context_summary,
                num_recent_messages=self.num_recent_messages,
            )

        if self.modify_query:
            # toggle off
            system_prompt += dedent("""
            Generate a new query on EXACTLY one of concepts that the
            evaluator's feedback says is missing.

            Evaluator feedback:
            {feedback}
            """)

        question_prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )
        prompt = question_prompt_template.invoke(
            {"query": state.query, "feedback": state.text_response_eval.summary}
        )
        self.logger.debug(f"{prompt =}")

        output = self.c_model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )

        self.logger.debug(f"{output =}")
        thought, answer = split_output_by_section(output.content, "<think>", "</think>")

        messages = state.messages
        output.content = answer
        messages.append(output)

        return {"messages": messages}

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

        # Do not add memory at this step: limit to provided context
        # if self.memory:
        #     system_prompt += add_memory_to_prompt(
        #         messages=state.messages,
        #         context_summary=state.context_summary,
        #         num_recent_messages=self.num_recent_messages,
        #     )
        #
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

        self.logger.debug(f"retrieval query: {state.messages[-1].content}")

        # current reference material
        reference_material = state.reference_material

        cleaned_query = state.messages[-1].content

        # new references, or more references for an existing query from all
        # stores
        res = self.stores.retrieve(domain_name=state.query_domain, query=cleaned_query)
        # rank info from all stores, keep top N
        # remember that when asking for more ks from the vector store, they'll
        # still return the initial ones, so we don't need to do any manual
        # merging here for more refs for a particular query
        sorted_res = sorted(res, key=lambda tup: tup[1], reverse=True)
        new_ref = {state.query_domain: sorted_res[: self.num_refs_max]}

        reference_material.update(new_ref)
        self.logger.debug(f"{reference_material =}")

        reference_material_text = self._serialize_reference(reference_material)
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
                ref_list = set(references.split())
                break

        if ref_list:
            answer_text += "\nReferences:" + "\n- ".join(ref_list)

        output.content = answer
        self.logger.debug(output.pretty_repr())

        messages = state.messages
        messages.append(output)

        return {"messages": messages, "reference_material": reference_material}

    def _get_reference_list(self, reference_material: Dict[str, List[Tuple]]) -> str:
        """Get a list of references from provided material

        Currently unused.

        :param reference_material:  Dict {cleaned query: list of tuples (doc, relevance score)}
        :returns: str representation of reference list

        """

        reference_list: list[str] = []
        for q, sorted_refs in reference_material.items():
            for r, score in sorted_refs:
                reference_list.append(r.metadata["url"])

        references = "## References\n"
        for ref in set(reference_list):
            references += f"\n- {ref}"

        self.logger.debug(f"{references =}")

        return references

    def _serialize_reference(self, reference_material: Dict[str, List[Tuple]]) -> str:
        """serialize references into text for use in context

        We also sort the documents based on their relevance scores.

        :param reference_material:  Dict {cleaned query: list of tuples (doc, relevance score)}
        :returns: str representation of reference

        """

        serialized = ""
        for q, sorted_refs in reference_material.items():
            ctr = 1
            serialized += f"## {q}\n"
            for r, score in sorted_refs:
                metadata = [
                    f"{key}: {val}"
                    for key, val in r.metadata.items()
                    if "header" in key.lower()
                ]
                metadata_str = f"### Document {ctr}/{len(sorted_refs)}: " + " | ".join(
                    metadata
                )
                serialized += "\n" + f"{metadata_str}\n"
                url = r.metadata.get("url", None)
                if url:
                    serialized += f"Reference URL: {url}\n"
                serialized += r.page_content
                ctr += 1

        reference_material_text = serialized.replace("{", "{{").replace("}", "}}")
        self.logger.debug(f"{reference_material_text =}")

        return reference_material_text

    def _evaluate_answer_node(self, state: RAGState) -> dict:
        """Evaluate the answer"""
        assert self.c_model

        evaluator_prompt = dedent("""
            You are a critical grader evaluating an answer produced by a retrieval-augmented generation (RAG) system.

            You are given:
            1. The user's question.
            2. The retrieved context used to generate the answer.
            3. The system's answer.

            Your job:
            - Judge the answer strictly based on the context.
            - DO NOT use external knowledge.
            - ALWAYS provide your answer as a JSON object matching the provided
              schema, with these values:

            {{
              "confidence": 0-1,        // How sufficent the context is to answer the question
              "coverage": 0-1,          // How well the context includes information about all parts/sub-questions in the query
              "relevance": 0-1,         // How well the answer addresses the question
              "groundedness": 0-1,      // How well it sticks to the provided context
              "coherence": 0-1,         // Logical flow and clarity
              "conciseness": 0-1,       // Avoids fluff or repetition
              "next_step": "continue", "retrieve_more_info", "modify_query", "rewrite_answer", "undefined"
              "summary": "A brief natural-language justification for the grades."
            }}

            Guidance for values:

            "coverage" and "confidence" are based ONLY on the context, NOT on
            the generated answer. Relevance, groundedness, coherence,
            conciseness are related only to the generated answer.

            coverage:
            * 0.8 - 1.0: all sub-questions/topics/concepts present in context
            * 0.4 - 0.7: some covered, some missing
            * 0.0 - 0.3: most sub-questions missing

            confidence:
            * 0.8 - 1.0: clear, explicit, unambiguous context
            * 0.4 - 0.7: usable but incomplete
            * 0.0 - 0.3: vague/insufficient context

            relevance:
            * 0.8 - 1.0: fully addresses question
            * 0.4 - 0.7: partially addresses question
            * 0.0 - 0.3: barely addresses question

            groundedness:
            * 0.8 - 1.0: entirely based on context
            * 0.4 - 0.7: mixed grounded + inferred
            * 0.0 - 0.3: mostly hallucinated

            coherence:
            * 0.8 - 1.0: clear and logically structured
            * 0.4 - 0.7: understandable but uneven
            * 0.0 - 0.3: disorganised

            conciseness:
            * 0.8 - 1.0: mimimal + efficient
            * 0.4 - 0.7: moderately concise
            * 0.0 - 0.3: verbose

            Guidelines for 'next_step':
            1. high coverage, confident, relevant, grounded, with acceptable coherence and conciseness: return "continue".
            2. low coverage: return "modify_query".
            3. low confidence: return "retrieve_more_info".
            4. high coverage and confidence, low relevance, low groundedness, low coherence, or low conciseness: return "rewrite_answer"
            5. all coverage, confidence, relevance and groundedness are low: return "undefined".

            Always return a brief summary.
            """)

        question = state.query

        context = self._serialize_reference(state.reference_material)
        answer = state.messages[-1].content
        assert len(question)
        assert len(context)
        assert len(answer)

        prompt_template = ChatPromptTemplate(
            [
                ("system", evaluator_prompt),
                (
                    "human",
                    dedent("""
                        Question:
                        {question}

                        -----

                        Context
                        {context}

                        -----

                        Answer:
                        {answer}

                 """),
                ),
            ]
        )

        # can use | to merge these lines
        query_node_llm = self.c_model.with_structured_output(
            EvaluateAnswerSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
            }
        )

        output = query_node_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.0}}
        )
        self.logger.debug(f"{output =}")

        if output["parsing_error"]:
            result = parse_output_with_thought(output["raw"], EvaluateAnswerSchema)
        else:
            result = output["parsed"]

        # do not store the evaluation message in state
        return {"text_response_eval": result, "messages": state.messages}

    def _route_answer_evaluator_node(self, state: RAGState) -> str:
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
            return "continue"
        elif next_step == "modify_query" or resp.coverage < 0.3:
            self.modify_query = True
            return "modify_query"
        elif next_step == "retrieve_more_info" or (
            resp.coverage >= 0.5 and resp.confidence < 0.5
        ):
            # limit what max k we can have, otherwise, we end up pulling the
            # whole store..
            if self.stores.inc_k():
                return "retrieve_more_info"
            else:
                # we are already at max context, so we need to modify the query
                # to get a better result
                self.modify_query = True
                return "modify_query"
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
            return "rewrite_answer"
        else:
            return "undefined"

    def _route_query_domain_node(self, state: RAGState) -> str:
        """Route the query depending on LLM's result"""
        self.logger.debug(f"{state =}")
        query_domain = state.query_domain

        if query_domain in self.stores.domains and query_domain != "undefined":
            return "generate_retrieval_query"
        else:
            if self.stores.answer_non_domain.enabled:
                return "answer_general_question"
            else:
                return "refuse_to_answer"

    def _give_domain_answer_to_user_node(self, state: RAGState) -> dict:
        """Return the answer message to the user"""
        self.logger.debug(f"{state =}")

        messages = state.messages
        answer = messages[-1]

        self.logger.info(f"Returning final answer to user: {answer}")

        return {"message_for_user": answer.content}

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
        self.workflow.add_node("init_rag_state", self._init_rag_state_node)
        self.workflow.add_node(
            "classify_question_domain", self._classify_question_domain
        )

        self.workflow.add_node(
            "generate_retrieval_query", self._generate_retrieval_query_node
        )
        self.workflow.add_node(
            "answer_general_question", self._answer_general_question_node
        )
        self.workflow.add_node("refuse_to_answer", self._refuse_to_answer_node)
        self.workflow.add_node(
            "generate_answer_from_context", self._generate_answer_from_context_node
        )
        self.workflow.add_node("evaluate_answer", self._evaluate_answer_node)
        self.workflow.add_node(
            "give_domain_answer_to_user", self._give_domain_answer_to_user_node
        )
        self.workflow.add_node(
            "ask_user_for_clarification", self._ask_user_for_clarification_node
        )
        if self.memory:
            self.workflow.add_node("summarise_history", self._summarise_history_node)

        self.workflow.add_edge(START, "init_rag_state")
        self.workflow.add_edge("init_rag_state", "classify_question_domain")

        self.workflow.add_conditional_edges(
            "classify_question_domain",
            self._route_query_domain_node,
            {
                "generate_retrieval_query": "generate_retrieval_query",
                "answer_general_question": "answer_general_question",
                "refuse_to_answer": "refuse_to_answer",
            },
        )
        self.workflow.add_conditional_edges(
            "evaluate_answer",
            self._route_answer_evaluator_node,
            {
                "continue": "give_domain_answer_to_user",
                "retrieve_more_info": "generate_answer_from_context",
                "rewrite_answer": "generate_answer_from_context",
                "modify_query": "generate_retrieval_query",
                "undefined": "ask_user_for_clarification",
            },
        )

        self.workflow.add_edge(
            "generate_retrieval_query", "generate_answer_from_context"
        )
        self.workflow.add_edge("generate_answer_from_context", "evaluate_answer")

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

        if not os.environ.get("RUNNING_IN_DOCKER", 0):
            try:
                self.graph.get_graph().draw_mermaid_png(
                    output_file_path="rag-lang-graph.png"
                )
            except BaseException as e:
                self.logger.error("Something went wrong generating lang graph png")
                self.logger.error(e)

    async def run_graph_invoke_state(
        self, state: dict, thread_id: str = "default_thread"
    ):
        """Run the graph but accept and return states"""

        config = {"configurable": {"thread_id": thread_id}}

        if "query" not in state:
            self.logger.error(f"Provided state should include the key 'query': {state}")
            sys.exit(-1)

        final_state = await self.graph.ainvoke(state, config=config)
        self.logger.debug(final_state)
        return final_state

    async def run_graph_invoke(self, query: str, thread_id: str = "default_thread"):
        """Run the graph by using and returning string input"""

        config = {"configurable": {"thread_id": thread_id}}

        final_state = await self.graph.ainvoke({"query": query}, config=config)

        self.logger.debug(f"{final_state =}")
        if message := final_state.get("message_for_user", None):
            return message
        else:
            return "I was unable to answer"

    async def run_graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.graph.astream({"query": query}, config=config):
            for node, state in chunk.items():
                self.logger.debug(f"{node}: {repr(state)}")
                # all nodes must return dicts
                if message := state.get("message_for_user", None):
                    self.logger.info(f"User message: {message}")
                    yield message
                else:
                    self.logger.debug(f"Working in node: {node}")

    async def graph_stream(self, query: str, thread_id: str = "default_threaD"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        res = await self.graph.astream({"query": query}, config=config)
        return res
