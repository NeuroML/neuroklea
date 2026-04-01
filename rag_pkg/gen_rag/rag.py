#!/usr/bin/env python3
"""
General RAG implementation

File: rag.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from textwrap import dedent

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.graph import BaseLangGraph
from neuroml_ai_utils.llm import (
    add_memory_to_prompt,
    get_history_summary_prompt,
    get_last_n_conversations,
    parse_output_with_thought,
    setup_llm,
    split_output_by_section,
)
from neuroml_ai_utils.stores import serialize_reference

from .config import AppConfig
from .nodes.answer_user import AnswerUser
from .nodes.evaluator import Evaluator
from .schemas import EvaluateAnswerSchema, RAGState

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

        # toggle for answer generator
        self.modify_query = False

    def _setup_models(self) -> None:
        """Set up the LLM chat model"""
        self.c_model = setup_llm(self.config.chat_model, self.logger)

    async def get_graph(self):
        """Setup and get compiled graph"""
        await self.setup()
        return self.graph

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
        self.modify_query = False
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
        msg = "Sorry. I cannot answer this query as it does not fall into my permitted domains. Available domains are:\n"
        msg += "\n- ".join([""] + self.stores.domains)
        msg += "\n\n\nPlease try another query."

        return {"message_for_user": msg}

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
        # add warning if we're falling back to training data for a domain query
        if (
            self.stores.vs_config.fallback_to_training_data
            and state.query_domain != "undefined"
        ):
            answer += (
                "\n\n"
                + self.stores.vs_config.fallback_to_training_data.warning
                + "\n\n"
            )

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
                ref_list = set(references.split())
                break

        if ref_list:
            answer_text += "\nReferences:" + "\n- ".join(ref_list)

        output.content = answer
        self.logger.debug(output.pretty_repr())

        messages = state.messages
        messages.append(output)

        return {"messages": messages, "reference_material": reference_material}

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
            self.logger.debug("returning: continue")
            return "continue"
        elif not self.modify_query and (
            next_step == "modify_query" or resp.coverage < 0.3
        ):
            self.modify_query = True
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
                if not self.modify_query:
                    self.modify_query = True
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
            self.workflow.add_node("summarise_history", self._summarise_history_node)

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
        self.workflow.add_edge(
            "generate_retrieval_query", "generate_answer_from_context"
        )
        self.workflow.add_edge("generate_answer_from_context", "evaluate_answer")

        self.workflow.add_conditional_edges(
            "evaluate_answer",
            self._route_answer_evaluator_node,
            {
                "continue": "give_domain_answer_to_user",
                "retrieve_more_info": "generate_answer_from_context",
                "rewrite_answer": "generate_answer_from_context",
                "modify_query": "generate_retrieval_query",
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
