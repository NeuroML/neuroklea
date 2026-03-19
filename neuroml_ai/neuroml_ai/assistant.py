#!/usr/bin/env python3
"""
Assistant combining various (RAG/code agents)

File: assistant.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import os
import sys
from textwrap import dedent

from gen_rag.rag import RAG
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.llm import (
    add_memory_to_prompt,
    parse_output_with_thought,
    setup_llm,
)
from neuroml_ai_utils.logging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)

from .config import AppConfig
from .schemas import AssistantState, QueryTypeSchema


class NML_Assistant(object):
    """NeuroML Assistant class"""

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
    ):
        self.g_model = None
        self.c_model = None
        self.config_file = os.getenv("NML_AI_CONFIG_FILE", "assistant.env")
        self.config: AppConfig

        self.checkpointer = InMemorySaver()

        # number of conversations after which to summarise
        # no need to summarise after each
        # 5 rounds: 10 messages
        self.num_recent_messages = 10

        self.logger = logging.getLogger("NML_AI_Assistant")
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

    def _init_rag_state_node(self, state: AssistantState) -> dict:
        """Initialise, reset state before next iteration"""
        return {
            "query_type": QueryTypeSchema(),
            "message_for_user": "",
            "reference_material": {},
        }

    def _guard_node(self, state: AssistantState) -> str:
        """LLM decides what type the user query is"""
        assert self.g_model
        self.logger.debug(f"{state =}")

        messages = state.messages
        messages.append(HumanMessage(content=state.query))

        system_prompt = ""

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        prompt = prompt_template.invoke({"query": state.query})
        self.logger.debug(f"{prompt = }")

        output = self.g_model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )

        if "unsafe" in output.content:
            return "unsafe"
        else:
            return "safe"

    # Note: does not prevent DDoS attacks. Do we want to exit if the user query is unsafe?
    def _decline_to_respond_node(self, state: AssistantState) -> dict:
        return {
            "message_for_user": "I cannot respond to this query. Please try another.",
        }

    def _classify_query_node(self, state: AssistantState) -> dict:
        """LLM decides what type the user query is"""
        assert self.c_model
        self.logger.debug(f"{state =}")

        messages = state.messages
        messages.append(HumanMessage(content=state.query))

        system_prompt = dedent("""
            You are an expert query classifier.
            Classify the user input into exactly one category based on its intent

            Valid categories (in order of priority):

            - question: The query is a request for information.
            - task: The user is asking you to perform an action. The action will be performed in a later step.

            Rules:

            - Choose exactly ONE category
            - Base your decision on semantic intent
            - Do not explain your reasoning
            - Do not include any other additional text
            - Provide your answer ONLY as a JSON object matching the requested schema.
            - Take past conversation history and context into account.

            Examples:

            - "What is a neuron?": {{"query_type": "question"}}
            - "What is the capital of France?": {{"query_type": "question"}}
            - "What are we talking about?": {{"query_type": "question"}}
            - "Run this code": {{"query_type": "task"}}
            - "Run this command": {{"query_type": "task"}}
            - "Run this simulation": {{"query_type": "task"}}
            """)

        system_prompt += add_memory_to_prompt(
            messages=state.messages,
            context_summary=state.context_summary,
            num_recent_messages=self.num_recent_messages,
        )

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        query_node_llm = self.c_model.with_structured_output(
            QueryTypeSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke({"query": state.query})

        self.logger.debug(f"{prompt = }")

        output = query_node_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        if output["parsing_error"]:
            query_type_result = parse_output_with_thought(
                output["raw"], QueryTypeSchema
            )
        else:
            query_type_result = output["parsed"]
            if isinstance(query_type_result, str):
                query_type_result = QueryTypeSchema(query_type=query_type_result)
            elif isinstance(query_type_result, dict):
                query_type_result = QueryTypeSchema(**query_type_result)
            else:
                if not isinstance(query_type_result, QueryTypeSchema):
                    self.logger.critical(
                        f"Received unexpected query classification: {query_type_result =}"
                    )
                    query_type_result = QueryTypeSchema(query_type="undefined")

        self.logger.debug(f"{query_type_result =}")
        return {
            "query_type": query_type_result,
            "messages": messages,
        }

    def _route_query_node(self, state: AssistantState) -> str:
        """Route the query depending on LLM's result"""
        self.logger.debug(f"{state =}")
        query_type = state.query_type.query_type

        return query_type

    def _code_node(self, state: AssistantState):
        """Dummy code node

        :param state: TODO
        :returns: TODO

        """
        return {"message_for_user": "Coding support is a work in progress"}

    def _setup_chat_model(self):
        """Set up the LLM chat model"""
        self.c_model = setup_llm(self.config.chat_model, self.logger)

    def _setup_guard_model(self):
        """Set up the LLM guard model"""
        self.g_model = setup_llm(self.config.guard_model, self.logger)

    async def setup(self):
        """Set up basics."""
        self._setup_guard_model()
        self._setup_chat_model()
        await self._create_graph()

    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(AssistantState)
        self.workflow.add_node("init_state", self._init_rag_state_node)
        self.workflow.add_node("guard", self._guard_node)
        self.workflow.add_node("classify_query", self._classify_query_node)
        self.workflow.add_node("decline_to_respond", self._decline_to_respond_node)

        self._rag_node = RAG(memory=False)
        self._rag_node_graph = await self._rag_node.get_graph()
        self.workflow.add_node("rag_graph", self._rag_node_graph)
        self.workflow.add_node("code_graph", self._code_node)

        self.workflow.add_edge(START, "init_state")
        self.workflow.add_edge("init_state", "guard")

        self.workflow.add_conditional_edges(
            "guard",
            {
                "unsafe": "decline_to_respond",
                "safe": "classify_query",
                "task": "classify_query",
            },
        )
        self.workflow.add_conditional_edges(
            "classify_query",
            self._route_query_node,
            {
                "undefined": "rag_graph",
                "question": "rag_graph",
                "task": "code_graph",
            },
        )
        self.workflow.add_edge("code_graph", END)
        self.workflow.add_edge("decline_to_respond", END)

        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        if not os.environ.get("RUNNING_IN_DOCKER", 0):
            try:
                self.graph.get_graph().draw_mermaid_png(
                    output_file_path="nml-ai-assistant-lang-graph.png"
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
