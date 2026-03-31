#!/usr/bin/env python3
"""
NeuroML CodeAI implementation

File: code_ai.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import os
import sys
from functools import cached_property
from pathlib import Path
from textwrap import dedent
from typing import Any

from fastmcp import Client
from fastmcp.mcp_config import MCPConfig

# from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_json_schema
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.llm import load_prompt, parse_output_with_thought, setup_llm
from neuroml_ai_utils.logging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)

from neuroml_code_ai import prompts
from neuroml_code_ai.nodes.goal_setter import GoalSetterNode

from .api.conf import AppConfig
from .schemas import CodeAIState, GoalSchema, PlanSchema, ToolCallSchema

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class CodeAI(object):
    """NeuroML CodeAI implementation"""

    checkpointer = InMemorySaver()

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        memory: bool = True,
    ):
        """Initialise"""
        self.c_model = None
        self.r_model = None
        self.mcp_client: Client

        # number of conversations after which to summarise
        # no need to summarise after each
        # 5 rounds: 10 messages
        self.num_recent_messages = 10

        self.mcp_tools = None

        self.config_file = os.getenv("CODE_AI_CONFIG_FILE", "code_ai.env")
        self.config: AppConfig

        self.logger = logging.getLogger("NeuroML-AI-codegen")
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

    async def setup(self):
        """Set up basics."""

        self._load_config()
        self._setup_models()
        self._create_mcp_client()

        async with self.mcp_client:
            self.mcp_tools = await self.mcp_client.list_tools()
        self.logger.debug(f"{self.mcp_tools =}")

        self.logger.debug(f"{self.tool_description =}")
        await self._create_graph()

    @cached_property
    def tool_description(self):
        """Get the tool description"""
        ctr = 0
        description = ""
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

    def _setup_models(self):
        """Set up the LLM chat model"""
        self.c_model = setup_llm(self.config.code_model, self.logger)
        if self.config.code_model == self.config.reasoning_model:
            self.r_model = self.c_model
            self.logger.info(
                f"Same model used for both chat and reasoning: {self.config.code_model}"
            )
        else:
            self.r_model = setup_llm(self.config.reasoning_model, self.logger)

    def _load_config(self):
        """Load configuration from file"""
        cfg_path = Path(self.config_file)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Could not find config file: {self.config_file}")

        self.config = AppConfig(_env_file=self.config_file)
        assert self.config
        self.logger.debug(f"Config file: {self.config_file}")
        self.logger.debug(f"Config: {self.config}")

    def _create_mcp_client(self):
        """Create MCP client from config"""
        mcp_config_text = ""
        with open(self.config.mcp_config_file, "r") as f:
            mcp_config_text = json.load(f)
            self.logger.debug(f"{mcp_config_text = }")

        self.mcp_config = MCPConfig(**mcp_config_text)
        self.logger.debug(f"{self.mcp_config = }")
        self.mcp_client = Client(self.mcp_config)
        assert self.mcp_client

    def _init_graph_state_node(self, state: CodeAIState) -> dict:
        """Initialise, reset state before next iteration"""
        return {
            "message_for_user": "",
            "plan": PlanSchema(),
            "goal": GoalSchema(),
            "tool_call": None,
            "tool_responses": [],
        }

    async def _planner_node(self, state: CodeAIState) -> dict:
        my_model = self.r_model
        assert my_model
        self.logger.debug(f"{state =}")

        system_prompt = load_prompt(
            prompt_name="planner",
            prompt_registry_location=Path(prompts.__file__).parent,
        )
        self.logger.debug(f"{system_prompt = }")

        OutputSchema = PlanSchema
        self.logger.debug(f"{convert_to_json_schema(OutputSchema) =}")

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        planner_llm = my_model.with_structured_output(
            OutputSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke(
            {
                "query": state.query,
                "goal": state.goal,
                "step_list": state.plan.step_list,
                "current_step_index": state.plan.current_step_index,
                "artefacts": state.artefacts,
                "observations": state.tool_responses,
                "tools_description": self.tool_description,
                "output_schema": convert_to_json_schema(OutputSchema),
            }
        )

        self.logger.debug(f"{prompt = }")

        output = planner_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.01}}
        )

        self.logger.debug(f"{output = }")

        if output["parsing_error"]:
            plan_result = parse_output_with_thought(output["raw"], OutputSchema)
            self.logger.debug(f"parse error {plan_result =}")
        else:
            plan_result = output["parsed"]
            self.logger.debug(f"parsed {plan_result =}")
            # error: it gives a slightly different output
            # {'step_list': [{'step_id': 1, 'summary': 'Recursively list all .md and .py files in the current directory', 'tool_call': True, 'inputs': "Directory: '.', pattern: '*.md *.py', recursive: True", 'output': 'List of full paths to .md and .py files'}], 'status': 'ready', 'current_step': 1}
            # TODO:debug and fix: need to include example output in prompt? why doesn't structured output work
            if isinstance(plan_result, dict):
                plan_result = OutputSchema(**plan_result)
            else:
                if not isinstance(plan_result, OutputSchema):
                    self.logger.critical(
                        f"Received unexpected query classification: {plan_result =}"
                    )
                    plan_result = OutputSchema(status="failed")

        self.logger.debug(f"{plan_result =}")

        # Generate a plan summary and send to user
        plan_summary = "## Plan summary:\n\n"
        for step in plan_result.step_list:
            plan_summary += f"- {step.step_number}: {step.description}"

        plan = state.plan
        plan.step_list = plan_result.step_list

        return {"plan": plan, "message_for_user": plan_summary}

    async def _tool_picker_node(self, state: CodeAIState) -> dict:
        assert self.c_model
        self.logger.debug(f"{state =}")
        current_step_index = state.plan.current_step_index
        current_step = state.plan.step_list[current_step_index]

        system_prompt = load_prompt(
            prompt_name="tool_picker",
            prompt_registry_location=Path(prompts.__file__).parent,
        )
        self.logger.debug(f"{system_prompt = }")

        OutputSchema = ToolCallSchema

        prompt_template = ChatPromptTemplate([("system", system_prompt)])

        # can use | to merge these lines
        planner_llm = self.c_model.with_structured_output(
            OutputSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke(
            {
                "current_step": current_step,
                "artefacts": state.artefacts,
                "observations": state.tool_responses,
                "tools_description": self.tool_description,
                # TODO: investigate use of OutputSchema.model_json_schema()
                "output_schema": convert_to_json_schema(OutputSchema),
            }
        )

        self.logger.debug(f"{prompt = }")

        output = planner_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.01}}
        )

        self.logger.debug(f"{output = }")

        if output["parsing_error"]:
            tool_picker_result = parse_output_with_thought(output["raw"], OutputSchema)
        else:
            tool_picker_result = output["parsed"]
            if isinstance(tool_picker_result, dict):
                tool_picker_result = OutputSchema(**tool_picker_result)
            else:
                if not isinstance(tool_picker_result, OutputSchema):
                    self.logger.critical(
                        f"Received unexpected LLM output: {tool_picker_result =}"
                    )
                    tool_picker_result = OutputSchema(status="failed")

        self.logger.debug(f"{tool_picker_result =}")

        return {"tool_call": tool_picker_result}

    async def _tool_caller_node(self, state: CodeAIState) -> dict:
        self.logger.debug(f"{state =}")

        plan = state.plan
        current_step = plan.step_list[plan.current_step_index]
        result: dict[str, Any] = {}

        # call tool if it is in the current state
        if state.tool_call:
            # TODO: retry X times if fails before marking as failed
            tool_call = state.tool_call
            assert tool_call

            tool_responses = state.tool_responses
            async with self.mcp_client:
                tool_result = await self.mcp_client.call_tool(
                    name=tool_call.tool, arguments=tool_call.args, raise_on_error=False
                )
            tool_responses.append(tool_result)

            if tool_result.is_error:
                current_step.status = "failed"
            else:
                current_step.status = "done"

            # TODO: populate artefacts
            result["tool_responses"] = tool_responses
            self.logger.debug(f"{tool_responses =}")

        plan.current_step_index += 1

        result["plan"] = plan
        return result

    # TODO
    async def _evaluator_node(self, state: CodeAIState) -> dict:
        plan = state.plan
        result = {}

        # if all steps completed
        if plan.current_step_index > len(plan.step_list):
            plan.status = "completed"
            result["plan"] = plan

        # if any steps failed?

        return result

    async def _step_router_node(self, state: CodeAIState) -> str:
        return state.plan.status

    def _give_answer_to_user_node(self, state: CodeAIState) -> dict:
        """Return the message to the user"""
        self.logger.debug(f"{state =}")

        answer = state.message_for_user
        self.logger.info(f"Returning final answer to user: {answer}")

        return {"message_for_user": answer}

    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(CodeAIState)
        self.workflow.add_node("init_graph_state", self._init_graph_state_node)
        # self.workflow.add_node("summarise_history", self._summarise_history_node)

        self._goal_setter_node = GoalSetterNode(
            logger=self.logger,
            model=self.r_model,
            temperature=0.01,
            output_schema=GoalSchema,
            system_prompt_file="goal",
            human_prompt_file="goal_human",
            memory=False,
        )
        self.workflow.add_node("goal_setter", self._goal_setter_node.execute)

        self.workflow.add_node("planner", self._planner_node)
        self.workflow.add_node("tool_picker", self._tool_picker_node)
        self.workflow.add_node("tool_caller", self._tool_caller_node)
        self.workflow.add_node("evaluator", self._evaluator_node)
        self.workflow.add_node("step_router", self._step_router_node)
        self.workflow.add_node("give_answer_to_user", self._give_answer_to_user_node)

        self.workflow.add_edge(START, "init_graph_state")
        self.workflow.add_edge("init_graph_state", "goal_setter")
        self.workflow.add_edge("goal_setter", "planner")
        self.workflow.add_edge("planner", "tool_picker")
        self.workflow.add_edge("tool_picker", "tool_caller")

        self.workflow.add_conditional_edges(
            "evaluator",
            self._step_router_node,
            {
                "not_started": "planner",
                "in_progress": "tool_picker",
                "failed": "planner",
                "aborted": "give_answer_to_user",
                "completed": "give_answer_to_user",
            },
        )
        self.workflow.add_edge("give_answer_to_user", END)

        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        if not os.environ.get("RUNNING_IN_DOCKER", 0):
            try:
                self.graph.get_graph().draw_mermaid_png(
                    output_file_path="code-ai-lang-graph.png"
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
