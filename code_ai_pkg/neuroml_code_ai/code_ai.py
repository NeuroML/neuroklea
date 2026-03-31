#!/usr/bin/env python3
"""
NeuroML CodeAI implementation

File: code_ai.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from functools import cached_property
from pathlib import Path
from textwrap import dedent
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_json_schema
from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.graph import BaseLangGraph
from neuroml_ai_utils.llm import load_prompt, parse_output_with_thought, setup_llm
from neuroml_ai_utils.stores import VectorStores

from neuroml_code_ai import prompts
from neuroml_code_ai.nodes.goal_setter import GoalSetterNode
from neuroml_code_ai.nodes.init_graph import InitGraphStateNode
from neuroml_code_ai.nodes.planner import PlannerNode

from .api.conf import AppConfig
from .schemas import CodeAIState, GoalSchema, ToolCallSchema

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class CodeAI(BaseLangGraph):
    """NeuroML CodeAI implementation"""

    config_class = AppConfig
    config_env_var = "CODE_AI_CONFIG_FILE"
    config_file_default = "code_ai.env"
    logger_name = "NeuroML-AI-codegen"

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        memory: bool = True,
    ):
        """Initialise"""
        super().__init__(logging_level=logging_level, memory=memory)

        self.r_model = None

        self.mcp_tools = None

        self.stores: VectorStores | None = None

    def _setup_models(self) -> None:
        """Set up the LLM chat model"""
        self.c_model = setup_llm(self.config.code_model, self.logger)
        if self.config.code_model == self.config.reasoning_model:
            self.r_model = self.c_model
            self.logger.info(
                f"Same model used for both chat and reasoning: {self.config.code_model}"
            )
        else:
            self.r_model = setup_llm(self.config.reasoning_model, self.logger)

    async def _pre_graph(self) -> None:
        """List MCP tools and optionally set up vector stores."""
        async with self.mcp_client:
            self.mcp_tools = await self.mcp_client.list_tools()
        self.logger.debug(f"{self.mcp_tools =}")
        self.logger.debug(f"{self.tool_description =}")

        if self.config.vs_config_file:
            self.stores = VectorStores(
                vs_config_file=self.config.vs_config_file, logger=self.logger
            )
            self.stores.setup()
            self.logger.info(
                f"Vector stores loaded from {self.config.vs_config_file}: {self.stores.domains}"
            )

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
                    tool_picker_result = OutputSchema(tool="INVALID")

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
        self._init_graph_state_node = InitGraphStateNode(logger=self.logger)
        self.workflow.add_node("init_graph_state", self._init_graph_state_node.execute)
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

        self._planner_node = PlannerNode(
            logger=self.logger, model=self.r_model, temperature=0.01
        )
        self._planner_node.set_tools_description(self.tool_description)
        self.workflow.add_node("planner", self._planner_node.execute)
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

        if self.checkpointer:
            self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        else:
            self.graph = self.workflow.compile()

        self._export_graph_png("code-ai-lang-graph.png")
