#!/usr/bin/env python3
"""
NeuroML CodeAI implementation

File: code_ai.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from functools import cached_property
from textwrap import dedent
from typing import final

from langgraph.graph import END, START, StateGraph
from neuroml_ai_utils.graph.base import BaseLangGraph
from neuroml_ai_utils.llm import setup_llm

from neuroml_code_ai.nodes.answer_user import AnswerUser
from neuroml_code_ai.nodes.evaluator import Evaluator
from neuroml_code_ai.nodes.explore_planner import ExplorePlanner
from neuroml_code_ai.nodes.goal_setter import GoalSetter
from neuroml_code_ai.nodes.init_graph import InitGraphState
from neuroml_code_ai.nodes.planner import Planner
from neuroml_code_ai.nodes.tools_caller import ToolsCaller
from neuroml_code_ai.nodes.tools_picker import ToolsPicker

from .config import AppConfig
from .schemas import CodeAIState, GoalSchema

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


@final
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

    @cached_property
    def tool_description(self):
        """Get the tool description"""
        if self.mcp_client:
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
        else:
            return ""

    async def _step_router_node(self, state: CodeAIState) -> str:
        return state.task_plan.status

    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(CodeAIState)
        self._init_graph_state_node = InitGraphState(logger=self.logger)
        self.workflow.add_node("init_graph_state", self._init_graph_state_node.execute)
        # self.workflow.add_node("summarise_history", self._summarise_history_node)

        self._goal_setter_node = GoalSetter(
            logger=self.logger,
            model=self.r_model,
            temperature=0.01,
            output_schema=GoalSchema,
            memory=False,
        )
        self.workflow.add_node("goal_setter", self._goal_setter_node.execute)

        self._explore_planner_node = ExplorePlanner(
            logger=self.logger, model=self.r_model, temperature=0.01
        )
        self.workflow.add_node("explore_planner", self._explore_planner_node.execute)

        self._planner_node = Planner(
            logger=self.logger, model=self.r_model, temperature=0.01
        )
        self._planner_node.set_tools_description(self.tool_description)
        self._tools_picker_node = ToolsPicker(
            logger=self.logger, model=self.r_model, temperature=0.01
        )
        self._tools_picker_node.set_tools_description(self.tool_description)
        self._tools_caller_node = ToolsCaller(
            logger=self.logger, mcp_client=self.mcp_client
        )
        self._evaluator_node = Evaluator(logger=self.logger)
        self._answer_user_node = AnswerUser(logger=self.logger)
        self.workflow.add_node("planner", self._planner_node.execute)
        self.workflow.add_node("tools_picker", self._tools_picker_node.execute)
        # TODO: modify to use a ToolOrchestrator that can call multiple tools
        # in parallel asynchronously
        # Note that this depends on how the agent is setup---if it's setup to
        # run one call at a time, this isn't required, but ideally, it should
        # be able to call multiple tools---but the prompts/state schema will
        # need to updated for that
        self.workflow.add_node("tools_caller", self._tools_caller_node.execute)
        # Evaluator: needs to handle failed tool calls and ask the planner to
        # update the plan if required
        self.workflow.add_node("evaluator", self._evaluator_node.execute)
        self.workflow.add_node("step_router", self._step_router_node)
        self.workflow.add_node("give_answer_to_user", self._answer_user_node.execute)

        self.workflow.add_edge(START, "init_graph_state")
        self.workflow.add_edge("init_graph_state", "goal_setter")
        self.workflow.add_edge("goal_setter", "explore_planner")
        self.workflow.add_edge("explore_planner", "tools_picker")
        self.workflow.add_edge("planner", "tools_picker")
        self.workflow.add_edge("tools_picker", "tools_caller")

        self.workflow.add_conditional_edges(
            "evaluator",
            self._step_router_node,
            {
                "not_started": "planner",
                "in_progress": "tools_picker",
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
