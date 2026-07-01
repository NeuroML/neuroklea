#!/usr/bin/env python3
"""
Klea code framework implementation

File: klea_code.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import sys
from typing import Any, final, override

from fastmcp.mcp_config import MCPConfig, StdioMCPServer
from klea_utils.graph.base import BaseLangGraph
from klea_utils.llm import setup_llm
from klea_utils.nodes.fixed_answer import FixedAnswer
from klea_utils.nodes.guard import GuardNode
from klea_utils.nodes.guard_router import GuardRouterNode
from langgraph.graph import END, START, StateGraph

from klea_code.nodes.answer_user import AnswerUser
from klea_code.nodes.evaluator import Evaluator
from klea_code.nodes.explore_planner import ExplorePlanner
from klea_code.nodes.goal_setter import GoalSetter
from klea_code.nodes.init_graph import InitGraphState
from klea_code.nodes.planner import Planner
from klea_code.nodes.tools_caller import ToolsCaller
from klea_code.nodes.tools_picker import ToolsPicker
from klea_code.nodes.tools_router import ToolsRouter

from .config import AppConfig, AppEnv
from .schemas import GoalSchema, KleaCodeState

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


@final
class KleaCode(BaseLangGraph):
    """Klea Code implementation"""

    env_class = AppEnv
    env_var = "KLEA_CODE_ENV_FILE"
    env_file_default = "klea_code.env"
    config_class = AppConfig
    logger_name = "KleaCode"

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        memory: bool = True,
    ):
        """Initialise"""
        super().__init__(logging_level=logging_level, memory=memory)

        self.r_model = None
        self.g_model = None

    def _setup_models(self) -> None:
        """Set up the LLM chat model"""
        self.c_model = setup_llm(self.app_env.chat_model, self.logger)
        if self.app_env.chat_model == self.app_env.reasoning_model:
            self.r_model = self.c_model
            self.logger.info(
                f"Same model used for both chat and reasoning: {self.app_env.chat_model}"
            )
        else:
            self.r_model = setup_llm(self.app_env.reasoning_model, self.logger)
        self.g_model = setup_llm(self.app_env.guard_model, self.logger)

    @override
    def _configure_resources(self) -> None:
        """Configure MCP servers and a default domain.

        Merges the external MCP server (if any) with the bundled tools server
        into a single MCPConfig, and sets up a single domain that includes
        both so tool descriptions are built correctly.
        """
        # Build bundled server config
        bundle_server = StdioMCPServer(
            command=sys.executable,
            args=["-m", "klea_code.tools.bundled"],
        )

        # Merge external + bundled
        ext_servers: dict[str, Any] = dict(self.app_config.mcp_servers)
        all_servers = {**ext_servers, "bundled": bundle_server}

        self.mcp_config = MCPConfig(mcpServers=all_servers)
        self.domain_mcp_configs = {"code": MCPConfig(mcpServers=all_servers)}

    # TODO: replace with class
    async def _step_router_node(self, state: KleaCodeState) -> str:
        return state.plan.status

    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(KleaCodeState)

        self._init_graph_state_node = InitGraphState(
            logger=self.logger, label="Initializing"
        )
        self.workflow.add_node(
            self._init_graph_state_node.label, self._init_graph_state_node.execute
        )

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

        self._goal_setter_node = GoalSetter(
            logger=self.logger,
            label="Setting goal",
            model=self.r_model,
            temperature=0.01,
            output_schema=GoalSchema,
            memory=False,
        )
        self.workflow.add_node(
            self._goal_setter_node.label, self._goal_setter_node.execute
        )

        self._explore_planner_node = ExplorePlanner(
            logger=self.logger, label="Exploring", model=self.r_model, temperature=0.01
        )
        self.workflow.add_node(
            self._explore_planner_node.label, self._explore_planner_node.execute
        )

        self._planner_node = Planner(
            logger=self.logger, label="Planning", model=self.r_model, temperature=0.01
        )
        self._planner_node.set_tools_description(self.tools_description)
        self._tools_picker_node = ToolsPicker(
            logger=self.logger,
            label="Selecting tools",
            model=self.r_model,
            temperature=0.01,
        )
        self._tools_picker_node.set_tools_description(self.tools_description)
        self._tools_caller_node = ToolsCaller(
            logger=self.logger, label="Running tools", mcp_client=self.mcp_client
        )
        self._tools_router_node = ToolsRouter(logger=self.logger, label="Routing tools")
        self._evaluator_node = Evaluator(logger=self.logger, label="Evaluating")
        self._answer_user_node = AnswerUser(
            logger=self.logger, label="Preparing response"
        )
        self.workflow.add_node(self._planner_node.label, self._planner_node.execute)
        # TODO: modify to use a ToolOrchestrator that can call multiple tools
        # in parallel asynchronously
        # Note that this depends on how the agent is setup---if it's setup to
        # run one call at a time, this isn't required, but ideally, it should
        # be able to call multiple tools---but the prompts/state schema will
        # need to updated for that
        self.workflow.add_node(
            self._tools_caller_node.label, self._tools_caller_node.execute
        )
        self.workflow.add_node(
            self._tools_picker_node.label, self._tools_picker_node.execute
        )
        # Evaluator: needs to handle failed tool calls and ask the planner to
        # update the plan if required
        self.workflow.add_node(self._evaluator_node.label, self._evaluator_node.execute)
        self.workflow.add_node(
            self._answer_user_node.label, self._answer_user_node.execute
        )

        self.workflow.add_edge(START, self._init_graph_state_node.label)
        self.workflow.add_edge(
            self._init_graph_state_node.label, self._guard_node.label
        )
        self.workflow.add_conditional_edges(
            self._guard_node.label,
            self._guard_router_node.execute,
            {
                "safe": self._goal_setter_node.label,
                "unsafe": self._decline_to_answer_node.label,
            },
        )
        self.workflow.add_edge(
            self._goal_setter_node.label, self._explore_planner_node.label
        )
        self.workflow.add_edge(
            self._explore_planner_node.label, self._tools_picker_node.label
        )
        self.workflow.add_edge(self._planner_node.label, self._tools_picker_node.label)
        self.workflow.add_edge(
            self._tools_picker_node.label, self._tools_caller_node.label
        )
        # TODO: we probably need a node here that takes tools output from
        # picker and puts them in the right state field for exploration
        # TODO: we also need some flag that decides whether the next step here
        # should be planning or evaluation. If it's coming off exploration, it
        # needs to go to planning. If it's in the plan, it needs to go to
        # evaluation
        self.workflow.add_conditional_edges(
            self._tools_caller_node.label,
            self._tools_router_node.execute,
            {
                "failed": self._tools_picker_node.label,
                "explored": self._planner_node.label,
                "continue": self._evaluator_node.label,
            },
        )

        self.workflow.add_conditional_edges(
            self._evaluator_node.label,
            self._step_router_node,
            {
                # should never be here
                "not_started": self._planner_node.label,
                # next step
                "in_progress": self._tools_picker_node.label,
                # plan isn't working
                "failed": self._planner_node.label,
                "aborted": self._answer_user_node.label,
                "completed": self._answer_user_node.label,
            },
        )
        self.workflow.add_edge(self._answer_user_node.label, END)
        self.workflow.add_edge(self._decline_to_answer_node.label, END)

        if self.checkpointer:
            self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        else:
            self.graph = self.workflow.compile()

        self._export_graph_png("code-ai-lang-graph.png")
