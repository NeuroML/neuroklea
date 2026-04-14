#!/usr/bin/env python3
"""
Schemas used in the RAG

File: gen_rag/schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastmcp.client.client import CallToolResult
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Any, Dict, List, Literal


class ToolCallSchema(BaseModel):
    """Schema for tool call response."""

    tool: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class CodeSchema(BaseModel):
    code: str = ""
    version: int = 0
    patches: List[str] = []


class StepSchema(BaseModel):
    step_number: int = 1
    description: str = ""
    suggested_tools: list[str] = Field(default_factory=list)
    depends_on: list[int] = []
    status: Literal["pending", "done", "failed"] = Field(default="pending")


class PlanSchema(BaseModel):
    step_list: List[StepSchema] = Field(default_factory=list)
    status: Literal["not_started", "in_progress", "completed", "failed", "aborted"] = (
        Field(default="not_started")
    )
    current_step_index: int = 0


class GoalSchema(BaseModel):
    goal: str = ""
    success_criteria: str = ""


class ArtefactSchema(BaseModel):
    id_: str = ""
    type_: str = ""
    content: Any
    # mtime!
    metadata: dict[str, Any] = {}


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    context_size: int = 0


class Discovery(BaseModel):
    # when it was created
    timestamp: int = 0
    # TODO
    # general: files, scripts
    # NeuroML specific: files, semantic info (ions/parameters)
    pass


class CodeAIState(BaseModel):
    """The state of the graph"""

    query: str = ""
    messages: List[AnyMessage] = Field(default_factory=list)
    guard_decision: str = "safe"
    usage_metrics: TokenUsage

    # code string if any
    code: CodeSchema = CodeSchema()

    # planning related
    goal: GoalSchema = GoalSchema()
    plan: PlanSchema = PlanSchema()
    step_outputs: Dict[int, list[CallToolResult]] = Field(default_factory=dict)
    # global project discovery information
    # only to be updated if files change
    discovery_persistent: Discovery = Discovery()
    # per step cache
    discovery_per_step: Discovery = Discovery()

    # { id -> artefact }
    artefacts: Dict[str, ArtefactSchema] = Field(default_factory=dict)

    # summarised version of context so far
    context_summary: str = ""

    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""
