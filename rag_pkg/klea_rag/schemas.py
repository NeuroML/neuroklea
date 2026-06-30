#!/usr/bin/env python3
"""
Schemas used in the RAG

File: klea_rag/schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any, Dict, List, Literal, Tuple

from fastmcp.client.client import CallToolResult
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field


class EvaluateAnswerSchema(BaseModel):
    """Evaluation of LLM generated answer. Descriptions given in the main prompt"""

    confidence: float = 0.0
    coverage: float = 0.0
    relevance: float = 0.0
    groundedness: float = 0.0
    coherence: float = 0.0
    conciseness: float = 0.0
    next_step: Literal[
        "continue", "retrieve_more_info", "modify_query", "rewrite_answer", "undefined"
    ] = Field(default="undefined")
    summary: str = ""


class ToolCallSchema(BaseModel):
    """Schema for tool call response."""

    tool: str = ""
    args: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


# For Tool Picker
class ToolCallsSchema(BaseModel):
    tool_calls: list[ToolCallSchema] = Field(default_factory=list)


class RAGState(BaseModel):
    """The state of the graph"""

    query: str = ""
    # schema for this is computed at run time for the classifier node
    query_domains: list[str] = ["undefined"]
    text_response_eval: EvaluateAnswerSchema = EvaluateAnswerSchema()
    guard_decision: str = "unsafe"
    messages: List[AnyMessage] = Field(default_factory=list)

    # summarised version of context so far
    context_summary: str = ""

    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""

    # tool calls
    tool_calls: list[ToolCallSchema] = Field(default_factory=list)
    tool_results: list[CallToolResult] = Field(default_factory=list)

    # reference material from retrievals
    reference_material: Dict[str, List[Tuple]] = Field(default_factory=dict)

    # number of retrieval query modification attempts in evaluator loop
    retrieval_attempts: int = 0

    # number of answer rewrite attempts in evaluator loop
    rewrite_attempts: int = 0

    # generated retrieval query for the current round
    retrieval_query: str = ""
