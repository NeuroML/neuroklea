#!/usr/bin/env python3
"""
Schemas used in the RAG

File: gen_rag/schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Dict, List, Literal, Tuple


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


class RAGState(BaseModel):
    """The state of the graph"""

    query: str = ""
    query_domain: str = "undefined"
    text_response_eval: EvaluateAnswerSchema = EvaluateAnswerSchema()
    messages: List[AnyMessage] = Field(default_factory=list)

    # summarised version of context so far
    context_summary: str = ""

    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""

    # reference material from retrievals
    reference_material: Dict[str, List[Tuple]] = Field(default_factory=dict)

    # whether the retrieval query has been modified based on evaluator feedback
    query_modified: bool = False
