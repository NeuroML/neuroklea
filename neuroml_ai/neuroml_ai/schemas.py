#!/usr/bin/env python3
"""
Schema definitions

File: schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import Dict, List, Literal, Tuple


class QueryTypeSchema(BaseModel):
    """Schema for query type."""

    query_type: Literal["undefined", "question", "task"] = Field(
        default="undefined",
    )


# Note that a few of these are shared with the subgraphs
class AssistantState(BaseModel):
    query: str = ""
    query_type: QueryTypeSchema = QueryTypeSchema()
    messages: List[AnyMessage] = Field(default_factory=list)

    # summarised version of context so far
    context_summary: str = ""

    message_for_user: str = ""

    # reference material from retrievals
    reference_material: Dict[str, List[Tuple]] = Field(default_factory=dict)
