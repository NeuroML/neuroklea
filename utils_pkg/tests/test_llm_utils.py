#!/usr/bin/env python3
"""
Test llm utils

File:

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import unittest

import pytest

from klea_utils.llm import parse_model_name, split_output_by_section


@pytest.mark.parametrize(
    argnames=[
        "test_id",
        "text",
        "start_mark",
        "end_mark",
        "expected_delim",
        "expected_other",
    ],
    argvalues=[
        (
            "1",
            "some chat <thinking>some thought </thinking>",
            "<thinking>",
            "</thinking>",
            "some thought",
            "some chat",
        ),
        (
            "2",
            "<thinking>some thought </thinking>",
            "<thinking>",
            "</thinking>",
            "some thought",
            "",
        ),
        (
            "3",
            "some chat <thinking>some thought </thinking> some more chat",
            "<thinking>",
            "</thinking>",
            "some thought",
            "some chat  some more chat",
        ),
        (
            "4",
            "all thought no chat</thinking>",
            "<thinking>",
            "</thinking>",
            "all thought no chat",
            "\nNOTE: NO START MARKER FOUND",
        ),
        (
            "5",
            "<thinking>all thought no chat",
            "<thinking>",
            "</thinking>",
            "all thought no chat",
            "\nNOTE: NO END MARKER FOUND",
        ),
        (
            "6",
            "<nothinking>all chat",
            "<thinking>",
            "</thinking>",
            "",
            "<nothinking>all chat",
        ),
        (
            "7",
            "<thinking>some thought</thinking>",
            "<thinking>",
            "",
            "some thought</thinking>",
            "",
        ),
        (
            "8",
            "<thinking>some thought</thinking>some chat<thinking>more thought</thinking>",
            "<thinking>",
            "</thinking>",
            "some thoughtmore thought",
            "some chat",
        ),
        (
            "9",
            "<thinking>some thought</thinking>some chat<thinking>more thought</thinking> more chat",
            "<thinking>",
            "</thinking>",
            "some thoughtmore thought",
            "some chat more chat",
        ),
        (
            "10",
            "start chat <thinking>some thought</thinking>some chat<thinking>more thought</thinking>",
            "<thinking>",
            "</thinking>",
            "some thoughtmore thought",
            "start chat some chat",
        ),
        (
            "11",
            "start chat <thinking>some thought</thinking>some chat<thinking>more thought</thinking> end chat",
            "<thinking>",
            "</thinking>",
            "some thoughtmore thought",
            "start chat some chat end chat",
        ),
    ],
)
def test_split_output_by_section(
    test_id, text, start_mark, end_mark, expected_delim, expected_other
):
    delim, other = split_output_by_section(text, start_mark, end_mark)
    assert delim == expected_delim
    assert other == expected_other


@pytest.mark.parametrize(
    argnames=[
        "raw",
        "expected_provider",
        "expected_model",
        "expected_suffix",
    ],
    argvalues=[
        # provider:model:tag -> provider + model:tag
        ("ollama:bge-m3:latest", "ollama", "bge-m3:latest", None),
        ("ollama:qwen3:0.6b", "ollama", "qwen3:0.6b", None),
        # huggingface:org/model:suffix -> provider, model, suffix
        (
            "huggingface:intfloat/multilingual-e5-large:auto",
            "huggingface",
            "intfloat/multilingual-e5-large",
            "auto",
        ),
        # huggingface:org/model -> provider, model, no suffix
        ("huggingface:org/model", "huggingface", "org/model", None),
        # provider:model (2 parts) for any provider
        ("openai:gpt-4o", "openai", "gpt-4o", None),
        ("anthropic:claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5", None),
        ("deepseek:deepseek-chat", "deepseek", "deepseek-chat", None),
        ("google_genai:gemini-2.0-flash", "google_genai", "gemini-2.0-flash", None),
        # bare model name, no provider
        ("bge-m3", None, "bge-m3", None),
    ],
)
def test_parse_model_name(raw, expected_provider, expected_model, expected_suffix):
    parsed = parse_model_name(raw)
    assert parsed.provider == expected_provider
    assert parsed.model_name == expected_model
    assert parsed.suffix == expected_suffix


if __name__ == "__main__":
    unittest.main()
