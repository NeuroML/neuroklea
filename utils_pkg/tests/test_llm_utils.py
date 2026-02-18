#!/usr/bin/env python3
"""
Test llm utils

File:

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import unittest

import pytest

from neuroml_ai_utils.llm import split_output_by_section


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


if __name__ == "__main__":
    unittest.main()
