#!/usr/bin/env python3
"""
API related common utils

File: klea_utils/api.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import httpx
from pydantic import AnyUrl
from pydantic import ValidationError as PydanticValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


def validate_url(value: str) -> str:
    """Return *value* if it is a valid HTTP(S) URL, else raise ``ValueError``."""
    try:
        AnyUrl(value)
    except PydanticValidationError:
        raise ValueError(f"'{value}' is not a valid HTTP(S) URL")
    return value


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.HTTPStatusError, httpx.ReadError, httpx.ReadTimeout)
    ),
    reraise=True,
)
async def check_api_is_ready(url: str):
    """Exponentially drop off checking that API is ready

    :param url: url of health end point
    :type url: str

    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

        return response.json()
