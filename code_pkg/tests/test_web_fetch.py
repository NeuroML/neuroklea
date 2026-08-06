#!/usr/bin/env python3
"""
Tests for the bundled web_fetch tool.

File: code_pkg/tests/test_web_fetch.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import aiohttp
import pytest
from fastmcp import Client

from klea_code.tools.bundled import bundle_server
from klea_code.tools.web_fetch import web_fetch


class MockContext:
    """Test stub replacing fastmcp.Context"""

    def __init__(self):
        self.lifespan_context = {}

    def set_state(self, key, val):
        self.lifespan_context[key] = val


class _FakeResponse:
    def __init__(
        self,
        text: str,
        status: int = 200,
        content_type: str = "text/html",
    ):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._text = text

    async def text(self) -> str:
        return self._text


class _FakeGetCM:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def get(self, url, **kwargs):
        return _FakeGetCM(response=self._response, error=self._error)


def _ctx_with_session(session) -> MockContext:
    ctx = MockContext()
    ctx.set_state("aiohttp_session", session)
    return ctx


@pytest.mark.asyncio
async def test_web_fetch_rejects_invalid_url():
    ctx = MockContext()
    result = await web_fetch(ctx=ctx, url="not-a-url")
    assert result["content"] == ""
    assert result["status_code"] is None
    assert "http://" in result["error"] or "https://" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_missing_session():
    ctx = MockContext()
    result = await web_fetch(ctx=ctx, url="https://example.com")
    assert result["content"] == ""
    assert "session" in result["error"].lower()


@pytest.mark.asyncio
async def test_web_fetch_success_strips_html():
    fake = _FakeResponse(
        "<html><body><h1>Hello</h1><script>x()</script></body></html>",
        status=200,
        content_type="text/html; charset=utf-8",
    )
    result = await web_fetch(
        ctx=_ctx_with_session(_FakeSession(response=fake)),
        url="https://example.com",
    )
    assert result["status_code"] == 200
    assert "Hello" in result["content"]
    assert "<h1>" not in result["content"]
    assert "x()" not in result["content"]
    assert result["truncated"] is False
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_plain_text_passthrough():
    fake = _FakeResponse("plain body", status=200, content_type="text/plain")
    result = await web_fetch(
        ctx=_ctx_with_session(_FakeSession(response=fake)),
        url="https://example.com/raw.txt",
    )
    assert result["content"] == "plain body"
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_truncates():
    fake = _FakeResponse("abcdefghij", status=200, content_type="text/plain")
    result = await web_fetch(
        ctx=_ctx_with_session(_FakeSession(response=fake)),
        url="https://example.com",
        max_chars=4,
    )
    assert result["content"] == "abcd"
    assert result["truncated"] is True
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_http_error():
    fake = _FakeResponse("missing", status=404, content_type="text/plain")
    result = await web_fetch(
        ctx=_ctx_with_session(_FakeSession(response=fake)),
        url="https://example.com/missing",
    )
    assert result["status_code"] == 404
    assert "404" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_timeout():
    result = await web_fetch(
        ctx=_ctx_with_session(_FakeSession(error=TimeoutError())),
        url="https://example.com",
        timeout=1.0,
    )
    assert result["content"] == ""
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_client_error():
    result = await web_fetch(
        ctx=_ctx_with_session(
            _FakeSession(error=aiohttp.ClientConnectionError("refused"))
        ),
        url="https://example.com",
    )
    assert result["content"] == ""
    assert "refused" in result["error"]


@pytest.mark.asyncio
async def test_bundled_server_exposes_web_fetch():
    async with Client(transport=bundle_server) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        assert "list_files" in names
        assert "web_fetch" in names
