"""Tests for the medical expert web search tool."""

from services.medical_expert_agent.tools.search_web import search_web


def test_search_web_returns_empty_when_tavily_key_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert search_web("diabetes clinical guidance") == []
