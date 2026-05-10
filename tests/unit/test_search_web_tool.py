"""Tests for the medical expert web search tool."""

import tavily

from services.medical_expert_agent.tools.search_web import medical_web_max_results, search_web


def test_search_web_returns_empty_when_tavily_key_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("MEDICAL_WEB_SEARCH_API_KEY", raising=False)

    assert search_web("diabetes clinical guidance") == []


def test_search_web_filters_to_trusted_medical_domains_by_default(monkeypatch):
    calls = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def search(self, **kwargs):
            calls.update(kwargs)
            return {"results": []}

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", raising=False)
    monkeypatch.setattr(tavily, "TavilyClient", FakeTavilyClient)

    assert search_web("diabetes clinical guidance") == []
    assert calls["include_domains"] == ["medlineplus.gov", "pubmed.ncbi.nlm.nih.gov"]


def test_search_web_uses_configured_domain_filter(monkeypatch):
    calls = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def search(self, **kwargs):
            calls.update(kwargs)
            return {"results": []}

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("MEDICAL_WEB_SEARCH_INCLUDE_DOMAINS", "medlineplus.gov, cdc.gov")
    monkeypatch.setattr(tavily, "TavilyClient", FakeTavilyClient)

    assert search_web("diabetes clinical guidance") == []
    assert calls["include_domains"] == ["medlineplus.gov", "cdc.gov"]


def test_search_web_uses_marge_web_rag_max_results(monkeypatch):
    calls = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        def search(self, **kwargs):
            calls.update(kwargs)
            return {
                "results": [
                    {"title": "one", "content": "a", "url": "https://medlineplus.gov/1", "score": 1},
                    {"title": "two", "content": "b", "url": "https://medlineplus.gov/2", "score": 2},
                    {"title": "three", "content": "c", "url": "https://medlineplus.gov/3", "score": 3},
                ]
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "2")
    monkeypatch.setattr(tavily, "TavilyClient", FakeTavilyClient)

    docs = search_web("diabetes clinical guidance", max_results=5)

    assert calls["max_results"] == 2
    assert len(docs) == 2


def test_medical_web_max_results_clamps_invalid_env(monkeypatch):
    monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "bad")

    assert medical_web_max_results(requested=5) == 3

    monkeypatch.setenv("MARGE_WEB_RAG_MAX_RESULTS", "9")

    assert medical_web_max_results(requested=5) == 5
