"""Web search tool for the medical expert agent.

This is intentionally a thin Tavily-backed retriever that returns the shared
`RetrievedDocument` shape. The expert agent can inject these documents into
its prompt as RAG context and cite them in `MedicalExpertResponse`.
"""

import os
from typing import Any

from packages.schemas.retrieval import RetrievedDocument

DEFAULT_MAX_RESULTS = 3


def search_web(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[RetrievedDocument]:
    """Search the web for medical context using Tavily.

    Returns an empty list when web RAG is not configured. This keeps the
    medical expert usable in local/test environments that do not have Tavily
    credentials or the optional `medical-kb` dependencies installed.
    """
    query = query.strip()
    if not query:
        return []

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        return []

    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(
            query=query,
            search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
            max_results=max_results,
            include_answer=False,
        )
    except Exception:
        return []

    results = response.get("results", []) if isinstance(response, dict) else []
    documents: list[RetrievedDocument] = []
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or item.get("url") or f"Web result {index + 1}")
        snippet = str(item.get("content") or item.get("snippet") or item.get("raw_content") or "")
        source_url = item.get("url")
        score = _coerce_score(item.get("score"))

        documents.append(
            RetrievedDocument(
                title=title,
                snippet=snippet or title,
                source_url=str(source_url) if source_url else None,
                retrieval_source="web",
                score=score,
            )
        )

    return documents


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
