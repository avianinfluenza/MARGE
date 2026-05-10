"""Expert-only live medical web search tool.

The orchestrator never imports this module. It is wired only inside
`MedicalExpertAgent`, so any `search_medical_web` call in the trace proves the
expert model initiated the retrieval step.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from packages.schemas.retrieval import RetrievedDocument

TOOL_NAME = "search_medical_web"
TOOL_DESCRIPTION = (
    "Search the live web for current, citable medical guidance. Use this before "
    "making guideline, diagnostic-threshold, treatment, or quantitative clinical "
    "claims. Prefer authoritative sources such as ADA, CDC, WHO, NICE, USPSTF, "
    "NIH, and peer-reviewed clinical references."
)


class ToolInput(BaseModel):
    query: str = Field(
        description=(
            "Focused medical search query, including the condition, value, "
            "threshold, or guideline body when relevant."
        )
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of web results to return.",
    )


async def search_medical_web(query: str, max_results: int = 3) -> dict[str, Any]:
    """Run a Tavily-backed web search and return normalized retrieval docs.

    Missing optional setup is returned as a warning instead of raising so the
    expert consultation can still complete while the trace clearly shows that
    web search was attempted but not configured.
    """

    api_key = os.getenv("TAVILY_API_KEY") or os.getenv("MEDICAL_WEB_SEARCH_API_KEY")
    if not api_key:
        return {
            "query": query,
            "documents": [],
            "warning": "TAVILY_API_KEY is not set; live web search was not executed.",
        }

    try:
        from tavily import TavilyClient
    except ImportError:
        return {
            "query": query,
            "documents": [],
            "warning": (
                "tavily-python is not installed. Install the medical-kb extra "
                "to enable live web search."
            ),
        }

    client = TavilyClient(api_key=api_key)
    raw = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
    )

    documents: list[dict[str, Any]] = []
    for idx, item in enumerate(raw.get("results", [])[:max_results], start=1):
        score = item.get("score")
        try:
            score_value = float(score) if score is not None else float(idx)
        except (TypeError, ValueError):
            score_value = float(idx)

        doc = RetrievedDocument(
            title=str(item.get("title") or "Untitled medical web result"),
            snippet=str(item.get("content") or item.get("snippet") or ""),
            source_url=item.get("url"),
            retrieval_source="web",
            score=score_value,
        )
        documents.append(doc.model_dump(mode="json"))

    return {"query": query, "documents": documents, "warning": None}
