"""Medical expert sub-agent.

Two implementations:

- `StubMedicalExpert`: deterministic fixed-response stub used in unit tests
  and for cheap local sanity checks. Sync.
- `MedicalExpertAgent`: real LLM-backed expert with its own ChatModel and
  its own conversation memory (separate from the orchestrator's). Async.

Both expose `consult(question, findings) -> MedicalExpertResponse`. The
orchestrator's `consult_medical_expert` tool is async-aware and handles
either.

Role: the expert reasons in clinical terms only. It does NOT know about
the orchestrator's ML predictors and never recommends "use model X" — see
`services/medical_expert_agent/system_prompt.md`.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packages.schemas.retrieval import (
    Citation,
    MedicalExpertResponse,
    RetrievedDocument,
)

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"

# ---------------- Stub (sync, for tests) ----------------

_STUB_DOC = RetrievedDocument(
    title="WHO clinical guideline (stub)",
    snippet=(
        "Stub citation used during early development. The orchestrator's "
        "downstream consumers can rely on the same shape they will see when "
        "the real medical_expert sub-agent is wired up."
    ),
    source_url="https://stub.example.org/who-guideline",
    retrieval_source="local_kb",
)

_STUB_REASONING = (
    "Based on the supplied findings (stub response): the patient's profile "
    "warrants further targeted screening. Clinical judgement should "
    "incorporate the ML findings, the patient's history, and standard "
    "guideline-driven thresholds. This is a stub response — replace with the "
    "real medical_expert sub-agent before any clinical use."
)


class StubMedicalExpert:
    """Returns a fixed MedicalExpertResponse without invoking any LLM."""

    def consult(self, question: str, findings: dict[str, Any]) -> MedicalExpertResponse:
        return MedicalExpertResponse(
            reasoning=_STUB_REASONING,
            citations=[Citation(document=_STUB_DOC, supporting_quote=None)],
        )


# ---------------- Real expert (async, BeeAI ChatModel) ----------------


class MedicalExpertAgent:
    """LLM-backed medical expert sub-agent with its own context.

    - Holds its own `ChatModel` (typically a stronger / different model
      than the orchestrator's).
    - Maintains its own `UnconstrainedMemory` so the expert can remember
      previous consultations within a session (helpful when the
      orchestrator probes the same case from multiple angles).
    - System prompt comes from `services/medical_expert_agent/system_prompt.md`
      and pins the role boundary (clinical reasoning only, no ML awareness).

    Usage:
        expert = MedicalExpertAgent.from_env()
        async with ...:
            response = await expert.consult("...", {...})
    """

    def __init__(
        self,
        llm: "ChatModel",
        system_prompt: str | None = None,
        *,
        web_search: Callable[[str, int], list[RetrievedDocument]] | None = None,
        enable_web_search: bool = True,
        max_web_results: int = 3,
        max_web_search_calls_per_turn: int = 1,
    ) -> None:
        from beeai_framework.memory import UnconstrainedMemory

        self._llm = llm
        self._system_prompt = system_prompt or _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        self._memory = UnconstrainedMemory()
        self._event_sink: Callable[[dict[str, Any]], None] | None = None
        self._web_search = web_search
        self._enable_web_search = enable_web_search
        self._max_web_results = max_web_results
        self._max_web_search_calls_per_turn = max(0, max_web_search_calls_per_turn)
        self._web_search_calls_this_turn = 0
        self._turn_limit_active = False

    @classmethod
    def from_env(cls) -> "MedicalExpertAgent":
        """Build with the LLM configured for `Role.MEDICAL_EXPERT` in .env."""
        from packages.llm_provider.client import build_chat_model_for_role
        from packages.llm_provider.settings import Role
        from services.medical_expert_agent.tools.search_web import medical_web_max_results

        return cls(
            llm=build_chat_model_for_role(Role.MEDICAL_EXPERT),
            max_web_results=medical_web_max_results(),
        )

    @property
    def llm(self) -> "ChatModel":
        return self._llm

    def set_event_sink(
        self, sink: Callable[[dict[str, Any]], None] | None
    ) -> None:
        """Attach a per-turn trace sink for expert-internal events."""

        self._event_sink = sink
        self._turn_limit_active = sink is not None
        if sink is not None:
            self._web_search_calls_this_turn = 0

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        self._event_sink(event)

    @staticmethod
    def _format_findings(findings: dict[str, Any]) -> str:
        if not findings:
            return ""
        try:
            body = json.dumps(findings, indent=2, ensure_ascii=False, default=str)
        except Exception:
            body = str(findings)
        return f"\n\nClinical context:\n```json\n{body}\n```"

    @staticmethod
    def _fallback_citations(documents: list[RetrievedDocument]) -> list[Citation]:
        if documents:
            return [
                Citation(document=doc, supporting_quote=doc.snippet or None)
                for doc in documents
            ]
        return [Citation(document=_STUB_DOC, supporting_quote=None)]

    @staticmethod
    def _parse_response_text(
        text: str,
        fallback_citations: list[Citation],
    ) -> MedicalExpertResponse:
        try:
            payload = json.loads(text)
        except Exception:
            return MedicalExpertResponse(
                reasoning=text,
                citations=fallback_citations,
            )

        citations: list[Citation] = []
        for raw in payload.get("citations") or []:
            try:
                citations.append(Citation.model_validate(raw))
            except Exception:
                continue

        return MedicalExpertResponse(
            reasoning=payload.get("reasoning") or text,
            citations=citations or fallback_citations,
            abstained=bool(payload.get("abstained", False)),
            abstain_reason=payload.get("abstain_reason"),
        )

    @staticmethod
    def _result_text(result: Any) -> str:
        structured = getattr(result, "output_structured", None)
        text = getattr(structured, "response", None)
        if text:
            return text
        answer = getattr(result, "answer", None)
        text = getattr(answer, "text", None)
        if text:
            return text
        if hasattr(result, "get_text_content"):
            try:
                return result.get_text_content() or ""
            except Exception:
                pass
        return str(result)

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(k): MedicalExpertAgent._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [MedicalExpertAgent._to_jsonable(x) for x in obj]
        if hasattr(obj, "to_json_safe"):
            try:
                return MedicalExpertAgent._to_jsonable(obj.to_json_safe())
            except Exception:
                pass
        if hasattr(obj, "model_dump"):
            try:
                return MedicalExpertAgent._to_jsonable(obj.model_dump(mode="json"))
            except Exception:
                pass
        return repr(obj)[:1000]

    @staticmethod
    def _citations_from_tool_payload(payload: Any) -> list[Citation]:
        if not isinstance(payload, dict):
            return []
        docs = payload.get("documents")
        if not isinstance(docs, list):
            return []

        citations: list[Citation] = []
        for raw_doc in docs:
            try:
                doc = RetrievedDocument.model_validate(raw_doc)
            except Exception:
                continue
            citations.append(Citation(document=doc, supporting_quote=doc.snippet or None))
        return citations

    async def _search_medical_web_once_per_turn(
        self,
        query: str,
        max_results: int = 3,
    ) -> dict[str, Any]:
        from services.medical_expert_agent.tools.search_web import (
            _medical_web_include_domains,
            search_medical_web,
        )

        if self._web_search_calls_this_turn >= self._max_web_search_calls_per_turn:
            return {
                "query": query,
                "max_results": 0,
                "include_domains": _medical_web_include_domains(),
                "documents": [],
                "warning": (
                    "search_medical_web is limited to one actual web search per "
                    "user turn; this additional query was not executed."
                ),
                "skipped_due_to_turn_limit": True,
                "calls_used_this_turn": self._web_search_calls_this_turn,
                "max_calls_per_turn": self._max_web_search_calls_per_turn,
            }

        self._web_search_calls_this_turn += 1
        result = await search_medical_web(query=query, max_results=max_results)
        if isinstance(result, dict):
            result.setdefault("skipped_due_to_turn_limit", False)
            result["calls_used_this_turn"] = self._web_search_calls_this_turn
            result["max_calls_per_turn"] = self._max_web_search_calls_per_turn
        return result

    def _wire_tool_logging(
        self,
        tools: list[Any],
        citations: list[Citation],
        seen_citations: set[str],
    ) -> None:
        for tool in tools:

            def make_recorder(tool_name: str):
                pending: dict[str, Any] = {}

                def on_evt(data: Any, event: Any) -> None:
                    event_name = getattr(event, "name", "")
                    if event_name == "start":
                        pending["input"] = self._to_jsonable(getattr(data, "input", None))
                        self._emit_event(
                            {
                                "kind": "tool_call",
                                "agent": "expert",
                                "name": tool_name,
                                "input": pending["input"],
                            }
                        )
                    elif event_name == "success":
                        output = self._to_jsonable(getattr(data, "output", None))
                        for citation in self._citations_from_tool_payload(output):
                            key = (
                                citation.document.source_url
                                or citation.document.title
                                or citation.document.snippet
                            )
                            if key in seen_citations:
                                continue
                            seen_citations.add(key)
                            citations.append(citation)
                        self._emit_event(
                            {
                                "kind": "tool_output",
                                "agent": "expert",
                                "name": tool_name,
                                "input": pending.get("input"),
                                "output": output,
                                "success": True,
                            }
                        )
                        pending.clear()
                    elif event_name == "error":
                        err = getattr(data, "error", None)
                        self._emit_event(
                            {
                                "kind": "tool_output",
                                "agent": "expert",
                                "name": tool_name,
                                "input": pending.get("input"),
                                "error": f"{type(err).__name__}: {err}",
                                "success": False,
                            }
                        )
                        pending.clear()

                return on_evt

            try:
                tool.emitter.match("*", make_recorder(tool.name))
            except Exception:
                pass

    async def _consult_direct(
        self,
        question: str,
        findings: dict[str, Any],
    ) -> MedicalExpertResponse:
        """Compatibility path for simple LLM-only tests and injected RAG."""

        from beeai_framework.backend.message import SystemMessage, UserMessage

        user_msg = f"{question}{self._format_findings(findings)}"
        retrieved_docs: list[RetrievedDocument] = []
        if self._enable_web_search and self._web_search is not None:
            retrieved_docs = self._web_search(question, self._max_web_results)
            if retrieved_docs:
                context = [
                    doc.model_dump(mode="json")
                    for doc in retrieved_docs
                ]
                user_msg += (
                    "\n\nretrieved_context:\n```json\n"
                    f"{json.dumps(context, indent=2, ensure_ascii=False)}"
                    "\n```"
                )

        messages = [SystemMessage(self._system_prompt), UserMessage(user_msg)]
        result = await self._llm.run(messages)
        text = (
            result.get_text_content()
            if hasattr(result, "get_text_content")
            else str(result)
        ) or ""
        return self._parse_response_text(
            text,
            fallback_citations=self._fallback_citations(retrieved_docs),
        )

    async def consult(
        self, question: str, findings: dict[str, Any]
    ) -> MedicalExpertResponse:
        if self._web_search is not None or not self._enable_web_search:
            return await self._consult_direct(question, findings)

        from beeai_framework.agents.requirement import RequirementAgent

        user_msg = f"{question}{self._format_findings(findings)}"
        citations: list[Citation] = []
        seen_citations: set[str] = set()
        if not self._turn_limit_active:
            self._web_search_calls_this_turn = 0

        from services.medical_expert_agent.tools._adapter import expert_tools_as_beeai

        tools = expert_tools_as_beeai(
            {"search_medical_web": self._search_medical_web_once_per_turn}
        )
        self._wire_tool_logging(tools, citations, seen_citations)

        agent = RequirementAgent(
            llm=self._llm,
            memory=self._memory,
            tools=tools,
            requirements=[],
            name="MARGE Medical Expert",
            description=(
                "Clinical expert sub-agent with expert-only medical web search."
            ),
            instructions=self._system_prompt,
            final_answer_as_tool=False,
        )
        result = await agent.run(user_msg)
        return MedicalExpertResponse(
            reasoning=self._result_text(result),
            citations=citations,
        )


def build_medical_expert_agent() -> StubMedicalExpert | MedicalExpertAgent:
    """Build the configured expert, or a stub when no expert env is set."""

    if not os.getenv("MEDICAL_EXPERT_PRIMARY") and not os.getenv("LLM_PROVIDER"):
        return StubMedicalExpert()
    return MedicalExpertAgent.from_env()
