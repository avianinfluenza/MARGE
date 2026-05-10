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

    def __init__(self, llm: "ChatModel", system_prompt: str | None = None) -> None:
        from beeai_framework.memory import UnconstrainedMemory

        self._llm = llm
        self._system_prompt = system_prompt or _SYSTEM_PROMPT_PATH.read_text()
        self._memory = UnconstrainedMemory()
        self._initialized = False

    @classmethod
    def from_env(cls) -> "MedicalExpertAgent":
        """Build with the LLM configured for `Role.MEDICAL_EXPERT` in .env."""
        from packages.llm_provider.client import build_chat_model_for_role
        from packages.llm_provider.settings import Role

        return cls(llm=build_chat_model_for_role(Role.MEDICAL_EXPERT))

    @property
    def llm(self) -> "ChatModel":
        return self._llm

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        from beeai_framework.backend.message import SystemMessage

        await self._memory.add(SystemMessage(self._system_prompt))
        self._initialized = True

    @staticmethod
    def _format_findings(findings: dict[str, Any]) -> str:
        if not findings:
            return ""
        try:
            body = json.dumps(findings, indent=2, ensure_ascii=False, default=str)
        except Exception:
            body = str(findings)
        return f"\n\nClinical context:\n```json\n{body}\n```"

    async def consult(
        self, question: str, findings: dict[str, Any]
    ) -> MedicalExpertResponse:
        from beeai_framework.backend.message import AssistantMessage, UserMessage

        await self._ensure_initialized()

        user_msg = f"{question}{self._format_findings(findings)}"
        await self._memory.add(UserMessage(user_msg))

        result = await self._llm.run(self._memory.messages)
        text = (
            result.get_text_content()
            if hasattr(result, "get_text_content")
            else str(result)
        ) or ""

        await self._memory.add(AssistantMessage(text))

        # Citations: empty for now. When a `search_*` tool is wired in a
        # later slice, the consult flow turns into an internal mini-agent
        # loop that populates citations from retrieved documents.
        return MedicalExpertResponse(reasoning=text, citations=[])
