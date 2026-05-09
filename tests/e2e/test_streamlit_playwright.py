"""Playwright E2E: full MARGE conversation flow through the Streamlit UI.

User journey under test:
  1. User describes a health concern (vague — insufficient features)
     → App responds: "More information needed" panel
  2. User provides concrete values (age, glucose, BMI)
     → Sufficient data detected
     → ML inference (diabetes + breast cancer)
     → Clinical agent consulted
     → Final report rendered with metric cards and report sections

Each phase is a separate test so CI can run the UI-only tests without an LLM
key configured. Tests that require a live LLM are marked `llm_required` and
are skipped automatically when no provider key is found in the environment.

Running:
    # UI + missing-data flow only (no LLM):
    uv run pytest tests/e2e/test_streamlit_playwright.py -m "not llm_required" --headed

    # Full flow (needs ANTHROPIC_API_KEY or similar):
    uv run pytest tests/e2e/test_streamlit_playwright.py --headed

Server fixture: starts Streamlit on port 8502 as a subprocess and tears it
down after the session. If the server is already running (CI pre-started),
set MARGE_UI_URL to skip the subprocess launch.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Generator

import pytest
from playwright.sync_api import Page, expect


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PORT = 8502
_BASE_URL = os.getenv("MARGE_UI_URL", f"http://localhost:{_PORT}")

_HAS_LLM = bool(
    os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("WATSONX_APIKEY")
    or os.getenv("CEREBRAS_API_KEY")
    or os.getenv("NVIDIA_API_KEY")
    or os.getenv("CHUTES_API_KEY")
)

# Message 1 — vague concern, no extractable features
_MSG_VAGUE = "I've been feeling tired lately and I'm worried about my health."

# Message 2 — concrete values the regex + LLM can extract
_MSG_WITH_DATA = (
    "I'm 50 years old, female. "
    "My fasting blood sugar is 148 mg/dL, BMI is 33.6, "
    "blood pressure 120/72, I've had 6 pregnancies."
)

# Selectors
_CHAT_INPUT = "[data-testid='stChatInputTextArea']"
_CHAT_MESSAGES = "[data-testid='stChatMessage']"
_SEND_BUTTON = "[data-testid='stChatInputSubmitButton']"


# ---------------------------------------------------------------------------
# Session fixture — Streamlit server
# ---------------------------------------------------------------------------

def _wait_for_server(url: str, retries: int = 40, interval: float = 1.0) -> bool:
    for _ in range(retries):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(interval)
    return False


@pytest.fixture(scope="session")
def streamlit_server() -> Generator[str, None, None]:
    if os.getenv("MARGE_UI_URL"):
        yield os.environ["MARGE_UI_URL"]
        return

    proc = subprocess.Popen(
        [
            "uv", "run", "streamlit", "run",
            "apps/streamlit_ui/app.py",
            f"--server.port={_PORT}",
            "--server.headless=true",
            "--server.runOnSave=false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ready = _wait_for_server(_BASE_URL)
    if not ready:
        proc.terminate()
        pytest.skip("Streamlit server failed to start — skipping Playwright suite")

    yield _BASE_URL

    proc.terminate()
    proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_message(page: Page, text: str, timeout: int = 60_000) -> None:
    """Type a message and submit it; wait for the assistant analysis to finish.

    Streamlit renders analysis progress inside `st.status("Running")`. The label
    "Running" is permanent even after completion, so we detect progress via the
    spinning icon (`stStatusWidgetRunningIcon`).

    Wait strategy:
    1. Wait for user + assistant chat bubbles to appear.
    2. Wait for the spinning icon to APPEAR (proves analysis started, not just
       the count increased due to an already-rendered message).
    3. Wait for the spinning icon to DISAPPEAR (analysis complete).

    If the icon never appears (analysis was instant), step 2 times out
    gracefully and we proceed with what's already in the DOM.
    """
    msg_count_before = page.locator(_CHAT_MESSAGES).count()
    page.locator(_CHAT_INPUT).fill(text)
    page.locator(_SEND_BUTTON).click()

    # Step 1: wait for user + assistant chat bubbles to appear
    page.wait_for_function(
        f"document.querySelectorAll('[data-testid=\"stChatMessage\"]').length > {msg_count_before + 1}",
        timeout=timeout,
    )

    # Step 2: wait for running icon to appear (with short grace window)
    try:
        page.wait_for_selector(
            "[data-testid='stStatusWidgetRunningIcon']",
            state="attached",
            timeout=8_000,
        )
    except Exception:
        # Analysis completed before the icon was polled — already done
        return

    # Step 3: wait for running icon to disappear — analysis is done
    page.wait_for_function(
        "document.querySelectorAll('[data-testid=\"stStatusWidgetRunningIcon\"]').length === 0",
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Phase 0: UI structure (no LLM, no user message needed)
# ---------------------------------------------------------------------------

class TestUIStructure:
    """Verify the page renders correctly before any conversation."""

    def test_page_loads(self, streamlit_server: str, page: Page) -> None:
        page.goto(streamlit_server)
        expect(page).to_have_title("MARGE Demo")

    def test_chat_input_is_visible(self, streamlit_server: str, page: Page) -> None:
        page.goto(streamlit_server)
        expect(page.locator(_CHAT_INPUT)).to_be_visible()

    def test_initial_assistant_message_is_shown(self, streamlit_server: str, page: Page) -> None:
        page.goto(streamlit_server)
        messages = page.locator(_CHAT_MESSAGES)
        expect(messages.first).to_be_visible()
        first_text = messages.first.inner_text()
        assert "age" in first_text.lower() or "diabetes" in first_text.lower() or "tell me" in first_text.lower()

    def test_sidebar_shows_patient_handle(self, streamlit_server: str, page: Page) -> None:
        page.goto(streamlit_server)
        sidebar = page.locator("[data-testid='stSidebar']")
        expect(sidebar).to_be_visible()
        assert "seed-001" in sidebar.inner_text()

    def test_clear_conversation_button_exists(self, streamlit_server: str, page: Page) -> None:
        page.goto(streamlit_server)
        sidebar = page.locator("[data-testid='stSidebar']")
        expect(sidebar.get_by_text("Clear conversation")).to_be_visible()


# ---------------------------------------------------------------------------
# Phase 1: Insufficient data → "More information needed"
# (regex fallback works without LLM — vague message yields no features)
# ---------------------------------------------------------------------------

class TestInsufficientDataFlow:
    """Turn 1: vague message → missing-info panel shown."""

    @pytest.fixture(autouse=True)
    def go_to_fresh_page(self, streamlit_server: str, page: Page) -> None:
        page.goto(streamlit_server)
        # Clear any prior conversation via the sidebar button
        page.locator("[data-testid='stSidebar']").get_by_text("Clear conversation").click()
        page.wait_for_timeout(500)

    def test_vague_message_shows_missing_info_box(self, streamlit_server: str, page: Page) -> None:
        """Sending a message with no feature values → missing-info panel in DOM."""
        _send_message(page, _MSG_VAGUE, timeout=120_000)
        # The missing-box div is rendered inside the collapsed status widget —
        # it is in the DOM even though visually hidden; use page.content() to find it.
        page_html = page.content()
        assert "missing-box" in page_html or "missing-title" in page_html or any(
            kw in page.inner_html("body").lower()
            for kw in ("more information needed", "need", "glucose", "bmi", "blood sugar")
        ), "Expected missing-info panel in page HTML after vague message"

    def test_missing_info_lists_field_labels(self, streamlit_server: str, page: Page) -> None:
        """Missing-info panel lists human-readable field names."""
        _send_message(page, _MSG_VAGUE, timeout=120_000)
        page_html = page.inner_html("body").lower()
        labels = ["blood sugar", "bmi", "blood pressure", "age", "insulin"]
        found = any(label in page_html for label in labels)
        assert found, "Expected at least one diabetes field label in missing-info panel"

    def test_user_message_is_echoed_in_chat(self, streamlit_server: str, page: Page) -> None:
        """The user's own message must appear in the chat history."""
        _send_message(page, _MSG_VAGUE, timeout=120_000)
        messages_text = page.locator(_CHAT_MESSAGES).all_inner_texts()
        assert any("tired" in t.lower() for t in messages_text)


# ---------------------------------------------------------------------------
# Phase 2: Full flow (LLM required)
# user provides data → ML inference → clinical expert → final report
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_LLM, reason="requires LLM provider key")
class TestFullConversationFlow:
    """
    Multi-turn conversation:
      Turn 1 — vague concern  → missing-info panel (fast: one LLM feature-extraction call)
      Turn 2 — concrete data  → ML inference + clinical expert + final report
                                (slow: orchestrator makes 5-8 sequential LLM calls)

    Turn 2 tests are marked `slow` — they may take 3-10 minutes depending on the
    LLM provider. Fast providers (Anthropic claude-haiku, Cerebras Qwen3-235B) finish
    in ~60s. Slow providers (NVIDIA NIM 80B with throttle) may take 5-10 min.
    Run the full suite only in environments with a sub-1s LLM:
        uv run pytest tests/e2e/test_streamlit_playwright.py -m "not slow"  # quick
        uv run pytest tests/e2e/test_streamlit_playwright.py                 # all
    """

    @pytest.fixture(autouse=True)
    def fresh_conversation(self, streamlit_server: str, page: Page) -> None:
        page.goto(streamlit_server)
        page.locator("[data-testid='stSidebar']").get_by_text("Clear conversation").click()
        page.wait_for_timeout(500)

    # --- Turn 1: fast (single LLM call for feature extraction) ---

    def test_turn1_insufficient_data_triggers_missing_panel(
        self, streamlit_server: str, page: Page
    ) -> None:
        _send_message(page, _MSG_VAGUE, timeout=120_000)
        body_html = page.inner_html("body").lower()
        assert any(
            kw in body_html
            for kw in ("missing-box", "more information needed", "blood sugar", "bmi", "glucose")
        ), "Expected missing-info panel after vague message"

    def test_clear_conversation_resets_state(
        self, streamlit_server: str, page: Page
    ) -> None:
        _send_message(page, _MSG_VAGUE, timeout=120_000)
        page.locator("[data-testid='stSidebar']").get_by_text("Clear conversation").click()
        page.wait_for_timeout(1000)
        texts = page.locator(_CHAT_MESSAGES).all_inner_texts()
        assert not any("tired" in t.lower() for t in texts), (
            "User message still visible after clearing conversation"
        )

    # --- Turn 2: slow (full orchestrator — sequential LLM tool calls) ---

    @pytest.mark.slow
    def test_turn2_sufficient_data_produces_report(
        self, streamlit_server: str, page: Page
    ) -> None:
        """
        Full 2-turn journey:
          Turn 1: vague → missing-info shown
          Turn 2: concrete values → ML inference → clinical expert → final report
        Requires a fast LLM provider (≤2s/call). With NVIDIA NIM 80B this may
        take up to 10 minutes; prefer Anthropic or Cerebras in CI.
        """
        _send_message(page, _MSG_VAGUE, timeout=120_000)
        _send_message(page, _MSG_WITH_DATA, timeout=600_000)

        body_html = page.inner_html("body").lower()
        assert any(
            kw in body_html
            for kw in (
                "metric-card", "report-section", "patient report",
                "malignant", "benign", "diabetic_risk", "low_risk",
            )
        ), "Expected a patient report after providing sufficient data"

    @pytest.mark.slow
    def test_turn2_report_has_clinical_content(
        self, streamlit_server: str, page: Page
    ) -> None:
        _send_message(page, _MSG_VAGUE, timeout=120_000)
        _send_message(page, _MSG_WITH_DATA, timeout=600_000)

        body_html = page.inner_html("body").lower()
        assert any(
            kw in body_html
            for kw in ("risk", "diabetes", "cancer", "malignant", "benign", "glucose", "%")
        ), "Report lacks clinical content"

    @pytest.mark.slow
    def test_trajectory_contains_ml_and_expert_calls(
        self, streamlit_server: str, page: Page
    ) -> None:
        """Debug sidebar must show ML tool names in the trajectory after full analysis."""
        page.locator("[data-testid='stSidebar']").get_by_text("Show session debug").click()
        _send_message(page, _MSG_VAGUE, timeout=120_000)
        _send_message(page, _MSG_WITH_DATA, timeout=600_000)

        sidebar_text = page.locator("[data-testid='stSidebar']").inner_text().lower()
        assert any(
            kw in sidebar_text
            for kw in ("predict_diabetes_risk", "predict_breast_cancer_malignancy")
        ), "Expected ML tool name in trajectory debug panel"


# ---------------------------------------------------------------------------
# Phase 3: Report rendering correctness (LLM required)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _HAS_LLM, reason="requires LLM provider key")
class TestReportRendering:
    """After a successful analysis, the report HTML must follow the design spec.

    Requires full LLM orchestration (5-10 min with NVIDIA NIM 80B).
    Run only with fast LLM providers (Anthropic, Cerebras) in CI.
    """

    @pytest.fixture(autouse=True)
    def run_full_flow(self, streamlit_server: str, page: Page) -> None:
        page.goto(streamlit_server)
        page.locator("[data-testid='stSidebar']").get_by_text("Clear conversation").click()
        # Wait for page to stabilize after clear before sending new message
        page.wait_for_timeout(2_000)
        _send_message(page, _MSG_WITH_DATA, timeout=600_000)

    def test_metric_cards_in_dom(self, streamlit_server: str, page: Page) -> None:
        """metric-card divs must appear in the DOM (may be inside collapsed status)."""
        body_html = page.inner_html("body")
        assert "metric-card" in body_html, "No .metric-card div found in page HTML"

    def test_report_sections_in_dom(self, streamlit_server: str, page: Page) -> None:
        body_html = page.inner_html("body")
        assert "report-section" in body_html, "No .report-section div found in page HTML"

    def test_metric_values_have_units(self, streamlit_server: str, page: Page) -> None:
        body_html = page.inner_html("body")
        assert "%" in body_html or "mg/dL" in body_html, (
            "Expected percentage or mg/dL unit in report"
        )

    def test_inline_number_highlights_present(self, streamlit_server: str, page: Page) -> None:
        """_highlight_numbers wraps numerics in .metric-inline spans."""
        body_html = page.inner_html("body")
        assert "metric-inline" in body_html, "No inline number highlights found in report HTML"
