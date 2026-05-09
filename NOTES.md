# Project notes

Operational notes that don't belong in `architecture.md` or `README.md`.

## Live-LLM verification status (May 2026)

### Provider-specific gotchas (root-caused via raw HTTP capture)

After capturing the actual request bodies BeeAI sends and replaying them with
`curl`, three OpenAI-compat-incompatibility issues surfaced. All are now fixed
in `packages/llm_provider/client._build_openai_compat`:

1. **`response_format=json_schema` instead of `tools`**.
   BeeAI's `RequirementAgent` defaults to `tool_call_fallback_via_response_format=True`,
   which packs every tool into one `anyOf` JSON-Schema and asks the model to
   produce structured output. NVIDIA NIM (and Cerebras) do not handle the
   nested `anyOf` reliably — requests hang or return malformed output.
   **Fix**: pass `tool_call_fallback_via_response_format=False` so BeeAI
   sends native OpenAI `tools=[...]`.

2. **`tool_choice="required"` not supported**.
   Cerebras / NVIDIA NIM / Chutes do not honour `tool_choice={"required"}`.
   When BeeAI receives plain text from a model it told to "must call a tool",
   it raises `ToolChoiceError`.
   **Fix**: pass `tool_choice_support={"auto","single","none"}` so BeeAI
   never asks for `required`.

3. **NVIDIA NIM 397B serving capacity**.
   `qwen/qwen3.5-397b-a17b` on free NIM credits is currently throttled
   to the point that even single short prompts time out. The same key
   gets sub-second responses from `qwen/qwen3-next-80b-a3b-instruct`,
   so `_DEFAULT_NVIDIA_MODEL` now points at the 80B variant. Override
   with `NVIDIA_MODEL_ID=qwen/qwen3.5-397b-a17b` once NIM stabilises.

### Slice "A" closed — full happy path live ✓

NVIDIA NIM (Qwen3-Next-80B) end-to-end (smoke v8):
```
Trajectory: get_patient_history → predict_breast_cancer_malignancy
            → predict_diabetes_risk → consult_medical_expert → final_report
[ok] final_report reached: True
ToolErrors: 0   tool_choice errors: 0
```

The `final_report` natural-language `response` correctly synthesised the ML
findings, the expert's validation, an ask-back follow-up, and the safety
reminder — all in one tool call (single-terminal refactor working as
designed).

### Status after fixes (slice "A" complete)

What works end-to-end (verified live):

- 5 providers register; `build_chat_model()` returns the right adapter.
- Direct `ChatModel.run([UserMessage(...)])` succeeds for Cerebras and
  NVIDIA NIM (80B variant) using native `tools=[...]`.
- The 7-tool surface (5 local + 2 ML MCP) is exposed on the agent.
- `orchestrator_agent` async-context-manager keeps the MCP session
  open for the lifetime of `agent.run()` — MCP tool calls succeed
  with no `ToolError`.
- `final_answer_as_tool=False` removes BeeAI's auto-final tool, so
  the only path to a user answer is our gated `final_report`.
- Emitter hook on each MCP tool records `predict_*` calls into the
  `ProtocolEnforcer` trajectory.
- Per-provider throttle subclass on `OpenAIChatModel._create` enforces
  inter-call gap (Cerebras 4.0s, NVIDIA 1.7s).

Live agent loop on NVIDIA NIM (Qwen3-Next-80B), trajectory observed:
- iter 1: `get_patient_history(seed-001)` ✓
- iter 2: `predict_breast_cancer_malignancy(...)` ✓ (recorded via emitter)
- iter 3: `predict_diabetes_risk(...)` ✓
- ✗ `consult_medical_expert` not called — Qwen 80B emitted the call
  as `<tool_call>...</tool_call>` text inside the answer body
  instead of an actual tool call (model-quality issue, not wiring)
- ✗ `final_report` consequently not reached

What's currently blocked (provider-side, not code):

- **Cerebras `csk-…` key has exhausted its rolling-60s quota window.**
  Direct `curl` to the API returns 429 immediately. Even with our 4.0s
  throttle, the very first agent call gets 429. Wait ~1 hour for the
  window to clear, or use a fresh key.
- **NVIDIA NIM `qwen3.5-397b-a17b`** still capacity-throttled — single
  `curl "hi"` times out. We default to `qwen3-next-80b-a3b-instruct`.
- **Chutes**: marketed as free, but the API returns 402 (TAO balance
  required). Configured for the day credits are added.

Next slice options (in priority order):

1. **Stronger orchestrator model** — once a stable Cerebras window opens
   or a fresh key arrives, Qwen3-235B's instruction following should call
   `consult_medical_expert` properly instead of describing it in text.
2. **Use BeeAI Requirements** — RequirementAgent has a `requirements`
   parameter that can declaratively encode "after `predict_*`, call
   `consult_medical_expert` before `final_report`". This deterministically
   forces the call regardless of model quality. Heavier change but
   architecture-aligned.
3. **Wire real medical expert sub-agent** — replace `StubMedicalExpert`
   with a BeeAI agent + Tavily/Exa search + `enforce_citation` middleware.
4. **Streamlit UI** — visual demo on top of the working orchestrator core.

What to add next slice:

1. **Per-provider rate limiter** in `packages/llm_provider/`: a small
   wrapper that enforces a min-gap between calls (Cerebras → 2.2s,
   NVIDIA → 1.6s). Should subclass or duck-type ChatModel and intercept
   `run()` to insert `asyncio.sleep` as needed.
2. **Retry-on-429 with exponential backoff**: BeeAI has `Retryable`
   utilities — wire them into the LLM adapter or agent so transient
   429s self-recover.
3. **Switch test path to Anthropic**: for development the hackathon
   doesn't need the IBM-stack endpoint — Anthropic Haiku in the dev
   loop, swap to NVIDIA/Cerebras only for the demo recording.
4. **Investigate the NVIDIA hang**: capture the exact HTTP request
   (headers + body) BeeAI sends on the second iteration, replay with
   `curl` to isolate whether it's a NIM-side issue or a BeeAI
   serialisation bug.

## Verification scripts

- `scripts/smoke_test.py` — ML model + MCP smoke (all green, in CI).
- `scripts/diag_agent.py` — step-by-step agent build (no tool use).
  Always passes — confirms single-iteration agent works.
- `scripts/diag_agent_tooluse.py` — tool-using agent + BeeAI emitter
  event tracing. Run unbuffered:
  ```bash
  uv run python -u scripts/diag_agent_tooluse.py > /tmp/marge.log 2>&1
  tail -F /tmp/marge.log
  ```
- `scripts/orchestrator_smoke.py` — full live demo path. Currently
  hits the multi-iteration issues above; will work once the rate
  limiter and retry layer land.

## Observation: BeeAI deprecations to address

- `ToolCallingAgent` is deprecated → migrated to `RequirementAgent` already.
- `agent.emitter.match("*.*", ...)` is deprecated → switch to `on(...)`
  in `diag_agent_tooluse.py`.
- `RequirementAgent`'s `Requirement` system could express the
  protocol invariants (ML-then-expert-then-final_report) declaratively
  and replace `enforce_protocol` middleware in a future refactor.
