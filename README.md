# MARGE

Multi-agent ML-Reasoning Guidance Engine — a clinical diagnosis assist system that wraps niche tabular ML models behind a dynamic LLM orchestrator, with a separate medical expert sub-agent for clinical reasoning.

See [`overview.md`](./overview.md) for the concept and [`architecture.md`](./architecture.md) for design intent + folder structure.

## Status

- **ML stack** (XGBoost + CatBoost over MCP): two models registered, drop-in pattern verified.
- **BeeAI orchestrator**: assembled (`RequirementAgent`, 5 local tools + 2 MCP-discovered ML tools), protocol middleware enforced; live LLM tested against NVIDIA NIM (Qwen3.5-397B).
- **LLM provider abstraction** (`packages/llm_provider/`): five providers (Anthropic, watsonx, Cerebras, NVIDIA NIM, Chutes) + per-role routing (orchestrator vs medical expert) + opt-in `FallbackChatModel`.
- **Medical expert sub-agent**: LLM-backed `MedicalExpertAgent` wired via `MEDICAL_EXPERT_*`; optional Tavily web RAG injects citable medical context into expert consultations.
- **Streamlit UI**: not started.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Create the venv and install core dependencies
uv sync

# 2. Train the demo ML models (writes joblib artifacts under services/ml_mcp_server/artifacts/)
uv run python -m packages.ml_training.train_breast_cancer
uv run python -m packages.ml_training.train_diabetes

# 3. Smoke test: walks every registered model and verifies direct call == MCP call
uv run python scripts/smoke_test.py

# 4. Run the full unit + integration test suite
uv run pytest tests/ -v

# 5. (optional, requires LLM API keys) Live end-to-end smoke through the BeeAI orchestrator
cp .env.example .env  # then paste your provider keys
uv sync --extra orchestrator
uv run python scripts/orchestrator_smoke.py

# 5b. Step-by-step diagnostic of the orchestrator pipeline (each phase printed)
uv run python -u scripts/diag_agent.py
```

Optional dependency groups (install when working on those layers):

```bash
uv sync --extra orchestrator   # BeeAI Framework + watsonx.ai + anthropic
uv sync --extra ui             # Streamlit + matplotlib + plotly
uv sync --extra medical-kb     # Chroma + sentence-transformers + Tavily web RAG
uv sync --extra dev            # pytest + ruff
```

For medical-expert web RAG, set `TAVILY_API_KEY` in `.env`. Disable it with
`MARGE_ENABLE_WEB_RAG=false`.

## Layout

```
apps/         user-facing entrypoints (orchestrator, streamlit_ui)
services/     long-running components (MCP servers, sub-agents)
packages/     shared libraries (schemas, llm_provider, medical_kb, eval)
scripts/      operational scripts (smoke test, training, dev runner)
tests/        unit / integration / e2e
```

The orchestrator (`apps/orchestrator/`) only depends on `services/` and `packages/`. Services do not depend on each other. See `architecture.md` §4 for the full layering rules.
