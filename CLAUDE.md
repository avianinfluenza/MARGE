# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

**MARGE** (Multi-agent ML-Reasoning Guidance Engine) — a clinical ML orchestration system where a BeeAI-powered orchestrator agent selects niche tabular ML models, collects their predictions + SHAP XAI scores, consults a medical expert sub-agent for clinical reasoning, and produces a sourced final report. The orchestrator never produces medical claims from its own knowledge.

## Commands

```bash
# Install all dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_enforce_protocol.py -v

# Run tests by marker
uv run pytest -m unit
uv run pytest -m integration

# Lint
uv run ruff check .
uv run ruff format .

# Start the ML MCP server (stdio transport)
uv run python -m services.ml_mcp_server.server

# Train all ML artifacts (run once before starting the server)
uv run python -m packages.ml_training.train_diabetes
uv run python -m packages.ml_training.train_breast_cancer

# Run the Streamlit UI
uv run streamlit run apps/streamlit_ui/app.py
```

## Architecture

### Layering Rules (strictly enforced)

```
apps/  →  services/  →  packages/
```

1. `apps/` depends on `services/` and `packages/`. Never the reverse.
2. `services/` depend only on `packages/`. Services are independent — `ml_mcp_server` cannot import from `medical_expert_agent`.
3. `packages/schemas/` is the only module imported everywhere.
4. `apps/orchestrator/` accesses the medical expert only through the `consult_expert` tool — never directly.
5. `services/medical_expert_agent/` never reads patient records (orchestrator-only scope).

### How New ML Models Are Added

The MCP server auto-discovers models at startup — adding a model requires **one file only**:

1. Create `services/ml_mcp_server/models/your_model.py`
2. Define a class that subclasses `MLModel` (from `models/_base.py`) or `DynamicMLAgent` (from `models/_agent_factory.py`)
3. Implement the four required members: `name`, `metadata`, `input_schema`, `predict()`, `sample_inputs()`
4. `registry.py` imports every non-`_` prefixed module in `models/`, finds the subclass, instantiates it, and the MCP server exposes it as a tool

The `DynamicMLAgent` factory pattern (`models/_agent_factory.py`) handles the common case: pass an `AgentConfig` with feature names and artifact path; the factory builds a Pydantic input schema dynamically, runs K-Fold XGBoost ensemble training, sets up SHAP explainability, and serializes to `.joblib`. If the artifact exists on disk, it loads directly (init-or-train lifecycle).

For CSV-based datasets, `ingest_csv_and_build_agent()` in `_agent_factory.py` reads a CSV, auto-configures an `AgentConfig`, trains, and generates the drop-in wrapper class file.

### Orchestrator Protocol Enforcement

Two BeeAI middleware hooks enforce safety structurally (not via prompting):

- **`apps/orchestrator/middleware/enforce_protocol.py`** — blocks `final_report` unless at least one ML tool result and one `consult_medical_expert` result are in the current trajectory.
- **`services/medical_expert_agent/middleware/enforce_citation.py`** — rejects expert responses that contain clinical claims without citations, forcing a re-answer with retrieved sources.

Valid terminal actions that bypass the ML+expert precondition: `abstain` and `ask_user_back`.

### MCP Surfaces

| Server | Transport | Who uses it |
|---|---|---|
| `services/ml_mcp_server/` | stdio | Orchestrator (via BeeAI MCP client) |
| `services/patient_data_mcp_server/` | stdio | Orchestrator only |

### LLM Provider

`packages/llm_provider/client.py` exposes `get_llm()` — returns Granite via watsonx.ai by default, falls back to Anthropic Claude if `LLM_PROVIDER=anthropic` is set. Neither `apps/` nor `services/` import raw watsonx/Anthropic SDKs directly.

### Key Schemas (packages/schemas/)

- `prediction.py` — `Prediction`, `XAIScore`, `ModelMetadata`
- `patient.py` — `PatientRecord`, `ClinicalFeature`
- `retrieval.py` — `RetrievedDocument`, `Citation`, `MedicalExpertResponse`

### ML Artifacts

Serialized `.joblib` files live in `services/ml_mcp_server/artifacts/` (gitignored). Each artifact bundles the fitted ensemble, feature names, and test metrics. Missing artifacts trigger auto-training on first server start (for `DynamicMLAgent` subclasses that call `self.train()` in `__init__`).
