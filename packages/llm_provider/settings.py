"""Environment-driven settings for the LLM provider abstraction.

Two layers:

1. `LLMSettings` — a single provider's configuration. Selected by
   `LLM_PROVIDER` env var; provider-specific keys (e.g. `CEREBRAS_API_KEY`)
   resolve to fields on this dataclass.

2. `RoleConfig` — per-role primary + optional fallback. Each agent role
   (orchestrator vs medical expert) has its own `{ROLE}_PRIMARY` and
   `{ROLE}_FALLBACK` env var pointing to a provider name. The provider's
   credentials still come from the same `{PROVIDER}_API_KEY` env var, so a
   single key serves both roles when both happen to share a provider.

Env vars:
- LLM_PROVIDER         — "anthropic" (default), "watsonx", "cerebras",
                         "nvidia", "chutes", "featherless"
- ANTHROPIC_API_KEY / ANTHROPIC_MODEL_ID
- WATSONX_API_KEY / WATSONX_PROJECT_ID / WATSONX_URL / WATSONX_MODEL_ID
- CEREBRAS_API_KEY / CEREBRAS_MODEL_ID / CEREBRAS_BASE_URL
- NVIDIA_API_KEY / NVIDIA_MODEL_ID / NVIDIA_BASE_URL
- CHUTES_API_KEY / CHUTES_MODEL_ID / CHUTES_BASE_URL
- FEATHERLESS_API_KEY / FEATHERLESS_MODEL_ID / FEATHERLESS_BASE_URL
- ORCHESTRATOR_PRIMARY / ORCHESTRATOR_FALLBACK
- MEDICAL_EXPERT_PRIMARY / MEDICAL_EXPERT_FALLBACK
"""

import os
from dataclasses import dataclass
from enum import Enum

_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_WATSONX_MODEL = "ibm/granite-3-8b-instruct"

_DEFAULT_CEREBRAS_MODEL = "qwen-3-235b-a22b-instruct-2507"
_DEFAULT_CEREBRAS_URL = "https://api.cerebras.ai/v1"

# NOTE: NVIDIA NIM's hosted Qwen3.5-397B is currently capacity-throttled —
# even single short prompts time out. The 80B "next" variant is responsive
# (0.5s round trip) and supports tool calling. Override with
# NVIDIA_MODEL_ID=qwen/qwen3.5-397b-a17b once NIM stabilises.
_DEFAULT_NVIDIA_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
_DEFAULT_NVIDIA_URL = "https://integrate.api.nvidia.com/v1"

_DEFAULT_CHUTES_MODEL = "moonshotai/Kimi-K2.5-TEE"
_DEFAULT_CHUTES_URL = "https://llm.chutes.ai/v1"

_DEFAULT_FEATHERLESS_MODEL = "moonshotai/Kimi-K2.5"
_DEFAULT_FEATHERLESS_URL = "https://api.featherless.ai/v1"


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    WATSONX = "watsonx"
    CEREBRAS = "cerebras"
    NVIDIA = "nvidia"
    CHUTES = "chutes"
    FEATHERLESS = "featherless"


class Role(str, Enum):
    ORCHESTRATOR = "orchestrator"
    MEDICAL_EXPERT = "medical_expert"


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration for a single provider."""

    provider: Provider
    model_id: str
    api_key: str | None = None
    base_url: str | None = None
    project_id: str | None = None  # watsonx-only

    @classmethod
    def from_env(cls, provider: Provider | str | None = None) -> "LLMSettings":
        """Build settings for `provider` (or LLM_PROVIDER env var, default anthropic)."""
        provider_str = (
            provider if isinstance(provider, Provider)
            else (provider or os.getenv("LLM_PROVIDER", Provider.ANTHROPIC.value)).lower()
        )
        try:
            p = Provider(provider_str) if not isinstance(provider_str, Provider) else provider_str
        except ValueError as e:
            raise ValueError(
                f"Unknown LLM provider: {provider_str!r}. "
                f"Expected one of {[v.value for v in Provider]}."
            ) from e

        return _BUILDERS[p]()


def _anthropic() -> LLMSettings:
    return LLMSettings(
        provider=Provider.ANTHROPIC,
        model_id=os.getenv("ANTHROPIC_MODEL_ID", _DEFAULT_ANTHROPIC_MODEL),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def _watsonx() -> LLMSettings:
    return LLMSettings(
        provider=Provider.WATSONX,
        model_id=os.getenv("WATSONX_MODEL_ID", _DEFAULT_WATSONX_MODEL),
        api_key=os.getenv("WATSONX_API_KEY"),
        base_url=os.getenv("WATSONX_URL"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
    )


def _cerebras() -> LLMSettings:
    return LLMSettings(
        provider=Provider.CEREBRAS,
        model_id=os.getenv("CEREBRAS_MODEL_ID", _DEFAULT_CEREBRAS_MODEL),
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url=os.getenv("CEREBRAS_BASE_URL", _DEFAULT_CEREBRAS_URL),
    )


def _nvidia() -> LLMSettings:
    return LLMSettings(
        provider=Provider.NVIDIA,
        model_id=os.getenv("NVIDIA_MODEL_ID", _DEFAULT_NVIDIA_MODEL),
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_BASE_URL", _DEFAULT_NVIDIA_URL),
    )


def _chutes() -> LLMSettings:
    return LLMSettings(
        provider=Provider.CHUTES,
        model_id=os.getenv("CHUTES_MODEL_ID", _DEFAULT_CHUTES_MODEL),
        api_key=os.getenv("CHUTES_API_KEY"),
        base_url=os.getenv("CHUTES_BASE_URL", _DEFAULT_CHUTES_URL),
    )


def _featherless() -> LLMSettings:
    return LLMSettings(
        provider=Provider.FEATHERLESS,
        model_id=os.getenv("FEATHERLESS_MODEL_ID", _DEFAULT_FEATHERLESS_MODEL),
        api_key=os.getenv("FEATHERLESS_API_KEY"),
        base_url=os.getenv("FEATHERLESS_BASE_URL", _DEFAULT_FEATHERLESS_URL),
    )


_BUILDERS = {
    Provider.ANTHROPIC: _anthropic,
    Provider.WATSONX: _watsonx,
    Provider.CEREBRAS: _cerebras,
    Provider.NVIDIA: _nvidia,
    Provider.CHUTES: _chutes,
    Provider.FEATHERLESS: _featherless,
}


@dataclass(frozen=True)
class RoleConfig:
    """Per-role provider routing: primary + optional fallback."""

    role: Role
    primary: LLMSettings
    fallback: LLMSettings | None = None

    @classmethod
    def from_env(cls, role: Role) -> "RoleConfig":
        """Read `{ROLE}_PRIMARY` and `{ROLE}_FALLBACK` from env."""
        prefix = role.value.upper()
        primary_name = os.getenv(f"{prefix}_PRIMARY")
        fallback_name = os.getenv(f"{prefix}_FALLBACK")

        primary = LLMSettings.from_env(primary_name)  # falls back to LLM_PROVIDER default
        fallback = LLMSettings.from_env(fallback_name) if fallback_name else None
        return cls(role=role, primary=primary, fallback=fallback)
