"""Factory for BeeAI ChatModel instances and a fallback wrapper.

`build_chat_model(settings)`         — single provider; raises on missing key
`build_chat_model_for_role(role)`    — reads RoleConfig; wraps with fallback
`FallbackChatModel`                  — try primary, fall back to secondary
                                        on any exception. Other attribute
                                        access is proxied to primary.
"""

from typing import TYPE_CHECKING, Any

from packages.llm_provider.settings import (
    LLMSettings,
    Provider,
    Role,
    RoleConfig,
)

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel


# ---------------------------------------------------------------------------
# Provider -> ChatModel builders
# ---------------------------------------------------------------------------


def build_chat_model(settings: LLMSettings) -> "ChatModel":
    builder = _BUILDERS[settings.provider]
    return builder(settings)


def _build_anthropic(s: LLMSettings) -> "ChatModel":
    if not s.api_key:
        raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
    from beeai_framework.adapters.anthropic.backend.chat import AnthropicChatModel

    return AnthropicChatModel(model_id=s.model_id, api_key=s.api_key, base_url=s.base_url)


def _build_watsonx(s: LLMSettings) -> "ChatModel":
    if not s.api_key:
        raise ValueError("WATSONX_API_KEY is required when LLM_PROVIDER=watsonx.")
    if not s.project_id:
        raise ValueError("WATSONX_PROJECT_ID is required when LLM_PROVIDER=watsonx.")
    if not s.base_url:
        raise ValueError("WATSONX_URL is required when LLM_PROVIDER=watsonx.")
    from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel

    return WatsonxChatModel(
        model_id=s.model_id,
        api_key=s.api_key,
        project_id=s.project_id,
        base_url=s.base_url,
    )


def _build_openai_compat(s: LLMSettings, key_var: str) -> "ChatModel":
    if not s.api_key:
        raise ValueError(f"{key_var} is required when LLM_PROVIDER={s.provider.value}.")
    from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel

    return OpenAIChatModel(model_id=s.model_id, api_key=s.api_key, base_url=s.base_url)


def _build_cerebras(s: LLMSettings) -> "ChatModel":
    return _build_openai_compat(s, "CEREBRAS_API_KEY")


def _build_nvidia(s: LLMSettings) -> "ChatModel":
    return _build_openai_compat(s, "NVIDIA_API_KEY")


def _build_chutes(s: LLMSettings) -> "ChatModel":
    return _build_openai_compat(s, "CHUTES_API_KEY")


_BUILDERS = {
    Provider.ANTHROPIC: _build_anthropic,
    Provider.WATSONX: _build_watsonx,
    Provider.CEREBRAS: _build_cerebras,
    Provider.NVIDIA: _build_nvidia,
    Provider.CHUTES: _build_chutes,
}


# ---------------------------------------------------------------------------
# Fallback wrapper
# ---------------------------------------------------------------------------


class FallbackChatModel:
    """Try primary; fall back to `fallback` on any exception during `run()`.

    Other attribute access (model_id, provider_id, parameters, ...) is proxied
    to `primary` so the wrapper looks like a regular ChatModel to BeeAI.
    """

    def __init__(self, primary: "ChatModel", fallback: "ChatModel") -> None:
        # Use object.__setattr__ to avoid invoking __getattr__ shenanigans.
        object.__setattr__(self, "_primary", primary)
        object.__setattr__(self, "_fallback", fallback)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when the attribute is not found normally.
        return getattr(object.__getattribute__(self, "_primary"), name)

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        primary = object.__getattribute__(self, "_primary")
        fallback = object.__getattribute__(self, "_fallback")
        try:
            return await primary.run(*args, **kwargs)
        except Exception as e:
            print(
                f"[llm_provider] primary {primary.model_id} failed: {e!r}; "
                f"falling back to {fallback.model_id}"
            )
            return await fallback.run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Role-aware factory
# ---------------------------------------------------------------------------


def build_chat_model_for_role(role: Role, *, with_fallback: bool = False) -> "ChatModel":
    """Build a ChatModel for `role`.

    By default returns the primary only. The fallback configuration is still
    read so callers can inspect `RoleConfig.from_env(role).fallback` and
    swap manually (one-line env change).

    Pass `with_fallback=True` to wrap with `FallbackChatModel`. NOTE: the
    wrapper is unit-tested against duck-typed models but not yet integrated
    with BeeAI's `Run` object protocol — direct use with `RequirementAgent`
    will fail. Use only with custom callers until the BeeAI integration is
    added in a future slice.
    """
    cfg = RoleConfig.from_env(role)
    primary_model = build_chat_model(cfg.primary)
    if not with_fallback or cfg.fallback is None:
        return primary_model
    fallback_model = build_chat_model(cfg.fallback)
    return FallbackChatModel(primary_model, fallback_model)  # type: ignore[return-value]
