"""Factory for BeeAI ChatModel instances and a fallback wrapper.

`build_chat_model(settings)`         — single provider; raises on missing key
`build_chat_model_for_role(role)`    — reads RoleConfig; wraps with fallback
`FallbackChatModel`                  — try primary, fall back to secondary
                                        on any exception. Other attribute
                                        access is proxied to primary.

Per-provider throttle: free-tier Cerebras and NVIDIA NIM enforce strict
RPM limits and reject burst calls. We subclass `OpenAIChatModel` and
override `_create` / `_create_stream` to insert an `asyncio.sleep` so
back-to-back agent iterations stay under the per-minute cap.
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any

from packages.llm_provider.settings import (
    LLMSettings,
    Provider,
    Role,
    RoleConfig,
)

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel


# Min seconds between consecutive requests per provider (free tier).
# Cerebras 30 RPM nominal but in practice the new-account quota appears
# tighter — bumped to 4.0s. NVIDIA NIM 40 RPM = 1.5s; padded to 1.7s.
# Featherless free tier allows only 1 concurrent request — overlapping
# calls return 429 immediately. Throttle ensures serialization across
# orchestrator + expert (both use this provider in the default .env).
_THROTTLE_SECONDS = {
    Provider.CEREBRAS: 4.0,
    Provider.NVIDIA: 1.7,
    Provider.CHUTES: 0.0,  # account-based; throttling moot until credits added
    Provider.FEATHERLESS: 1.0,
}

# Per-(provider, api_key) shared throttle state. Two ChatModel instances
# only share a throttle when they hit the same provider AND use the same
# API key — that matches how upstream concurrency quotas are scoped.
# This lets two roles use the same provider with separate keys (e.g.,
# orchestrator + expert both on Featherless but with their own keys) and
# run truly concurrently instead of serializing on a shared lock.
_PROVIDER_THROTTLE_STATE: dict = {}


def _get_provider_throttle(provider: "Provider", api_key: str | None, min_gap: float):
    """Return a shared (last_time, lock) tuple per (provider, api_key).

    Lock is held through the actual API call so concurrent calls from
    different ChatModel instances using the same provider+key are
    serialized end-to-end. Different keys for the same provider get
    separate locks → independent quota.
    """
    import asyncio as _asyncio

    state_key = (provider, api_key or "default")
    if state_key not in _PROVIDER_THROTTLE_STATE:
        _PROVIDER_THROTTLE_STATE[state_key] = {
            "last_time": [0.0],
            "lock": _asyncio.Lock(),
            "min_gap": min_gap,
        }
    return _PROVIDER_THROTTLE_STATE[state_key]


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

    throttle = _THROTTLE_SECONDS.get(s.provider, 0.0)
    if throttle > 0:
        cls = _make_throttled_chat_model_cls(
            OpenAIChatModel, throttle, s.provider, s.api_key
        )
    else:
        cls = OpenAIChatModel

    # NOTE 1: tool_call_fallback_via_response_format=False forces BeeAI to
    # send native OpenAI `tools` instead of a `response_format=json_schema`
    # anyOf-over-all-tools. The fallback path is unsupported by most
    # OpenAI-compat free-tier endpoints.
    #
    # NOTE 2: tool_choice_support omits "required" — Cerebras, NVIDIA NIM,
    # and Chutes do not honour `tool_choice={"required"}`; BeeAI's default
    # raises if it asks for a tool call and the model returns plain text.
    # Restricting to {"auto","single","none"} lets BeeAI plan around it.
    #
    # NOTE 3: ignore_parallel_tool_calls=True silently drops extra tool
    # calls when a model returns multiple in one response (Kimi K2.5 on
    # Featherless does this). MARGE's MARGEProtocolRequirement evaluates
    # gate state per iteration, so parallel calls would bypass the
    # ML-after-expert ordering — sequential is the design.
    model = cls(
        model_id=s.model_id,
        api_key=s.api_key,
        base_url=s.base_url,
        tool_call_fallback_via_response_format=False,
        tool_choice_support={"auto", "single", "none"},
        ignore_parallel_tool_calls=True,
    )
    # BeeAI 0.1.79 keeps tool_choice_support as a class-level default for the
    # OpenAI-compatible adapter, so the constructor argument above may not
    # override it everywhere the runner consults the setting. Force the
    # provider-safe set on both the instance and the concrete class to prevent
    # tool_choice={"required"} calls against providers that do not support it.
    provider_safe_tool_choices = {"auto", "single", "none"}
    model.tool_choice_support = provider_safe_tool_choices
    type(model).tool_choice_support = provider_safe_tool_choices
    return model


def _make_throttled_chat_model_cls(
    base_cls, min_gap_seconds: float, provider: "Provider", api_key: str | None
):
    """Return a subclass of `base_cls` that serializes calls per (provider, api_key).

    The lock + last-call timestamp are shared across every ChatModel
    instance built for the same provider AND same api_key. Two roles on
    the same provider but with separate API keys get independent locks
    and run concurrently — matching how upstream quotas are scoped.

    The lock is held through the actual API call so concurrent requests
    from different ChatModel instances using the same key never overlap
    on the wire (required for free-tier providers like Featherless that
    only allow 1 concurrent request per key and otherwise return HTTP 429).
    """
    state = _get_provider_throttle(provider, api_key, min_gap_seconds)
    last_call_time = state["last_time"]
    lock = state["lock"]

    class ThrottledChatModel(base_cls):  # type: ignore[misc, valid-type]
        async def _create(self, input, run):  # type: ignore[override]
            async with lock:
                elapsed = time.monotonic() - last_call_time[0]
                if elapsed < min_gap_seconds:
                    await asyncio.sleep(min_gap_seconds - elapsed)
                try:
                    return await super()._create(input, run)
                finally:
                    last_call_time[0] = time.monotonic()

        async def _create_stream(self, input, _):  # type: ignore[override]
            async with lock:
                elapsed = time.monotonic() - last_call_time[0]
                if elapsed < min_gap_seconds:
                    await asyncio.sleep(min_gap_seconds - elapsed)
                try:
                    async for out in super()._create_stream(input, _):
                        yield out
                finally:
                    last_call_time[0] = time.monotonic()

    ThrottledChatModel.__name__ = f"Throttled{base_cls.__name__}"
    ThrottledChatModel.__qualname__ = ThrottledChatModel.__name__
    return ThrottledChatModel


def _build_cerebras(s: LLMSettings) -> "ChatModel":
    return _build_openai_compat(s, "CEREBRAS_API_KEY")


def _build_nvidia(s: LLMSettings) -> "ChatModel":
    return _build_openai_compat(s, "NVIDIA_API_KEY")


def _build_chutes(s: LLMSettings) -> "ChatModel":
    return _build_openai_compat(s, "CHUTES_API_KEY")


def _build_featherless(s: LLMSettings) -> "ChatModel":
    return _build_openai_compat(s, "FEATHERLESS_API_KEY")


_BUILDERS = {
    Provider.ANTHROPIC: _build_anthropic,
    Provider.WATSONX: _build_watsonx,
    Provider.CEREBRAS: _build_cerebras,
    Provider.NVIDIA: _build_nvidia,
    Provider.CHUTES: _build_chutes,
    Provider.FEATHERLESS: _build_featherless,
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
