"""Tests for the LLM provider abstraction.

Covers five providers (Anthropic, watsonx, Cerebras, NVIDIA NIM, Chutes),
per-role routing (orchestrator vs medical expert), and the fallback wrapper.

No live LLM calls — provider construction returns the right adapter class
and `build_chat_model_for_role` composes primary + fallback correctly.
"""

import os
from unittest.mock import patch

import pytest

from packages.llm_provider.client import (
    FallbackChatModel,
    build_chat_model,
    build_chat_model_for_role,
)
from packages.llm_provider.settings import (
    LLMSettings,
    Provider,
    Role,
    RoleConfig,
)


# ---------------------------------------------------------------------------
# LLMSettings — provider-level config
# ---------------------------------------------------------------------------


class TestLLMSettings:
    def test_defaults_to_anthropic(self):
        with patch.dict(os.environ, {}, clear=True):
            s = LLMSettings.from_env()
            assert s.provider == Provider.ANTHROPIC

    def test_reads_provider_from_env(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "watsonx"}, clear=True):
            s = LLMSettings.from_env()
            assert s.provider == Provider.WATSONX

    def test_unknown_provider_raises(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "no_such_one"}, clear=True):
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                LLMSettings.from_env()

    def test_anthropic_settings(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-test"},
            clear=True,
        ):
            s = LLMSettings.from_env()
            assert s.api_key == "sk-test"
            assert s.model_id.startswith("claude-")

    def test_watsonx_settings(self):
        env = {
            "LLM_PROVIDER": "watsonx",
            "WATSONX_API_KEY": "wx-test",
            "WATSONX_PROJECT_ID": "proj-1",
            "WATSONX_URL": "https://us-south.ml.cloud.ibm.com",
            "WATSONX_MODEL_ID": "ibm/granite-3-8b-instruct",
        }
        with patch.dict(os.environ, env, clear=True):
            s = LLMSettings.from_env()
            assert s.api_key == "wx-test"
            assert s.project_id == "proj-1"
            assert s.base_url == "https://us-south.ml.cloud.ibm.com"
            assert s.model_id == "ibm/granite-3-8b-instruct"

    def test_cerebras_settings(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "cerebras", "CEREBRAS_API_KEY": "csk-x"},
            clear=True,
        ):
            s = LLMSettings.from_env()
            assert s.provider == Provider.CEREBRAS
            assert s.api_key == "csk-x"
            assert s.base_url == "https://api.cerebras.ai/v1"
            assert "qwen" in s.model_id.lower()

    def test_nvidia_settings(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "nvidia", "NVIDIA_API_KEY": "nvapi-x"},
            clear=True,
        ):
            s = LLMSettings.from_env()
            assert s.provider == Provider.NVIDIA
            assert s.api_key == "nvapi-x"
            assert s.base_url == "https://integrate.api.nvidia.com/v1"
            assert "qwen" in s.model_id.lower()

    def test_chutes_settings(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "chutes", "CHUTES_API_KEY": "cpk-x"},
            clear=True,
        ):
            s = LLMSettings.from_env()
            assert s.provider == Provider.CHUTES
            assert s.api_key == "cpk-x"
            assert "chutes" in (s.base_url or "").lower()
            assert "kimi" in s.model_id.lower()


# ---------------------------------------------------------------------------
# build_chat_model — provider -> ChatModel adapter
# ---------------------------------------------------------------------------


class TestBuildChatModel:
    def test_returns_anthropic_chat_model(self):
        s = LLMSettings(
            provider=Provider.ANTHROPIC, model_id="claude-x", api_key="sk-test"
        )
        model = build_chat_model(s)
        from beeai_framework.adapters.anthropic.backend.chat import AnthropicChatModel

        assert isinstance(model, AnthropicChatModel)

    def test_returns_watsonx_chat_model(self):
        s = LLMSettings(
            provider=Provider.WATSONX,
            model_id="ibm/granite-3-8b-instruct",
            api_key="wx-test",
            project_id="proj-1",
            base_url="https://us-south.ml.cloud.ibm.com",
        )
        model = build_chat_model(s)
        from beeai_framework.adapters.watsonx.backend.chat import WatsonxChatModel

        assert isinstance(model, WatsonxChatModel)

    def test_cerebras_uses_openai_compat_adapter(self):
        s = LLMSettings(
            provider=Provider.CEREBRAS,
            model_id="qwen-3-235b-a22b-instruct-2507",
            api_key="csk-test",
            base_url="https://api.cerebras.ai/v1",
        )
        model = build_chat_model(s)
        from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel

        assert isinstance(model, OpenAIChatModel)
        assert model.tool_choice_support == {"auto", "single", "none"}

    def test_nvidia_uses_openai_compat_adapter(self):
        s = LLMSettings(
            provider=Provider.NVIDIA,
            model_id="qwen/qwen3.5-397b-a17b",
            api_key="nvapi-test",
            base_url="https://integrate.api.nvidia.com/v1",
        )
        model = build_chat_model(s)
        from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel

        assert isinstance(model, OpenAIChatModel)
        assert model.tool_choice_support == {"auto", "single", "none"}

    def test_chutes_uses_openai_compat_adapter(self):
        s = LLMSettings(
            provider=Provider.CHUTES,
            model_id="moonshotai/Kimi-K2.5-TEE",
            api_key="cpk-test",
            base_url="https://llm.chutes.ai/v1",
        )
        model = build_chat_model(s)
        from beeai_framework.adapters.openai.backend.chat import OpenAIChatModel

        assert isinstance(model, OpenAIChatModel)
        assert model.tool_choice_support == {"auto", "single", "none"}

    @pytest.mark.parametrize(
        "provider,key_var",
        [
            (Provider.ANTHROPIC, "ANTHROPIC_API_KEY"),
            (Provider.CEREBRAS, "CEREBRAS_API_KEY"),
            (Provider.NVIDIA, "NVIDIA_API_KEY"),
            (Provider.CHUTES, "CHUTES_API_KEY"),
        ],
    )
    def test_each_provider_requires_its_key(self, provider, key_var):
        # Build minimal settings without the api_key
        s = LLMSettings(provider=provider, model_id="x")
        with pytest.raises(ValueError, match=key_var):
            build_chat_model(s)


# ---------------------------------------------------------------------------
# RoleConfig — per-role primary + fallback
# ---------------------------------------------------------------------------


class TestRoleConfig:
    def test_orchestrator_reads_role_specific_env(self):
        env = {
            "ORCHESTRATOR_PRIMARY": "chutes",
            "ORCHESTRATOR_FALLBACK": "cerebras",
            "CEREBRAS_API_KEY": "csk-x",
            "CHUTES_API_KEY": "cpk-x",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = RoleConfig.from_env(Role.ORCHESTRATOR)
            assert cfg.primary.provider == Provider.CHUTES
            assert cfg.fallback is not None
            assert cfg.fallback.provider == Provider.CEREBRAS

    def test_medical_expert_reads_role_specific_env(self):
        env = {
            "MEDICAL_EXPERT_PRIMARY": "nvidia",
            "MEDICAL_EXPERT_FALLBACK": "cerebras",
            "CEREBRAS_API_KEY": "csk-x",
            "NVIDIA_API_KEY": "nvapi-x",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = RoleConfig.from_env(Role.MEDICAL_EXPERT)
            assert cfg.primary.provider == Provider.NVIDIA
            assert cfg.fallback is not None
            assert cfg.fallback.provider == Provider.CEREBRAS

    def test_no_fallback_is_optional(self):
        env = {"ORCHESTRATOR_PRIMARY": "chutes", "CHUTES_API_KEY": "cpk-x"}
        with patch.dict(os.environ, env, clear=True):
            cfg = RoleConfig.from_env(Role.ORCHESTRATOR)
            assert cfg.fallback is None

    def test_missing_primary_falls_back_to_anthropic(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=True):
            cfg = RoleConfig.from_env(Role.ORCHESTRATOR)
            assert cfg.primary.provider == Provider.ANTHROPIC


# ---------------------------------------------------------------------------
# FallbackChatModel + build_chat_model_for_role
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal duck-typed ChatModel for fallback wrapper tests."""

    def __init__(self, name: str, succeeds: bool, payload: str = "ok") -> None:
        self.model_id = name
        self.provider_id = "fake"
        self._succeeds = succeeds
        self._payload = payload
        self.run_calls = 0

    async def run(self, *args, **kwargs):
        self.run_calls += 1
        if not self._succeeds:
            raise RuntimeError(f"{self.model_id} failed")
        return f"{self.model_id}: {self._payload}"


class TestFallbackChatModel:
    @pytest.mark.asyncio
    async def test_returns_primary_result_when_primary_succeeds(self):
        primary = _FakeModel("primary", succeeds=True, payload="hello")
        fallback = _FakeModel("fallback", succeeds=True, payload="alt")
        m = FallbackChatModel(primary, fallback)
        result = await m.run("anything")
        assert result == "primary: hello"
        assert primary.run_calls == 1
        assert fallback.run_calls == 0

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_raises(self):
        primary = _FakeModel("primary", succeeds=False)
        fallback = _FakeModel("fallback", succeeds=True, payload="alt")
        m = FallbackChatModel(primary, fallback)
        result = await m.run("anything")
        assert result == "fallback: alt"
        assert primary.run_calls == 1
        assert fallback.run_calls == 1

    @pytest.mark.asyncio
    async def test_propagates_when_both_fail(self):
        primary = _FakeModel("primary", succeeds=False)
        fallback = _FakeModel("fallback", succeeds=False)
        m = FallbackChatModel(primary, fallback)
        with pytest.raises(RuntimeError, match="fallback"):
            await m.run("anything")

    def test_attribute_access_proxied_to_primary(self):
        primary = _FakeModel("primary", succeeds=True)
        fallback = _FakeModel("fallback", succeeds=True)
        m = FallbackChatModel(primary, fallback)
        assert m.model_id == "primary"
        assert m.provider_id == "fake"


class TestBuildChatModelForRole:
    def test_returns_primary_only_by_default(self):
        env = {
            "ORCHESTRATOR_PRIMARY": "chutes",
            "ORCHESTRATOR_FALLBACK": "cerebras",
            "CEREBRAS_API_KEY": "csk-x",
            "CHUTES_API_KEY": "cpk-x",
        }
        with patch.dict(os.environ, env, clear=True):
            model = build_chat_model_for_role(Role.ORCHESTRATOR)
            # Primary only — fallback wrapping is opt-in (not BeeAI-integrated yet).
            assert not isinstance(model, FallbackChatModel)

    def test_wraps_when_explicitly_requested(self):
        env = {
            "ORCHESTRATOR_PRIMARY": "chutes",
            "ORCHESTRATOR_FALLBACK": "cerebras",
            "CEREBRAS_API_KEY": "csk-x",
            "CHUTES_API_KEY": "cpk-x",
        }
        with patch.dict(os.environ, env, clear=True):
            model = build_chat_model_for_role(Role.ORCHESTRATOR, with_fallback=True)
            assert isinstance(model, FallbackChatModel)

    def test_returns_bare_model_when_no_fallback(self):
        env = {"ORCHESTRATOR_PRIMARY": "chutes", "CHUTES_API_KEY": "cpk-x"}
        with patch.dict(os.environ, env, clear=True):
            model = build_chat_model_for_role(Role.ORCHESTRATOR, with_fallback=True)
            assert not isinstance(model, FallbackChatModel)
