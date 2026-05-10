"""Tests for the OrchestratorBundle assembly.

Bundle wires five deterministic local tools (update_user, consult_medical_expert,
request_more_info, clinical_report, abstain) and the protocol enforcer.
Patient data and ML tools are attached via MCP at runtime and are not tested here.
"""

import pytest

from apps.orchestrator.agent import OrchestratorBundle, build_bundle
from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer


_EXPECTED_LOCAL_TOOLS = {
    "update_user",
    "conversational_reply",
    "consult_medical_expert",
    "request_more_info",
    "clinical_report",
    "abstain",
}


class TestBuildBundle:
    def test_returns_orchestrator_bundle(self):
        bundle = build_bundle()
        assert isinstance(bundle, OrchestratorBundle)

    def test_bundle_has_enforcer(self):
        bundle = build_bundle()
        assert isinstance(bundle.enforcer, ProtocolEnforcer)

    def test_bundle_has_six_local_tools(self):
        bundle = build_bundle()
        assert set(bundle.local_tools.keys()) == _EXPECTED_LOCAL_TOOLS

    @pytest.mark.asyncio
    async def test_bundle_local_tools_share_one_enforcer(self):
        bundle = build_bundle()
        await bundle.local_tools["consult_medical_expert"](question="?", findings={})
        assert bundle.enforcer.has_called("consult_medical_expert")

    def test_system_prompt_loaded(self):
        bundle = build_bundle()
        # System prompt rewrite happens in a later round; for now just check
        # it loads non-empty and references the new terminal tool name.
        assert bundle.system_prompt
        assert "consult_medical_expert" in bundle.system_prompt
