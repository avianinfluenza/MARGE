"""Tests for the OrchestratorBundle assembly.

Bundle wires two deterministic local tools (consult_medical_expert, final_report)
and the protocol enforcer. Patient data and ML tools are attached via MCP at
runtime and are not tested here.
"""

from apps.orchestrator.agent import OrchestratorBundle, build_bundle
from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer


class TestBuildBundle:
    def test_returns_orchestrator_bundle(self):
        bundle = build_bundle()
        assert isinstance(bundle, OrchestratorBundle)

    def test_bundle_has_enforcer(self):
        bundle = build_bundle()
        assert isinstance(bundle.enforcer, ProtocolEnforcer)

    def test_bundle_has_two_local_tools(self):
        bundle = build_bundle()
        expected = {"consult_medical_expert", "final_report"}
        assert set(bundle.local_tools.keys()) == expected

    def test_bundle_local_tools_share_one_enforcer(self):
        bundle = build_bundle()
        bundle.local_tools["consult_medical_expert"](question="?", findings={})
        assert bundle.enforcer.has_called("consult_medical_expert")

    def test_system_prompt_loaded(self):
        bundle = build_bundle()
        assert "ML Orchestrator" in bundle.system_prompt
        assert "consult_medical_expert" in bundle.system_prompt
        assert "final_report" in bundle.system_prompt
