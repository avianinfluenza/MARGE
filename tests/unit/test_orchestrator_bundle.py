"""Tests for the OrchestratorBundle assembly.

The bundle wires the deterministic pieces of the orchestrator (enforcer +
local tools + stubs). The BeeAI agent itself is constructed from the
bundle at run time and is not unit-tested here (it requires a live LLM).
"""

from apps.orchestrator.agent import OrchestratorBundle, build_bundle
from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from services.patient_data_mcp_server.sources._base import PatientSource


class TestBuildBundle:
    def test_returns_orchestrator_bundle(self):
        bundle = build_bundle()
        assert isinstance(bundle, OrchestratorBundle)

    def test_bundle_has_enforcer(self):
        bundle = build_bundle()
        assert isinstance(bundle.enforcer, ProtocolEnforcer)

    def test_bundle_has_patient_source(self):
        bundle = build_bundle()
        assert isinstance(bundle.patient_source, PatientSource)

    def test_bundle_has_all_five_local_tools(self):
        bundle = build_bundle()
        expected = {
            "get_patient_history",
            "consult_medical_expert",
            "final_report",
            "abstain",
            "ask_user_back",
        }
        assert set(bundle.local_tools.keys()) == expected

    def test_bundle_local_tools_share_one_enforcer(self):
        """Calling any tool must update the same enforcer the bundle exposes."""
        bundle = build_bundle()
        bundle.local_tools["get_patient_history"](handle="seed-001")
        assert bundle.enforcer.has_called("get_patient_history")

    def test_system_prompt_loaded(self):
        bundle = build_bundle()
        assert "ML Head Researcher" in bundle.system_prompt
        assert "consult_medical_expert" in bundle.system_prompt
