"""Local tool: fetch a patient record from the patient data source."""

from collections.abc import Callable

from apps.orchestrator.middleware.enforce_protocol import ProtocolEnforcer
from packages.schemas.patient import PatientRecord
from services.patient_data_mcp_server.sources._base import PatientSource

_TOOL_NAME = "get_patient_history"


def make_patient_history(
    source: PatientSource,
    enforcer: ProtocolEnforcer,
) -> Callable[..., PatientRecord]:
    """Build the get_patient_history tool bound to a specific source + enforcer."""

    def get_patient_history(handle: str) -> PatientRecord:
        """Fetch a patient record by handle (e.g., 'seed-001').

        Args:
            handle: Source-prefixed patient ID.

        Returns:
            PatientRecord with demographics + flat feature dict.
        """
        enforcer.record(_TOOL_NAME)
        return source.resolve(handle)

    return get_patient_history
