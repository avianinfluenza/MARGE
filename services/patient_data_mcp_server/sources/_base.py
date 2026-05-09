"""Abstract base class for patient data sources.

Each implementation (SQLite seed DB, CSV upload adapter, future EHR clients)
implements the same `resolve` + `list_handles` interface, so the
patient_data_mcp_server can dispatch by handle prefix without branching on
source type.
"""

from abc import ABC, abstractmethod

from packages.schemas.patient import PatientRecord


class PatientSource(ABC):
    """Resolve a handle (e.g., 'seed-001', 'upload-XXXX') to a PatientRecord."""

    @abstractmethod
    def resolve(self, handle: str) -> PatientRecord:
        """Return the PatientRecord for the given handle.

        Raises KeyError if the handle is not known to this source.
        """

    @abstractmethod
    def list_handles(self) -> list[str]:
        """Return all handles currently known to this source."""
