"""FastMCP server exposing patient data tools.

Run standalone (stdio transport):
    PATIENT_DB_PATH=sessions/my.db python -m services.patient_data_mcp_server.server

Or instantiated in-process via `build_patient_server(db_path)` and connected
with a FastMCP `Client`, exactly like `ml_mcp_server.server`.

Tools exposed:
    list_patients()                                    → list[str]
    get_patient(handle)                                → PatientRecord (as dict)
    update_patient(handle, feature_updates, notes_append) → PatientRecord (as dict)
"""

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from packages.schemas.patient import PatientRecord
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


def build_patient_server(db_path: Path) -> FastMCP:
    """Return an in-process FastMCP server backed by the given SQLite DB."""
    mcp = FastMCP("patient-data")
    source = SqlitePatientSource(db_path)

    @mcp.tool()
    def list_patients() -> list[str]:
        """List all patient handles available in this session."""
        return source.list_handles()

    @mcp.tool()
    def get_patient(handle: str) -> dict[str, Any]:
        """Fetch a full patient record by handle (demographics + feature dict)."""
        return source.resolve(handle).model_dump()

    @mcp.tool()
    def update_patient(
        handle: str,
        feature_updates: dict[str, Any] | None = None,
        notes_append: str | None = None,
    ) -> dict[str, Any]:
        """Update a patient's clinical features and/or append a note.

        Call this whenever the user mentions new clinical values (glucose, BMI,
        blood pressure, age, etc.) so they are persisted for later analysis.

        Args:
            handle: Patient handle, e.g. "csv-42" or "seed-001".
            feature_updates: Mapping of feature_name → numeric value to merge
                into the patient's feature dict. Pass null to skip.
            notes_append: Free-text to append to the patient's clinical notes.
                Pass null to skip.
        """
        return source.update(handle, feature_updates, notes_append).model_dump()

    return mcp


def main() -> None:
    import os

    db_path_str = os.environ.get("PATIENT_DB_PATH")
    if not db_path_str:
        raise SystemExit("PATIENT_DB_PATH env var is required.")
    server = build_patient_server(Path(db_path_str))
    server.run()


if __name__ == "__main__":
    main()
