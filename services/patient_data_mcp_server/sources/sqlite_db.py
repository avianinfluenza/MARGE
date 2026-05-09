"""SQLite-backed patient source.

Replaces the in-memory stub. Backed by a file-path SQLite DB created by
`csv_ingest.ingest_csv` or `csv_ingest.seed_demo_db`.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from packages.schemas.patient import PatientRecord
from services.patient_data_mcp_server.sources._base import PatientSource


class SqlitePatientSource(PatientSource):
    """Reads and updates patients from a SQLite database file."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)

    # ------------------------------------------------------------------
    # PatientSource ABC
    # ------------------------------------------------------------------

    def resolve(self, handle: str) -> PatientRecord:
        row = self._conn.execute(
            "SELECT handle, age, sex, features_json, notes FROM patients WHERE handle = ?",
            (handle,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown patient handle: {handle!r}")
        return self._row_to_record(row)

    def list_handles(self) -> list[str]:
        rows = self._conn.execute("SELECT handle FROM patients ORDER BY handle").fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Write path (called by MCP update_patient tool)
    # ------------------------------------------------------------------

    def update(
        self,
        handle: str,
        feature_updates: dict[str, Any] | None = None,
        notes_append: str | None = None,
    ) -> PatientRecord:
        record = self.resolve(handle)

        new_features = dict(record.features)
        if feature_updates:
            new_features.update(feature_updates)

        existing_notes = record.notes or ""
        if notes_append:
            new_notes: str | None = f"{existing_notes}\n{notes_append}".strip()
        else:
            new_notes = existing_notes or None

        with self._conn:
            self._conn.execute(
                "UPDATE patients SET features_json = ?, notes = ? WHERE handle = ?",
                (json.dumps(new_features), new_notes, handle),
            )

        return PatientRecord(
            handle=record.handle,
            age=record.age,
            sex=record.sex,
            features=new_features,
            notes=new_notes,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: tuple) -> PatientRecord:
        handle, age, sex, features_json, notes = row
        return PatientRecord(
            handle=handle,
            age=age,
            sex=sex,
            features=json.loads(features_json),
            notes=notes,
        )
