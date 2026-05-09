"""Tests for SqlitePatientSource (in-memory stub for the thin slice).

For the thin slice, the source returns a hardcoded set of seeded patients.
The real SQLite-backed implementation will replace this without changing
the public API (`resolve`, `list_handles`).
"""

import pytest

from packages.schemas.patient import PatientRecord
from services.patient_data_mcp_server.sources._base import PatientSource
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


class TestSqlitePatientSource:
    def test_implements_patient_source_abc(self):
        source = SqlitePatientSource()
        assert isinstance(source, PatientSource)

    def test_resolves_known_seed_handle(self):
        source = SqlitePatientSource()
        record = source.resolve("seed-001")
        assert isinstance(record, PatientRecord)
        assert record.handle == "seed-001"

    def test_resolved_record_has_features(self):
        source = SqlitePatientSource()
        record = source.resolve("seed-001")
        assert record.features
        for name, value in record.features.items():
            assert isinstance(name, str)
            assert isinstance(value, (int, float))

    def test_seed_001_has_features_for_both_ml_models(self):
        """Patient should be usable by both registered ML tools."""
        source = SqlitePatientSource()
        record = source.resolve("seed-001")
        # Breast cancer model needs the 30 sklearn features (a sample)
        assert "mean_radius" in record.features
        assert "worst_texture" in record.features
        # Diabetes model needs the 8 Pima features
        for k in ("preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"):
            assert k in record.features

    def test_unknown_handle_raises_keyerror(self):
        source = SqlitePatientSource()
        with pytest.raises(KeyError, match="seed-9999"):
            source.resolve("seed-9999")

    def test_lists_available_handles(self):
        source = SqlitePatientSource()
        handles = source.list_handles()
        assert "seed-001" in handles
