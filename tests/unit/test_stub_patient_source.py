"""Tests for SqlitePatientSource backed by a real SQLite file."""

import pytest

from packages.schemas.patient import PatientRecord
from services.patient_data_mcp_server.sources._base import PatientSource
from services.patient_data_mcp_server.sources.csv_ingest import seed_demo_db
from services.patient_data_mcp_server.sources.sqlite_db import SqlitePatientSource


@pytest.fixture()
def demo_source(tmp_path):
    db = tmp_path / "test.db"
    seed_demo_db(db)
    return SqlitePatientSource(db)


class TestSqlitePatientSource:
    def test_implements_patient_source_abc(self, demo_source):
        assert isinstance(demo_source, PatientSource)

    def test_resolves_known_seed_handle(self, demo_source):
        record = demo_source.resolve("seed-001")
        assert isinstance(record, PatientRecord)
        assert record.handle == "seed-001"

    def test_resolved_record_has_features(self, demo_source):
        record = demo_source.resolve("seed-001")
        assert record.features
        for name, value in record.features.items():
            assert isinstance(name, str)
            assert value is None or isinstance(value, (int, float))

    def test_seed_001_has_features_for_both_ml_models(self, demo_source):
        record = demo_source.resolve("seed-001")
        assert "mean_radius" in record.features
        assert "worst_texture" in record.features
        for k in ("preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age"):
            assert k in record.features

    def test_unknown_handle_raises_keyerror(self, demo_source):
        with pytest.raises(KeyError, match="seed-9999"):
            demo_source.resolve("seed-9999")

    def test_lists_available_handles(self, demo_source):
        handles = demo_source.list_handles()
        assert "seed-001" in handles

    def test_update_merges_features(self, demo_source):
        updated = demo_source.update("seed-001", feature_updates={"plas": 148.0, "age": 50.0})
        assert updated.features["plas"] == 148.0
        assert updated.features["age"] == 50.0

    def test_update_appends_notes(self, demo_source):
        demo_source.update("seed-001", notes_append="First note.")
        updated = demo_source.update("seed-001", notes_append="Second note.")
        assert "First note." in updated.notes
        assert "Second note." in updated.notes

    def test_update_persists_across_resolve(self, demo_source):
        demo_source.update("seed-001", feature_updates={"mass": 33.6})
        record = demo_source.resolve("seed-001")
        assert record.features["mass"] == 33.6
