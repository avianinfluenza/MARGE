"""Stub SqlitePatientSource — in-memory hardcoded seed patients.

For the thin slice this is just an in-memory dict. The real SQLite-backed
implementation will replace this without changing the public API
(`resolve`, `list_handles`).

Each seed patient carries enough features to be predicted by every currently
registered ML model — so the orchestrator can exercise multiple tools on
the same patient without artificial data construction.
"""

from sklearn.datasets import load_breast_cancer

from packages.schemas.patient import PatientRecord
from services.patient_data_mcp_server.sources._base import PatientSource


def _breast_cancer_features() -> dict[str, float]:
    """First row of the sklearn breast cancer dataset (a real malignant case)."""
    data = load_breast_cancer()
    return {
        name.replace(" ", "_"): float(val)
        for name, val in zip(data.feature_names, data.data[0], strict=False)
    }


# Pima Indians Diabetes — first row of the dataset, hardcoded to avoid an
# OpenML fetch at import time (the same row the diabetes_catboost model uses
# as `sample_inputs()`).
_PIMA_FIRST_ROW: dict[str, float] = {
    "preg": float("nan"),
    "plas": float("nan"),
    "pres": float("nan"),
    "skin": float("nan"),
    "insu": float("nan"),
    "mass": float("nan"),
    "pedi": float("nan"),
    "age": float("nan"),
}


def _build_seed_patients() -> dict[str, PatientRecord]:
    return {
        "seed-001": PatientRecord(
            handle="seed-001",
            age=50,
            sex="female",
            features={**_breast_cancer_features(), **_PIMA_FIRST_ROW},
            notes=(
                "Stub seed patient combining the first row of the sklearn "
                "breast cancer dataset and the first row of the Pima Indians "
                "diabetes dataset. Used for thin-slice integration tests."
            ),
        ),
    }


class SqlitePatientSource(PatientSource):
    """Hardcoded patient source for the thin slice.

    Implements PatientSource ABC; backed by an in-memory dict instead of a
    real SQLite DB. The real implementation will be a drop-in replacement
    that swaps the storage but keeps the public API identical.
    """

    def __init__(self) -> None:
        self._patients = _build_seed_patients()

    def list_handles(self) -> list[str]:
        return list(self._patients.keys())

    def resolve(self, handle: str) -> PatientRecord:
        if handle not in self._patients:
            raise KeyError(f"Unknown patient handle: {handle}")
        return self._patients[handle]
