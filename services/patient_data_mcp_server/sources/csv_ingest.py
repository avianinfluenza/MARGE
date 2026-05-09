"""Ingest a multi-patient CSV into a session-scoped SQLite database.

CSV format:
  - First column: patient identifier (becomes handle "csv-{value}")
  - Optional reserved columns: age, sex, notes  →  PatientRecord metadata
  - All other columns: numeric feature values

Provides `seed_demo_db` for demo sessions without an uploaded CSV.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    handle       TEXT PRIMARY KEY,
    age          INTEGER,
    sex          TEXT,
    features_json TEXT NOT NULL DEFAULT '{}',
    notes        TEXT
)
"""

_RESERVED_COLS = {"age", "sex", "notes"}


def _open(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def ingest_csv(csv_source: Path | str | bytes, db_path: Path) -> list[str]:
    """Parse multi-patient CSV, write to SQLite, return list of handles."""
    if isinstance(csv_source, bytes):
        import io
        df = pd.read_csv(io.BytesIO(csv_source))
    else:
        df = pd.read_csv(csv_source)

    if df.empty:
        raise ValueError("CSV contains no data rows.")

    id_col = df.columns[0]
    feature_cols = [c for c in df.columns if c != id_col and c.lower() not in _RESERVED_COLS]

    conn = _open(db_path)
    handles: list[str] = []
    with conn:
        for _, row in df.iterrows():
            handle = f"csv-{row[id_col]}"
            age = int(row["age"]) if "age" in row and pd.notna(row["age"]) else None
            sex_raw = row.get("sex")
            sex = str(sex_raw) if sex_raw is not None and pd.notna(sex_raw) else None
            notes_raw = row.get("notes")
            notes = str(notes_raw) if notes_raw is not None and pd.notna(notes_raw) else None

            features = {
                col: float(row[col]) if pd.notna(row[col]) else None
                for col in feature_cols
            }

            conn.execute(
                "INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?)",
                (handle, age, sex, json.dumps(features), notes),
            )
            handles.append(handle)

    conn.close()
    return handles


def seed_demo_db(db_path: Path) -> None:
    """Create a demo SQLite DB with seed-001 (breast-cancer + diabetes features)."""
    bc = load_breast_cancer()
    bc_features = {
        name.replace(" ", "_"): float(val)
        for name, val in zip(bc.feature_names, bc.data[0], strict=False)
    }
    diabetes_features: dict[str, float | None] = {
        "preg": None, "plas": None, "pres": None,
        "skin": None, "insu": None, "mass": None,
        "pedi": None, "age": None,
    }
    features = {**bc_features, **diabetes_features}

    conn = _open(db_path)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?)",
            (
                "seed-001",
                50,
                "female",
                json.dumps(features),
                "Demo seed patient — breast cancer features from Wisconsin Diagnostic dataset.",
            ),
        )
    conn.close()
