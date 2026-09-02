"""Tests de la construction de la table `wellness` (T-22).

Vérifié en conditions réelles (fetch intervals.icu) : `weight` est bien
en kg (une valeur de 83.0 pour un cycliste adulte, pas 83 lb — beaucoup
trop léger), et `id` est bien la date ISO-8601 du jour. `hrv` et
`sleepSecs` n'ont jamais été renseignés dans l'historique réel utilisé
pour vérifier — leur unité/format exact reste documenté comme non
vérifié empiriquement au-delà du nom du champ lui-même.
"""

from datetime import date

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from segment_predictor.storage.wellness import build_wellness_table


def _write_wellness(raw_dir, records: list[dict]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), raw_dir / "wellness.parquet")


def test_build_wellness_table_extracts_the_four_tracked_fields(tmp_path) -> None:
    raw_dir = tmp_path / "wellness"
    _write_wellness(
        raw_dir,
        [
            {
                "id": "2025-06-01",
                "hrv": 65.0,
                "sleepSecs": 27000,
                "restingHR": 48,
                "weight": 78.2,
                "ctl": 40.0,  # champ hors périmètre T-22, ne doit pas planter l'extraction
            }
        ],
    )
    conn = duckdb.connect(":memory:")

    build_wellness_table(conn, raw_dir)

    row = conn.execute("SELECT * FROM wellness").fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM wellness").description]
    record = dict(zip(cols, row, strict=True))
    assert record["date"] == date(2025, 6, 1)
    assert record["hrv"] == 65.0
    assert record["sleep_s"] == 27000
    assert record["resting_heart_rate_bpm"] == 48
    assert record["weight_kg"] == 78.2


def test_build_wellness_table_keeps_partial_days_with_nulls(tmp_path) -> None:
    raw_dir = tmp_path / "wellness"
    _write_wellness(
        raw_dir,
        [{"id": "2025-06-02", "hrv": None, "sleepSecs": None, "restingHR": 50, "weight": None}],
    )
    conn = duckdb.connect(":memory:")

    build_wellness_table(conn, raw_dir)

    row = conn.execute(
        "SELECT hrv, sleep_s, resting_heart_rate_bpm, weight_kg FROM wellness"
    ).fetchone()
    assert row == (None, None, 50, None)


def test_build_wellness_table_creates_empty_table_when_never_fetched(tmp_path) -> None:
    raw_dir = tmp_path / "wellness"  # jamais créé : fetch_wellness.py pas encore lancé
    conn = duckdb.connect(":memory:")

    build_wellness_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM wellness").fetchone()[0]
    assert count == 0
