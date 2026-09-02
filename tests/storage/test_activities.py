"""Tests de la construction de la table DuckDB `main.activities` (T-07).

Sélection curée : seul un sous-ensemble des ~59 champs bruts de Strava
est repris, typé et nommé en SI — le reste reste consultable via
raw.activities.
"""

from datetime import datetime

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segment_predictor.storage.activities import build_activities_table


def _write_raw_activities(raw_dir, rows: list[dict], filename: str = "activities.parquet") -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), raw_dir / filename)


def _raw_ride(activity_id: int, **overrides) -> dict:
    base = {
        "id": activity_id,
        "name": "Sortie",
        "type": "Ride",
        "sport_type": "Ride",
        "start_date": "2024-05-01T10:15:00Z",
        "distance": 42000.0,
        "moving_time": 3600,
        "elapsed_time": 3700,
        "total_elevation_gain": 500.0,
        "average_watts": 210.5,
        "device_watts": True,
        "average_heartrate": 145.0,
        "max_heartrate": 178.0,
        "average_cadence": 88.0,
    }
    base.update(overrides)
    return base


def test_build_activities_table_extracts_curated_typed_columns(tmp_path) -> None:
    raw_dir = tmp_path / "activities"
    _write_raw_activities(raw_dir, [_raw_ride(1)])

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir)

    row = conn.execute(
        "SELECT id, name, type, sport_type, start_date, distance_m, moving_time_s, "
        "elapsed_time_s, total_elevation_gain_m, average_watts, device_watts, "
        "average_heartrate, max_heartrate, average_cadence FROM activities"
    ).fetchone()

    assert row == (
        1,
        "Sortie",
        "Ride",
        "Ride",
        datetime(2024, 5, 1, 10, 15, 0),
        42000.0,
        3600,
        3700,
        500.0,
        210.5,
        True,
        145.0,
        178.0,
        88.0,
    )


def test_build_activities_table_allows_null_optional_metrics(tmp_path) -> None:
    """Une Hike n'a pas de watts/cadence : NULL légitime, pas une erreur."""
    raw_dir = tmp_path / "activities"
    _write_raw_activities(
        raw_dir,
        [
            _raw_ride(
                1,
                type="Hike",
                sport_type="Hike",
                average_watts=None,
                device_watts=None,
                average_cadence=None,
            )
        ],
    )

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir)

    row = conn.execute(
        "SELECT average_watts, device_watts, average_cadence FROM activities"
    ).fetchone()
    assert row == (None, None, None)


def test_build_activities_table_raises_when_core_field_missing(tmp_path) -> None:
    """id/name/type/distance/... sont attendus sur toute activité : pas de défaut silencieux."""
    raw_dir = tmp_path / "activities"
    raw = _raw_ride(1)
    del raw["distance"]
    _write_raw_activities(raw_dir, [raw])

    conn = duckdb.connect(":memory:")
    with pytest.raises(KeyError):
        build_activities_table(conn, raw_dir)


def test_build_activities_table_combines_multiple_raw_files(tmp_path) -> None:
    raw_dir = tmp_path / "activities"
    _write_raw_activities(raw_dir, [_raw_ride(1)], filename="run1.parquet")
    _write_raw_activities(raw_dir, [_raw_ride(2)], filename="run2.parquet")

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM activities").fetchone()[0]
    assert count == 2


def test_build_activities_table_replaces_existing_table(tmp_path) -> None:
    raw_dir = tmp_path / "activities"
    _write_raw_activities(raw_dir, [_raw_ride(1)])

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir)
    build_activities_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM activities").fetchone()[0]
    assert count == 1
