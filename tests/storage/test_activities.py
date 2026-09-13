"""Tests de la construction de la table DuckDB `main.activities` (T-07,
T-44b pour `user_id`).

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

_USER_ID = 1
_OTHER_USER_ID = 2


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
        "start_latlng": [47.7, 7.4],
    }
    base.update(overrides)
    return base


def test_build_activities_table_extracts_curated_typed_columns(tmp_path) -> None:
    raw_dir = tmp_path / "activities"
    _write_raw_activities(raw_dir, [_raw_ride(1)])

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir, user_id=_USER_ID)

    row = conn.execute(
        "SELECT id, name, type, sport_type, start_date, distance_m, moving_time_s, "
        "elapsed_time_s, total_elevation_gain_m, average_watts, device_watts, "
        "average_heartrate, max_heartrate, average_cadence, user_id FROM activities"
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
        _USER_ID,
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
    build_activities_table(conn, raw_dir, user_id=_USER_ID)

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
        build_activities_table(conn, raw_dir, user_id=_USER_ID)


def test_build_activities_table_combines_multiple_raw_files(tmp_path) -> None:
    raw_dir = tmp_path / "activities"
    _write_raw_activities(raw_dir, [_raw_ride(1)], filename="run1.parquet")
    _write_raw_activities(raw_dir, [_raw_ride(2)], filename="run2.parquet")

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir, user_id=_USER_ID)

    count = conn.execute("SELECT count(*) FROM activities").fetchone()[0]
    assert count == 2


def test_build_activities_table_extracts_start_latlng(tmp_path) -> None:
    raw_dir = tmp_path / "activities"
    _write_raw_activities(raw_dir, [_raw_ride(1, start_latlng=[47.7, 7.4])])

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir, user_id=_USER_ID)

    row = conn.execute("SELECT start_lat, start_lng FROM activities").fetchone()
    assert row == (47.7, 7.4)


def test_build_activities_table_allows_null_start_latlng(tmp_path) -> None:
    """Une activité indoor (home trainer) a start_latlng=[] côté Strava, pas absent."""
    raw_dir = tmp_path / "activities"
    _write_raw_activities(raw_dir, [_raw_ride(1, start_latlng=[])])

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir, user_id=_USER_ID)

    row = conn.execute("SELECT start_lat, start_lng FROM activities").fetchone()
    assert row == (None, None)


def test_build_activities_table_resync_replaces_only_that_users_rows(tmp_path) -> None:
    """Ré-appeler pour le MÊME utilisateur (ex. re-synchronisation) ne
    duplique pas ses lignes — remplacé, T-44b (DELETE + INSERT, plus un
    CREATE OR REPLACE qui effacerait aussi les autres utilisateurs)."""
    raw_dir = tmp_path / "activities"
    _write_raw_activities(raw_dir, [_raw_ride(1)])

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir, user_id=_USER_ID)
    build_activities_table(conn, raw_dir, user_id=_USER_ID)

    count = conn.execute(
        "SELECT count(*) FROM activities WHERE user_id = ?", [_USER_ID]
    ).fetchone()[0]
    assert count == 1


def test_build_activities_table_does_not_erase_another_users_rows(tmp_path) -> None:
    """Le cas critique du multi-utilisateur (T-44b) : synchroniser
    l'utilisateur B ne doit RIEN effacer des activités déjà en base pour
    l'utilisateur A."""
    raw_dir_a = tmp_path / "activities_a"
    raw_dir_b = tmp_path / "activities_b"
    _write_raw_activities(raw_dir_a, [_raw_ride(1)])
    _write_raw_activities(raw_dir_b, [_raw_ride(2)])

    conn = duckdb.connect(":memory:")
    build_activities_table(conn, raw_dir_a, user_id=_USER_ID)
    build_activities_table(conn, raw_dir_b, user_id=_OTHER_USER_ID)

    rows = conn.execute("SELECT id, user_id FROM activities ORDER BY id").fetchall()
    assert rows == [(1, _USER_ID), (2, _OTHER_USER_ID)]
