"""Tests du dépivotage des streams Strava vers `main.streams` (T-07,
T-44b pour `user_id`).

Le brut stocke 1 ligne par activité avec des colonnes-listes ; la table
transformée stocke 1 ligne par échantillon temporel (format long).
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segment_predictor.storage.streams import build_streams_table

_USER_ID = 1
_OTHER_USER_ID = 2


def _write_raw_stream(raw_dir, activity_id: int, streams: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([streams]), raw_dir / f"{activity_id}.parquet")


def _full_streams() -> dict:
    return {
        "time": {"data": [0, 1, 2]},
        "watts": {"data": [100, 150, 200]},
        "altitude": {"data": [10.0, 10.5, 11.0]},
        "latlng": {"data": [[45.10, 5.20], [45.11, 5.21], [45.12, 5.22]]},
        "distance": {"data": [0.0, 5.0, 10.0]},
        "heartrate": {"data": [120, 125, 130]},
        "cadence": {"data": [80, 82, 85]},
    }


def test_build_streams_table_unnests_one_row_per_sample(tmp_path) -> None:
    raw_dir = tmp_path / "streams"
    _write_raw_stream(raw_dir, 111, _full_streams())

    conn = duckdb.connect(":memory:")
    build_streams_table(conn, raw_dir, user_id=_USER_ID)

    rows = conn.execute(
        "SELECT activity_id, sample_index, t_s, watts, altitude_m, lat, lng, "
        "distance_m, heartrate, cadence, user_id FROM streams ORDER BY sample_index"
    ).fetchall()

    assert rows == [
        (111, 0, 0, 100, 10.0, 45.10, 5.20, 0.0, 120, 80, _USER_ID),
        (111, 1, 1, 150, 10.5, 45.11, 5.21, 5.0, 125, 82, _USER_ID),
        (111, 2, 2, 200, 11.0, 45.12, 5.22, 10.0, 130, 85, _USER_ID),
    ]


def test_build_streams_table_leaves_missing_streams_null(tmp_path) -> None:
    raw_dir = tmp_path / "streams"
    streams = _full_streams()
    del streams["heartrate"]
    del streams["cadence"]
    del streams["latlng"]
    _write_raw_stream(raw_dir, 222, streams)

    conn = duckdb.connect(":memory:")
    build_streams_table(conn, raw_dir, user_id=_USER_ID)

    rows = conn.execute(
        "SELECT heartrate, cadence, lat, lng, watts FROM streams ORDER BY sample_index"
    ).fetchall()

    assert rows == [
        (None, None, None, None, 100),
        (None, None, None, None, 150),
        (None, None, None, None, 200),
    ]


def test_build_streams_table_combines_multiple_activities(tmp_path) -> None:
    raw_dir = tmp_path / "streams"
    _write_raw_stream(raw_dir, 111, _full_streams())
    _write_raw_stream(raw_dir, 222, _full_streams())

    conn = duckdb.connect(":memory:")
    build_streams_table(conn, raw_dir, user_id=_USER_ID)

    total = conn.execute("SELECT count(*) FROM streams").fetchone()[0]
    assert total == 6  # 3 échantillons x 2 activités

    distinct_activities = conn.execute(
        "SELECT count(DISTINCT activity_id) FROM streams"
    ).fetchone()[0]
    assert distinct_activities == 2


def test_build_streams_table_resync_replaces_only_that_users_rows(tmp_path) -> None:
    raw_dir = tmp_path / "streams"
    _write_raw_stream(raw_dir, 111, _full_streams())

    conn = duckdb.connect(":memory:")
    build_streams_table(conn, raw_dir, user_id=_USER_ID)
    build_streams_table(conn, raw_dir, user_id=_USER_ID)

    total = conn.execute("SELECT count(*) FROM streams").fetchone()[0]
    assert total == 3  # pas doublé


def test_build_streams_table_does_not_erase_another_users_rows(tmp_path) -> None:
    """Cas critique du multi-utilisateur (T-44b) : synchroniser
    l'utilisateur B ne doit rien effacer des streams déjà en base pour A."""
    raw_dir_a = tmp_path / "streams_a"
    raw_dir_b = tmp_path / "streams_b"
    _write_raw_stream(raw_dir_a, 111, _full_streams())
    _write_raw_stream(raw_dir_b, 222, _full_streams())

    conn = duckdb.connect(":memory:")
    build_streams_table(conn, raw_dir_a, user_id=_USER_ID)
    build_streams_table(conn, raw_dir_b, user_id=_OTHER_USER_ID)

    rows = conn.execute(
        "SELECT DISTINCT activity_id, user_id FROM streams ORDER BY activity_id"
    ).fetchall()
    assert rows == [(111, _USER_ID), (222, _OTHER_USER_ID)]


def test_build_streams_table_rejects_misaligned_stream(tmp_path) -> None:
    """Le brut ne devrait jamais avoir un stream plus court/long que `time`
    (vérifié sur les 497 activités réelles) — si ça arrive quand même,
    on refuse de dépivoter plutôt que de décaler silencieusement les points.
    """
    raw_dir = tmp_path / "streams"
    streams = _full_streams()
    streams["watts"] = {"data": [100, 150]}  # 2 points au lieu de 3
    _write_raw_stream(raw_dir, 111, streams)

    conn = duckdb.connect(":memory:")
    with pytest.raises(ValueError, match="watts"):
        build_streams_table(conn, raw_dir, user_id=_USER_ID)


def test_build_streams_table_rejects_missing_time_stream(tmp_path) -> None:
    raw_dir = tmp_path / "streams"
    streams = _full_streams()
    del streams["time"]
    _write_raw_stream(raw_dir, 111, streams)

    conn = duckdb.connect(":memory:")
    with pytest.raises(ValueError, match="time"):
        build_streams_table(conn, raw_dir, user_id=_USER_ID)
