"""Tests de `user_segment_stats` et `user_starred_segments` (T-44c).

Le PR (`athlete_segment_stats`) est une donnée PAR ATHLÈTE — reflète le
token qui a fait la requête `GET /segments/{id}`, pas une propriété du
segment. Avant T-44c, il vivait à tort dans `segments` (voir
test_segments.py, qui ne teste plus que les faits physiques partagés).
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segment_predictor.storage.segments import (
    build_user_segment_stats_table,
    build_user_starred_segments_table,
)

_USER_ID = 1
_OTHER_USER_ID = 2


def _write_raw_segment(raw_dir, segment: dict, filename: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([segment]), raw_dir / filename)


def _segment_with_stats(segment_id: int, **stats_overrides) -> dict:
    stats = {"pr_elapsed_time": 252, "pr_date": "2023-07-14T09:15:00Z", "effort_count": 8}
    stats.update(stats_overrides)
    return {
        "id": segment_id,
        "name": "Segment",
        "distance": 1000.0,
        "average_grade": 5.0,
        "total_elevation_gain": 50.0,
        "start_latlng": [47.5, 7.4],
        "end_latlng": [47.51, 7.41],
        "map": {"polyline": "fake_polyline"},
        "xoms": {"kom": "1:00"},
        "athlete_segment_stats": stats,
    }


# ---- build_user_segment_stats_table ----------------------------------------------------


def test_build_user_segment_stats_table_reads_pr_and_effort_count(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    _write_raw_segment(raw_dir, _segment_with_stats(229781), "229781.parquet")

    conn = duckdb.connect(":memory:")
    build_user_segment_stats_table(conn, raw_dir, user_id=_USER_ID)

    row = conn.execute(
        "SELECT user_id, segment_id, pr_seconds, pr_date, effort_count FROM user_segment_stats"
    ).fetchone()
    assert row == (_USER_ID, 229781, 252, "2023-07-14T09:15:00Z", 8)


def test_build_user_segment_stats_table_handles_never_ridden_segment_across_files(
    tmp_path,
) -> None:
    """Régression T-06 : un segment jamais roulé a `athlete_segment_stats.
    pr_*` à None, ce qui fait inférer un type `null` par pyarrow pour CE
    fichier — incompatible avec le type réel (int64/string) d'un autre
    fichier où le champ est renseigné. Doit lire les deux malgré tout."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        _segment_with_stats(1, pr_elapsed_time=None, pr_date=None, effort_count=0),
        "1.parquet",
    )
    _write_raw_segment(
        raw_dir,
        _segment_with_stats(2, pr_elapsed_time=180, pr_date="2024-01-01T00:00:00Z", effort_count=5),
        "2.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_user_segment_stats_table(conn, raw_dir, user_id=_USER_ID)  # ne doit pas lever

    rows = conn.execute(
        "SELECT segment_id, pr_seconds, pr_date FROM user_segment_stats ORDER BY segment_id"
    ).fetchall()
    assert rows == [(1, None, None), (2, 180, "2024-01-01T00:00:00Z")]


def test_build_user_segment_stats_table_raises_when_stats_missing(tmp_path) -> None:
    """Pas de valeur par défaut silencieuse : un segment mal formé lève, il n'est pas ignoré."""
    raw_dir = tmp_path / "segments"
    segment = _segment_with_stats(1)
    del segment["athlete_segment_stats"]
    _write_raw_segment(raw_dir, segment, "1.parquet")

    conn = duckdb.connect(":memory:")
    with pytest.raises(KeyError):
        build_user_segment_stats_table(conn, raw_dir, user_id=_USER_ID)


def test_build_user_segment_stats_table_resync_replaces_only_that_users_rows(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    _write_raw_segment(raw_dir, _segment_with_stats(1), "1.parquet")

    conn = duckdb.connect(":memory:")
    build_user_segment_stats_table(conn, raw_dir, user_id=_USER_ID)
    build_user_segment_stats_table(conn, raw_dir, user_id=_USER_ID)

    count = conn.execute("SELECT count(*) FROM user_segment_stats").fetchone()[0]
    assert count == 1


def test_build_user_segment_stats_table_does_not_erase_another_users_stats(tmp_path) -> None:
    """Cas critique du multi-utilisateur (T-44c) : deux utilisateurs
    peuvent avoir chacun un PR sur le MÊME segment partagé — synchroniser
    B ne doit ni effacer ni écraser le PR de A sur ce segment."""
    raw_dir_a = tmp_path / "segments_a"
    raw_dir_b = tmp_path / "segments_b"
    _write_raw_segment(raw_dir_a, _segment_with_stats(1, pr_elapsed_time=252), "1.parquet")
    _write_raw_segment(raw_dir_b, _segment_with_stats(1, pr_elapsed_time=300), "1.parquet")

    conn = duckdb.connect(":memory:")
    build_user_segment_stats_table(conn, raw_dir_a, user_id=_USER_ID)
    build_user_segment_stats_table(conn, raw_dir_b, user_id=_OTHER_USER_ID)

    rows = conn.execute(
        "SELECT user_id, pr_seconds FROM user_segment_stats WHERE segment_id = 1 ORDER BY user_id"
    ).fetchall()
    assert rows == [(_USER_ID, 252), (_OTHER_USER_ID, 300)]


# ---- build_user_starred_segments_table -------------------------------------------------


def test_build_user_starred_segments_table_records_every_segment_in_raw_dir(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    _write_raw_segment(raw_dir, _segment_with_stats(1), "1.parquet")
    _write_raw_segment(raw_dir, _segment_with_stats(2), "2.parquet")

    conn = duckdb.connect(":memory:")
    build_user_starred_segments_table(conn, raw_dir, user_id=_USER_ID)

    rows = conn.execute(
        "SELECT user_id, segment_id FROM user_starred_segments ORDER BY segment_id"
    ).fetchall()
    assert rows == [(_USER_ID, 1), (_USER_ID, 2)]


def test_build_user_starred_segments_table_does_not_erase_another_users_stars(tmp_path) -> None:
    raw_dir_a = tmp_path / "segments_a"
    raw_dir_b = tmp_path / "segments_b"
    _write_raw_segment(raw_dir_a, _segment_with_stats(1), "1.parquet")
    _write_raw_segment(raw_dir_b, _segment_with_stats(2), "2.parquet")

    conn = duckdb.connect(":memory:")
    build_user_starred_segments_table(conn, raw_dir_a, user_id=_USER_ID)
    build_user_starred_segments_table(conn, raw_dir_b, user_id=_OTHER_USER_ID)

    rows = conn.execute(
        "SELECT user_id, segment_id FROM user_starred_segments ORDER BY user_id"
    ).fetchall()
    assert rows == [(_USER_ID, 1), (_OTHER_USER_ID, 2)]


def test_build_user_starred_segments_table_resync_replaces_only_that_users_rows(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    _write_raw_segment(raw_dir, _segment_with_stats(1), "1.parquet")

    conn = duckdb.connect(":memory:")
    build_user_starred_segments_table(conn, raw_dir, user_id=_USER_ID)
    build_user_starred_segments_table(conn, raw_dir, user_id=_USER_ID)

    count = conn.execute("SELECT count(*) FROM user_starred_segments").fetchone()[0]
    assert count == 1
