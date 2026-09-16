"""Tests du dépivotage du profil de segment Strava vers `main.segment_streams` (T-51b).

Même principe que streams.py (T-07) : le brut stocke 1 ligne par segment
avec des colonnes-listes, la table transformée stocke 1 ligne par
échantillon le long du tracé, format long. Différences avec les streams
d'ACTIVITÉ : pas de `time` (un segment n'a pas d'horodatage propre, le
profil est indexé par `distance`), pas de `user_id` (table PARTAGÉE, le
tracé d'un segment est le même pour tout le monde, T-44d/T-44c).
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segment_predictor.storage.segment_streams import build_segment_streams_table


def _write_raw_segment_streams(raw_dir, segment_id: int, streams: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([streams]), raw_dir / f"{segment_id}.parquet")


def _full_streams() -> dict:
    return {
        "distance": {"data": [0.0, 10.0, 20.0]},
        "altitude": {"data": [100.0, 101.0, 103.0]},
        "latlng": {"data": [[45.10, 5.20], [45.11, 5.21], [45.12, 5.22]]},
    }


def test_build_segment_streams_table_unnests_one_row_per_sample(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"
    _write_raw_segment_streams(raw_dir, 229781, _full_streams())

    conn = duckdb.connect(":memory:")
    build_segment_streams_table(conn, raw_dir)

    rows = conn.execute(
        "SELECT segment_id, sample_index, distance_m, altitude_m, lat, lng "
        "FROM segment_streams ORDER BY sample_index"
    ).fetchall()

    assert rows == [
        (229781, 0, 0.0, 100.0, 45.10, 5.20),
        (229781, 1, 10.0, 101.0, 45.11, 5.21),
        (229781, 2, 20.0, 103.0, 45.12, 5.22),
    ]


def test_build_segment_streams_table_leaves_missing_altitude_null(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"
    streams = _full_streams()
    del streams["altitude"]
    _write_raw_segment_streams(raw_dir, 111, streams)

    conn = duckdb.connect(":memory:")
    build_segment_streams_table(conn, raw_dir)

    rows = conn.execute("SELECT altitude_m FROM segment_streams ORDER BY sample_index").fetchall()

    assert rows == [(None,), (None,), (None,)]


def test_build_segment_streams_table_combines_multiple_segments(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"
    _write_raw_segment_streams(raw_dir, 111, _full_streams())
    _write_raw_segment_streams(raw_dir, 222, _full_streams())

    conn = duckdb.connect(":memory:")
    build_segment_streams_table(conn, raw_dir)

    segment_ids = conn.execute(
        "SELECT DISTINCT segment_id FROM segment_streams ORDER BY segment_id"
    ).fetchall()

    assert segment_ids == [(111,), (222,)]


def test_build_segment_streams_table_raises_when_distance_missing(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"
    streams = _full_streams()
    del streams["distance"]
    _write_raw_segment_streams(raw_dir, 111, streams)

    conn = duckdb.connect(":memory:")
    with pytest.raises(ValueError, match="distance"):
        build_segment_streams_table(conn, raw_dir)


def test_build_segment_streams_table_raises_on_misaligned_stream(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"
    streams = _full_streams()
    streams["altitude"]["data"] = [100.0, 101.0]  # 2 points au lieu de 3
    _write_raw_segment_streams(raw_dir, 111, streams)

    conn = duckdb.connect(":memory:")
    with pytest.raises(ValueError, match="alignement"):
        build_segment_streams_table(conn, raw_dir)


def test_build_segment_streams_table_replaces_on_rerun(tmp_path) -> None:
    # CREATE OR REPLACE (table PARTAGÉE, pas de user_id à préserver,
    # contrairement à build_streams_table) : un deuxième appel ne duplique
    # pas les lignes.
    raw_dir = tmp_path / "segment_streams"
    _write_raw_segment_streams(raw_dir, 111, _full_streams())

    conn = duckdb.connect(":memory:")
    build_segment_streams_table(conn, raw_dir)
    build_segment_streams_table(conn, raw_dir)

    count = conn.execute("SELECT COUNT(*) FROM segment_streams").fetchone()[0]
    assert count == 3
