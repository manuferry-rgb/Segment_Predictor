"""Tests des vues `raw.*` — passthrough SQL sur le Parquet brut, sans transformation."""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from segment_predictor.storage.raw_views import create_raw_views


def test_create_raw_views_exposes_activities_and_segments_untransformed(tmp_path) -> None:
    activities_dir = tmp_path / "strava_activities"
    activities_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"id": 1, "type": "Ride", "distance": 1000.0}]),
        activities_dir / "activities.parquet",
    )

    segments_dir = tmp_path / "strava_segments"
    segments_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"id": 42, "xoms": {"kom": "4:12"}}]),
        segments_dir / "42.parquet",
    )

    streams_dir = tmp_path / "strava_streams"
    streams_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"time": {"data": [0, 1]}}]),
        streams_dir / "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    create_raw_views(conn, activities_dir, streams_dir, segments_dir)

    activity_row = conn.execute("SELECT id, type, distance FROM raw.activities").fetchone()
    assert activity_row == (1, "Ride", 1000.0)

    segment_row = conn.execute("SELECT id, xoms.kom AS kom FROM raw.segments").fetchone()
    assert segment_row == (42, "4:12")

    stream_row = conn.execute("SELECT time.data AS time_data FROM raw.streams").fetchone()
    assert stream_row == ([0, 1],)


def test_create_raw_views_is_idempotent(tmp_path) -> None:
    dirs = {}
    for name in ("strava_activities", "strava_segments", "strava_streams"):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{"id": 1}]), directory / "f.parquet")
        dirs[name] = directory

    args = (dirs["strava_activities"], dirs["strava_streams"], dirs["strava_segments"])
    conn = duckdb.connect(":memory:")
    create_raw_views(conn, *args)
    create_raw_views(conn, *args)  # ne doit pas lever

    count = conn.execute("SELECT count(*) FROM raw.activities").fetchone()[0]
    assert count == 1
