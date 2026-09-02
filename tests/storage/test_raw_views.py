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


def test_create_raw_views_exposes_weather_when_provided(tmp_path) -> None:
    activities_dir = tmp_path / "strava_activities"
    activities_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": 1}]), activities_dir / "a.parquet")
    streams_dir = tmp_path / "strava_streams"
    streams_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"time": {"data": [0]}}]), streams_dir / "1.parquet")
    segments_dir = tmp_path / "strava_segments"
    segments_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": 1}]), segments_dir / "1.parquet")
    weather_dir = tmp_path / "open_meteo"
    weather_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"hourly": {"temperature_2m": [5.0]}}]),
        weather_dir / "47.7_7.4.parquet",
    )

    conn = duckdb.connect(":memory:")
    create_raw_views(conn, activities_dir, streams_dir, segments_dir, weather_dir)

    row = conn.execute("SELECT hourly.temperature_2m AS temps FROM raw.weather").fetchone()
    assert row == ([5.0],)


def test_create_raw_views_exposes_activity_details_when_provided(tmp_path) -> None:
    activities_dir = tmp_path / "strava_activities"
    activities_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": 1}]), activities_dir / "a.parquet")
    streams_dir = tmp_path / "strava_streams"
    streams_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"time": {"data": [0]}}]), streams_dir / "1.parquet")
    segments_dir = tmp_path / "strava_segments"
    segments_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": 1}]), segments_dir / "1.parquet")
    details_dir = tmp_path / "strava_activity_details"
    details_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"id": 1, "segment_efforts": []}]), details_dir / "1.parquet"
    )

    conn = duckdb.connect(":memory:")
    create_raw_views(
        conn,
        activities_dir,
        streams_dir,
        segments_dir,
        activity_details_raw_dir=details_dir,
    )

    row = conn.execute("SELECT id FROM raw.activity_details").fetchone()
    assert row == (1,)


def test_create_raw_views_skips_weather_when_not_provided(tmp_path) -> None:
    activities_dir = tmp_path / "strava_activities"
    activities_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": 1}]), activities_dir / "a.parquet")
    streams_dir = tmp_path / "strava_streams"
    streams_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"time": {"data": [0]}}]), streams_dir / "1.parquet")
    segments_dir = tmp_path / "strava_segments"
    segments_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": 1}]), segments_dir / "1.parquet")

    conn = duckdb.connect(":memory:")
    create_raw_views(conn, activities_dir, streams_dir, segments_dir)  # pas de weather_raw_dir

    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'raw'"
    ).fetchall()
    assert ("weather",) not in tables


def test_create_raw_views_does_not_crash_when_a_source_dir_is_empty_or_missing(tmp_path) -> None:
    """Régression : read_parquet vérifie le glob dès CREATE VIEW (pas
    paresseusement à la requête), donc un dossier pas encore peuplé faisait
    planter toute la construction — repéré en relançant build_database.py
    juste après avoir ajouté raw.weather, avant le premier fetch_weather.py."""
    activities_dir = tmp_path / "strava_activities"
    activities_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"id": 1}]), activities_dir / "a.parquet")
    streams_dir = tmp_path / "strava_streams"  # existe mais vide
    streams_dir.mkdir(parents=True)
    segments_dir = tmp_path / "strava_segments"  # n'existe pas du tout
    weather_dir = tmp_path / "open_meteo"  # n'existe pas du tout

    conn = duckdb.connect(":memory:")
    create_raw_views(
        conn, activities_dir, streams_dir, segments_dir, weather_dir
    )  # ne doit pas lever

    assert conn.execute("SELECT count(*) FROM raw.activities").fetchone()[0] == 1
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'raw'"
    ).fetchall()
    assert ("streams",) not in tables
    assert ("segments",) not in tables
    assert ("weather",) not in tables


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
