"""Tests de l'enrichissement météo des activités (T-14).

C'est ici, pas dans ingest, que les unités passent en SI (km/h -> m/s,
°C -> K, hPa -> Pa) et que l'heure exacte de chaque activité est
interpolée entre les relevés horaires de sa zone.
"""

from datetime import datetime

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segment_predictor.storage.weather import build_activity_weather_table


def _write_zone_weather(raw_dir, zone_lat: float, zone_lng: float, hourly: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([{"hourly": hourly}]),
        raw_dir / f"{zone_lat:.1f}_{zone_lng:.1f}.parquet",
    )


def _make_activities_table(conn: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    """`type` par défaut à "Ride" (le filtre Ride/VirtualRide n'est pas ce
    qu'on teste dans la plupart de ces tests) — une ligne peut le surcharger."""
    rows_with_type = [{"type": "Ride", **row} for row in rows]
    conn.register("_rows", pa.Table.from_pylist(rows_with_type))
    conn.execute("CREATE TABLE activities AS SELECT * FROM _rows")
    conn.unregister("_rows")


def test_build_activity_weather_table_interpolates_to_the_exact_hour(tmp_path) -> None:
    raw_dir = tmp_path / "weather"
    _write_zone_weather(
        raw_dir,
        47.7,
        7.4,
        {
            "time": ["2024-01-01T10:00", "2024-01-01T11:00"],
            "temperature_2m": [10.0, 14.0],  # +4°C sur l'heure
            "relative_humidity_2m": [60.0, 80.0],
            "surface_pressure": [1000.0, 1010.0],
            "wind_speed_10m": [10.0, 20.0],  # km/h
            "wind_direction_10m": [90.0, 90.0],
        },
    )
    conn = duckdb.connect(":memory:")
    _make_activities_table(
        conn,
        [
            {
                "id": 1,
                "start_date": datetime(2024, 1, 1, 10, 30, 0),  # pile à mi-chemin
                "start_lat": 47.71,
                "start_lng": 7.41,
            }
        ],
    )

    build_activity_weather_table(conn, raw_dir)

    row = conn.execute(
        "SELECT activity_id, temperature_k, relative_humidity_pct, pressure_pa, "
        "wind_speed_ms, wind_direction_rad FROM activity_weather"
    ).fetchone()

    assert row[0] == 1
    assert row[1] == pytest.approx(12.0 + 273.15)  # mi-chemin entre 10 et 14°C
    assert row[2] == pytest.approx(70.0)  # mi-chemin entre 60 et 80%
    assert row[3] == pytest.approx(100_500.0)  # mi-chemin entre 1000 et 1010 hPa, en Pa
    assert row[4] == pytest.approx(15.0 / 3.6)  # mi-chemin entre 10 et 20 km/h, converti en m/s
    assert row[5] == pytest.approx(1.5708, abs=1e-3)  # 90° constant -> pi/2


def test_build_activity_weather_table_interpolates_wind_direction_across_the_north_wrap(
    tmp_path,
) -> None:
    """350° puis 10° : une interpolation naïve donnerait 180° (faux, plein sud).
    La bonne réponse est ~0°/360° (le vent a juste traversé le nord)."""
    raw_dir = tmp_path / "weather"
    _write_zone_weather(
        raw_dir,
        47.7,
        7.4,
        {
            "time": ["2024-01-01T10:00", "2024-01-01T11:00"],
            "temperature_2m": [10.0, 10.0],
            "relative_humidity_2m": [60.0, 60.0],
            "surface_pressure": [1000.0, 1000.0],
            "wind_speed_10m": [10.0, 10.0],
            "wind_direction_10m": [350.0, 10.0],
        },
    )
    conn = duckdb.connect(":memory:")
    _make_activities_table(
        conn,
        [
            {
                "id": 1,
                "start_date": datetime(2024, 1, 1, 10, 30, 0),
                "start_lat": 47.7,
                "start_lng": 7.4,
            }
        ],
    )

    build_activity_weather_table(conn, raw_dir)

    wind_direction_rad = conn.execute("SELECT wind_direction_rad FROM activity_weather").fetchone()[
        0
    ]

    # ~0 rad (nord), certainement pas ~pi (sud, ce que donnerait une interpolation naïve)
    assert wind_direction_rad == pytest.approx(0.0, abs=0.1) or wind_direction_rad == pytest.approx(
        2 * 3.14159265, abs=0.1
    )


def test_build_activity_weather_table_skips_activities_without_position(tmp_path) -> None:
    raw_dir = tmp_path / "weather"
    _write_zone_weather(
        raw_dir,
        47.7,
        7.4,
        {
            "time": ["2024-01-01T10:00", "2024-01-01T11:00"],
            "temperature_2m": [10.0, 14.0],
            "relative_humidity_2m": [60.0, 80.0],
            "surface_pressure": [1000.0, 1010.0],
            "wind_speed_10m": [10.0, 20.0],
            "wind_direction_10m": [90.0, 90.0],
        },
    )
    conn = duckdb.connect(":memory:")
    _make_activities_table(
        conn,
        [
            {
                "id": 1,
                "start_date": datetime(2024, 1, 1, 10, 30, 0),
                "start_lat": 47.7,
                "start_lng": 7.4,
            },
            {
                "id": 2,
                "start_date": datetime(2024, 1, 1, 10, 30, 0),
                "start_lat": None,
                "start_lng": None,
            },
        ],
    )

    build_activity_weather_table(conn, raw_dir)

    ids = [r[0] for r in conn.execute("SELECT activity_id FROM activity_weather").fetchall()]
    assert ids == [1]


def test_build_activity_weather_table_skips_non_cycling_activities(tmp_path) -> None:
    """Hors périmètre du projet : une randonnée n'est pas enrichie même si sa
    zone météo est disponible."""
    raw_dir = tmp_path / "weather"
    _write_zone_weather(
        raw_dir,
        47.7,
        7.4,
        {
            "time": ["2024-01-01T10:00", "2024-01-01T11:00"],
            "temperature_2m": [10.0, 14.0],
            "relative_humidity_2m": [60.0, 80.0],
            "surface_pressure": [1000.0, 1010.0],
            "wind_speed_10m": [10.0, 20.0],
            "wind_direction_10m": [90.0, 90.0],
        },
    )
    conn = duckdb.connect(":memory:")
    _make_activities_table(
        conn,
        [
            {
                "id": 1,
                "type": "Hike",
                "start_date": datetime(2024, 1, 1, 10, 30, 0),
                "start_lat": 47.7,
                "start_lng": 7.4,
            }
        ],
    )

    build_activity_weather_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM activity_weather").fetchone()[0]
    assert count == 0


def test_build_activity_weather_table_skips_activities_with_no_zone_data(tmp_path) -> None:
    """Zone jamais téléchargée : on saute l'activité plutôt que d'inventer une valeur."""
    raw_dir = tmp_path / "weather"
    conn = duckdb.connect(":memory:")
    _make_activities_table(
        conn,
        [
            {
                "id": 1,
                "start_date": datetime(2024, 1, 1, 10, 30, 0),
                "start_lat": 47.7,
                "start_lng": 7.4,
            }
        ],
    )

    build_activity_weather_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM activity_weather").fetchone()[0]
    assert count == 0


def test_build_activity_weather_table_skips_activities_outside_covered_date_range(tmp_path) -> None:
    raw_dir = tmp_path / "weather"
    _write_zone_weather(
        raw_dir,
        47.7,
        7.4,
        {
            "time": ["2024-01-01T10:00", "2024-01-01T11:00"],
            "temperature_2m": [10.0, 14.0],
            "relative_humidity_2m": [60.0, 80.0],
            "surface_pressure": [1000.0, 1010.0],
            "wind_speed_10m": [10.0, 20.0],
            "wind_direction_10m": [90.0, 90.0],
        },
    )
    conn = duckdb.connect(":memory:")
    _make_activities_table(
        conn,
        [
            {
                "id": 1,
                "start_date": datetime(2025, 6, 1, 10, 30, 0),  # bien après la plage couverte
                "start_lat": 47.7,
                "start_lng": 7.4,
            }
        ],
    )

    build_activity_weather_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM activity_weather").fetchone()[0]
    assert count == 0


def test_build_activity_weather_table_replaces_existing_table(tmp_path) -> None:
    raw_dir = tmp_path / "weather"
    _write_zone_weather(
        raw_dir,
        47.7,
        7.4,
        {
            "time": ["2024-01-01T10:00", "2024-01-01T11:00"],
            "temperature_2m": [10.0, 14.0],
            "relative_humidity_2m": [60.0, 80.0],
            "surface_pressure": [1000.0, 1010.0],
            "wind_speed_10m": [10.0, 20.0],
            "wind_direction_10m": [90.0, 90.0],
        },
    )
    conn = duckdb.connect(":memory:")
    _make_activities_table(
        conn,
        [
            {
                "id": 1,
                "start_date": datetime(2024, 1, 1, 10, 30, 0),
                "start_lat": 47.7,
                "start_lng": 7.4,
            }
        ],
    )

    build_activity_weather_table(conn, raw_dir)
    build_activity_weather_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM activity_weather").fetchone()[0]
    assert count == 1
