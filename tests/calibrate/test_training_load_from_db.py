"""Tests de l'agrégation du TSS quotidien et de la charge d'entraînement
depuis la DB réelle (T-21)."""

from datetime import date, datetime, timedelta

import duckdb
import pyarrow as pa
import pytest

from segment_predictor.calibrate.training_load import (
    compute_daily_tss,
    compute_training_load_from_db,
    fill_daily_tss_series,
)
from segment_predictor.models.power import CriticalPowerFit

_EMPTY_SCHEMAS = {
    "activities": pa.schema(
        [
            ("id", pa.int64()),
            ("type", pa.string()),
            ("device_watts", pa.bool_()),
            ("start_date", pa.timestamp("us")),
            ("moving_time_s", pa.int64()),
        ]
    ),
    "streams": pa.schema(
        [
            ("activity_id", pa.int64()),
            ("sample_index", pa.int64()),
            ("t_s", pa.int64()),
            ("watts", pa.float64()),
        ]
    ),
}

_CP_FIT = CriticalPowerFit(
    cp_watts=250.0, w_prime_joules=18_000.0, r_squared=0.9, n_points=5, duration_range_s=(180, 1200)
)


def _make_db(activities: list[dict], streams: list[dict]):
    conn = duckdb.connect(":memory:")
    for name, rows in (("activities", activities), ("streams", streams)):
        table = (
            pa.Table.from_pylist(rows)
            if rows
            else pa.Table.from_pylist([], schema=_EMPTY_SCHEMAS[name])
        )
        conn.register("_rows", table)
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM _rows")
        conn.unregister("_rows")
    return conn


def _flat_stream_rows(activity_id: int, watts: float, n_seconds: int) -> list[dict]:
    return [
        {"activity_id": activity_id, "sample_index": i, "t_s": i, "watts": watts}
        for i in range(n_seconds)
    ]


# ---- compute_daily_tss ------------------------------------------------------------------------


def test_compute_daily_tss_sums_multiple_activities_on_the_same_day() -> None:
    same_day = datetime(2025, 6, 1, 8, 0, 0)
    activities = [
        {
            "id": 1,
            "type": "Ride",
            "device_watts": True,
            "start_date": same_day,
            "moving_time_s": 3600,
        },
        {
            "id": 2,
            "type": "Ride",
            "device_watts": True,
            "start_date": same_day.replace(hour=17),  # même jour, sortie du soir
            "moving_time_s": 3600,
        },
    ]
    # 250W constant pile au seuil : TSS = 100 par activité (définition Coggan)
    streams = _flat_stream_rows(1, 250.0, 3600) + _flat_stream_rows(2, 250.0, 3600)
    conn = _make_db(activities, streams)

    daily = compute_daily_tss(conn, threshold_power_w=250.0)

    assert daily[date(2025, 6, 1)] == pytest.approx(200.0, abs=0.5)  # 100 + 100, pas remplacé


def test_compute_daily_tss_ignores_non_cycling_activities() -> None:
    activities = [
        {
            "id": 1,
            "type": "Hike",
            "device_watts": True,
            "start_date": datetime(2025, 6, 1),
            "moving_time_s": 3600,
        }
    ]
    streams = _flat_stream_rows(1, 250.0, 3600)
    conn = _make_db(activities, streams)

    daily = compute_daily_tss(conn, threshold_power_w=250.0)

    assert daily == {}


def test_compute_daily_tss_skips_activity_too_short_for_normalized_power() -> None:
    """Une activité de 10s ne permet pas de fenêtre de 30s (normalized_power,
    T-21) : elle doit être ignorée, pas faire planter tout le calcul."""
    activities = [
        {
            "id": 1,
            "type": "Ride",
            "device_watts": True,
            "start_date": datetime(2025, 6, 1),
            "moving_time_s": 10,
        },
        {
            "id": 2,
            "type": "Ride",
            "device_watts": True,
            "start_date": datetime(2025, 6, 2),
            "moving_time_s": 3600,
        },
    ]
    streams = _flat_stream_rows(1, 250.0, 10) + _flat_stream_rows(2, 250.0, 3600)
    conn = _make_db(activities, streams)

    daily = compute_daily_tss(conn, threshold_power_w=250.0)

    assert date(2025, 6, 1) not in daily
    assert daily[date(2025, 6, 2)] == pytest.approx(100.0, abs=0.5)


# ---- fill_daily_tss_series --------------------------------------------------------------------


def test_fill_daily_tss_series_fills_missing_days_with_zero() -> None:
    daily = {date(2025, 6, 1): 50.0, date(2025, 6, 3): 80.0}

    series = fill_daily_tss_series(daily, date(2025, 6, 1), date(2025, 6, 3))

    assert series == [50.0, 0.0, 80.0]


def test_fill_daily_tss_series_raises_when_end_before_start() -> None:
    with pytest.raises(ValueError, match="end"):
        fill_daily_tss_series({}, date(2025, 6, 3), date(2025, 6, 1))


# ---- compute_training_load_from_db -------------------------------------------------------------


def test_compute_training_load_from_db_returns_one_point_per_day_in_range() -> None:
    activities = [
        {
            "id": 1,
            "type": "Ride",
            "device_watts": True,
            "start_date": datetime(2025, 6, 1),
            "moving_time_s": 3600,
        },
        {
            "id": 2,
            "type": "Ride",
            "device_watts": True,
            "start_date": datetime(2025, 6, 5),
            "moving_time_s": 3600,
        },
    ]
    streams = _flat_stream_rows(1, 250.0, 3600) + _flat_stream_rows(2, 250.0, 3600)
    conn = _make_db(activities, streams)

    dates, points = compute_training_load_from_db(conn, cp_fit=_CP_FIT)

    assert dates == [date(2025, 6, 1) + timedelta(days=i) for i in range(5)]
    assert len(points) == 5


def test_compute_training_load_from_db_raises_when_no_eligible_activity() -> None:
    conn = _make_db([], [])

    with pytest.raises(ValueError, match="activité"):
        compute_training_load_from_db(conn, cp_fit=_CP_FIT)
