"""Tests de la série temporelle de l'indice de performance (T-23)."""

from datetime import date, datetime

import duckdb
import pyarrow as pa
import pytest

from segment_predictor.calibrate.form import (
    build_form_regression_dataset,
    compute_performance_index_series,
    recent_performance_index_values,
)
from segment_predictor.calibrate.training_load import compute_training_load_from_db
from segment_predictor.models.power import CriticalPowerFit

_CP_FIT = CriticalPowerFit(
    cp_watts=250.0, w_prime_joules=20_000.0, r_squared=0.9, n_points=5, duration_range_s=(180, 1200)
)


def _make_db(activities: list[dict], streams: list[dict]):
    conn = duckdb.connect(":memory:")
    for name, rows in (("activities", activities), ("streams", streams)):
        conn.register("_rows", pa.Table.from_pylist(rows))
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM _rows")
        conn.unregister("_rows")
    return conn


def _flat_stream_rows(activity_id: int, watts: float, n_seconds: int) -> list[dict]:
    return [
        {"activity_id": activity_id, "sample_index": i, "t_s": i, "watts": watts}
        for i in range(n_seconds)
    ]


def _activity(activity_id: int, start_date: datetime, moving_time_s: int = 3600) -> dict:
    return {
        "id": activity_id,
        "type": "Ride",
        "device_watts": True,
        "start_date": start_date,
        "moving_time_s": moving_time_s,
    }


def test_first_ever_activity_at_a_duration_counts_as_maximal() -> None:
    """Rien à comparer encore : le premier effort connu à une durée donnée
    est par définition le record glissant, donc maximal."""
    activities = [_activity(1, datetime(2024, 1, 1))]
    streams = _flat_stream_rows(1, 300.0, 300)  # 300W pendant 300s
    conn = _make_db(activities, streams)

    points = compute_performance_index_series(conn, cp_fit=_CP_FIT, durations_s=[300])

    assert len(points) == 1
    assert points[0].date == date(2024, 1, 1)
    assert points[0].duration_s == 300
    assert points[0].actual_power_w == pytest.approx(300.0)
    expected_index = 300.0 / (_CP_FIT.cp_watts + _CP_FIT.w_prime_joules / 300)
    assert points[0].index == pytest.approx(expected_index)


def test_training_pace_activity_is_excluded_from_the_series() -> None:
    """Une sortie bien en dessous du record glissant (< 95%) n'est pas un
    effort maximal : elle ne doit pas apparaître dans la série."""
    activities = [_activity(1, datetime(2024, 1, 1)), _activity(2, datetime(2024, 1, 2))]
    streams = _flat_stream_rows(1, 300.0, 300) + _flat_stream_rows(2, 150.0, 300)  # rythme cool
    conn = _make_db(activities, streams)

    points = compute_performance_index_series(conn, cp_fit=_CP_FIT, durations_s=[300])

    assert len(points) == 1  # seulement l'activité 1
    assert points[0].date == date(2024, 1, 1)


def test_running_best_follows_chronological_order_not_insertion_order() -> None:
    """Les activités sont insérées dans le désordre chronologique : le
    calcul doit quand même suivre l'ordre des dates, pas l'ordre des lignes."""
    activities = [
        _activity(2, datetime(2024, 1, 5)),  # 2e chronologiquement, insérée en premier
        _activity(1, datetime(2024, 1, 1)),  # 1er chronologiquement
    ]
    # activité 1 (1er janvier) : record initial à 300W
    # activité 2 (5 janvier) : 290W, soit 96.7% du record glissant -> maximal
    streams = _flat_stream_rows(1, 300.0, 300) + _flat_stream_rows(2, 290.0, 300)
    conn = _make_db(activities, streams)

    points = compute_performance_index_series(conn, cp_fit=_CP_FIT, durations_s=[300])

    assert [p.date for p in points] == [date(2024, 1, 1), date(2024, 1, 5)]


def test_skips_durations_outside_cp_validity_range() -> None:
    activities = [_activity(1, datetime(2024, 1, 1))]
    streams = _flat_stream_rows(1, 400.0, 60)  # 60s, hors plage [180, 1200]
    conn = _make_db(activities, streams)

    points = compute_performance_index_series(conn, cp_fit=_CP_FIT, durations_s=[60])

    assert points == []


def test_skips_activity_too_short_for_the_stream() -> None:
    activities = [_activity(1, datetime(2024, 1, 1)), _activity(2, datetime(2024, 1, 2))]
    # activité 1 trop courte pour le stream
    streams = _flat_stream_rows(1, 300.0, 10) + _flat_stream_rows(2, 300.0, 300)
    conn = _make_db(activities, streams)

    points = compute_performance_index_series(conn, cp_fit=_CP_FIT, durations_s=[300])

    assert len(points) == 1
    assert points[0].date == date(2024, 1, 2)


def test_ignores_non_cycling_activities() -> None:
    activities = [
        {"id": 1, "type": "Hike", "device_watts": True, "start_date": datetime(2024, 1, 1)}
    ]
    streams = _flat_stream_rows(1, 300.0, 300)
    conn = _make_db(activities, streams)

    points = compute_performance_index_series(conn, cp_fit=_CP_FIT, durations_s=[300])

    assert points == []


# ---- build_form_regression_dataset -----------------------------------------------------------


def test_build_form_regression_dataset_joins_index_with_same_date_training_load() -> None:
    activities = [
        _activity(1, datetime(2024, 1, 1), moving_time_s=300),
        _activity(2, datetime(2024, 1, 2), moving_time_s=300),
    ]
    streams = _flat_stream_rows(1, 300.0, 300) + _flat_stream_rows(2, 305.0, 300)
    conn = _make_db(activities, streams)

    x, y, dates = build_form_regression_dataset(conn, cp_fit=_CP_FIT, durations_s=[300])

    assert x.shape == (2, 4)  # ctl, atl, tsb, duration_s
    assert dates == [date(2024, 1, 1), date(2024, 1, 2)]
    assert (x[:, 3] == 300).all()  # duration_s

    load_dates, load_points = compute_training_load_from_db(conn, cp_fit=_CP_FIT)
    load_by_date = dict(zip(load_dates, load_points, strict=True))
    for i, day in enumerate(dates):
        assert x[i, 0] == pytest.approx(load_by_date[day].ctl)
        assert x[i, 1] == pytest.approx(load_by_date[day].atl)
        assert x[i, 2] == pytest.approx(load_by_date[day].tsb)

    index_points = compute_performance_index_series(conn, cp_fit=_CP_FIT, durations_s=[300])
    assert list(y) == pytest.approx([p.index for p in index_points])


def test_build_form_regression_dataset_raises_when_no_index_points() -> None:
    activities = [_activity(1, datetime(2024, 1, 1), moving_time_s=60)]
    streams = _flat_stream_rows(1, 300.0, 60)  # < 180s : hors plage de validité CP
    conn = _make_db(activities, streams)

    with pytest.raises(ValueError, match="indice de performance"):
        build_form_regression_dataset(conn, cp_fit=_CP_FIT, durations_s=[60])


# ---- recent_performance_index_values (T-28) ----------------------------------------------------


def test_recent_performance_index_values_excludes_points_older_than_the_window() -> None:
    activities = [
        _activity(1, datetime(2024, 1, 1), moving_time_s=300),  # trop ancien
        _activity(2, datetime(2024, 5, 1), moving_time_s=300),  # dans la fenêtre
    ]
    streams = _flat_stream_rows(1, 300.0, 300) + _flat_stream_rows(2, 305.0, 300)
    conn = _make_db(activities, streams)

    values = recent_performance_index_values(
        conn,
        cp_fit=_CP_FIT,
        window_days=90,
        reference_date=date(2024, 5, 15),
        durations_s=[300],
    )

    index_points = compute_performance_index_series(conn, cp_fit=_CP_FIT, durations_s=[300])
    expected = [p.index for p in index_points if p.date == date(2024, 5, 1)]
    assert values == pytest.approx(expected)
    assert len(values) == 1  # pas l'activité de janvier


def test_recent_performance_index_values_defaults_reference_date_to_today() -> None:
    activities = [_activity(1, datetime.now(), moving_time_s=300)]
    streams = _flat_stream_rows(1, 300.0, 300)
    conn = _make_db(activities, streams)

    values = recent_performance_index_values(conn, cp_fit=_CP_FIT, durations_s=[300])

    assert len(values) == 1


def test_recent_performance_index_values_returns_empty_list_outside_window() -> None:
    activities = [_activity(1, datetime(2020, 1, 1), moving_time_s=300)]
    streams = _flat_stream_rows(1, 300.0, 300)
    conn = _make_db(activities, streams)

    values = recent_performance_index_values(
        conn, cp_fit=_CP_FIT, window_days=90, reference_date=date(2024, 5, 15), durations_s=[300]
    )

    assert values == []


def test_recent_performance_index_values_raises_on_non_positive_window() -> None:
    activities = [_activity(1, datetime(2024, 1, 1), moving_time_s=300)]
    streams = _flat_stream_rows(1, 300.0, 300)
    conn = _make_db(activities, streams)

    with pytest.raises(ValueError, match="window_days"):
        recent_performance_index_values(conn, cp_fit=_CP_FIT, window_days=0)
