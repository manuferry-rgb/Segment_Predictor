"""Tests du balayage "quels segments tenter aujourd'hui" (T-33).

Seule `best_wind_opportunity_today` est testée ici (pure : forecast dict
en entrée, pas d'I/O) — `scan_segments_for_today` combine DB et réseau,
comme `rank_forecast_windows_for_segment` (T-27), vérifiée manuellement
contre la vraie base plutôt qu'en test unitaire (même convention).
"""

from datetime import date, datetime

from segment_predictor.models.segment import SegmentChunk
from segment_predictor.predict.wind_scan import best_wind_opportunity_today

_CHUNKS = [SegmentChunk(0.0, 2000.0, 0.0, heading_rad=0.0)]  # plein nord
_TODAY = date(2026, 9, 10)


def _fake_forecast(
    hours_iso: list[str], wind_speed_kmh: list[float], wind_direction_deg: list[float]
):
    n = len(hours_iso)
    return {
        "hourly": {
            "time": hours_iso,
            "temperature_2m": [15.0] * n,
            "relative_humidity_2m": [60.0] * n,
            "surface_pressure": [1013.0] * n,
            "wind_speed_10m": wind_speed_kmh,
            "wind_direction_10m": wind_direction_deg,
        }
    }


def test_best_wind_opportunity_today_picks_the_most_favorable_hour() -> None:
    # cap plein nord : vent qui vient du nord (0°) = face ; du sud (180°) = dos.
    forecast = _fake_forecast(
        ["2026-09-10T08:00", "2026-09-10T14:00"],
        wind_speed_kmh=[20.0, 20.0],
        wind_direction_deg=[0.0, 180.0],
    )
    now = datetime(2026, 9, 10, 6, 0)

    result = best_wind_opportunity_today(forecast, _CHUNKS, 1, "Test", 2000.0, _TODAY, now)

    assert result is not None
    assert result.best_hour.hour == 14
    assert result.tailwind_fraction == 1.0
    assert result.segment_id == 1
    assert result.segment_name == "Test"
    assert result.distance_m == 2000.0


def test_best_wind_opportunity_today_ignores_hours_on_another_date() -> None:
    """Un créneau plus favorable mais un autre jour ne doit pas être
    choisi : "aujourd'hui" veut dire aujourd'hui, pas "un jour où c'est
    mieux" — cf `rank_forecast_windows_for_segment` qui balaie 10 jours,
    ce module en fait délibérément moins."""
    forecast = _fake_forecast(
        ["2026-09-10T14:00", "2026-09-11T14:00"],
        wind_speed_kmh=[20.0, 20.0],
        wind_direction_deg=[0.0, 180.0],  # 09-10 : face ; 09-11 : dos (mais pas aujourd'hui)
    )
    now = datetime(2026, 9, 10, 6, 0)

    result = best_wind_opportunity_today(forecast, _CHUNKS, 1, "Test", 2000.0, _TODAY, now)

    assert result is not None
    assert result.best_hour.date() == _TODAY
    assert result.tailwind_fraction == 0.0  # seule l'heure du 10 (face) était éligible


def test_best_wind_opportunity_today_excludes_past_hours() -> None:
    """Le créneau du matin est le plus favorable, mais il est déjà passé
    (now=9h) : pas actionnable, donc écarté même s'il est meilleur."""
    forecast = _fake_forecast(
        ["2026-09-10T07:00", "2026-09-10T16:00"],
        wind_speed_kmh=[20.0, 20.0],
        wind_direction_deg=[180.0, 0.0],  # 07h : dos (mais passé) ; 16h : face
    )
    now = datetime(2026, 9, 10, 9, 0)

    result = best_wind_opportunity_today(forecast, _CHUNKS, 1, "Test", 2000.0, _TODAY, now)

    assert result is not None
    assert result.best_hour.hour == 16
    assert result.tailwind_fraction == 0.0


def test_best_wind_opportunity_today_filters_to_hour_range() -> None:
    forecast = _fake_forecast(
        ["2026-09-10T05:00", "2026-09-10T12:00", "2026-09-10T22:00"],
        wind_speed_kmh=[20.0, 20.0, 20.0],
        wind_direction_deg=[180.0, 0.0, 180.0],  # 5h et 22h : dos (hors plage) ; 12h : face
    )
    now = datetime(2026, 9, 10, 0, 0)

    result = best_wind_opportunity_today(
        forecast, _CHUNKS, 1, "Test", 2000.0, _TODAY, now, min_hour=6, max_hour=21
    )

    assert result is not None
    assert result.best_hour.hour == 12
    assert result.tailwind_fraction == 0.0


def test_best_wind_opportunity_today_returns_none_when_no_eligible_hour() -> None:
    forecast = _fake_forecast(
        ["2026-09-10T23:00"], wind_speed_kmh=[20.0], wind_direction_deg=[180.0]
    )
    now = datetime(2026, 9, 10, 0, 0)

    result = best_wind_opportunity_today(
        forecast, _CHUNKS, 1, "Test", 2000.0, _TODAY, now, min_hour=6, max_hour=21
    )

    assert result is None
