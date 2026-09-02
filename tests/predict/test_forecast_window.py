"""Tests du classement des créneaux horaires sur prévision (T-27)."""

import pytest

from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk
from segment_predictor.predict.forecast_window import rank_forecast_windows

_CP_FIT = CriticalPowerFit(
    cp_watts=250.0, w_prime_joules=20_000.0, r_squared=0.9, n_points=5, duration_range_s=(180, 1200)
)
_MASS_KG = 75.0
_CDA_M2 = 0.30
_CRR = 0.005
_CHUNK = SegmentChunk(0.0, 2000.0, 0.0, heading_rad=0.0)  # plein nord


def _fake_forecast(hours: list[str], wind_speed_kmh: list[float], wind_direction_deg: list[float]):
    n = len(hours)
    return {
        "hourly": {
            "time": [f"2026-09-10T{h}" for h in hours],
            "temperature_2m": [15.0] * n,
            "relative_humidity_2m": [60.0] * n,
            "surface_pressure": [1013.0] * n,
            "wind_speed_10m": wind_speed_kmh,
            "wind_direction_10m": wind_direction_deg,
        }
    }


def test_rank_forecast_windows_filters_outside_hour_range() -> None:
    forecast = _fake_forecast(
        ["03:00", "10:00", "23:00"],
        wind_speed_kmh=[0.0, 0.0, 0.0],
        wind_direction_deg=[0.0, 0.0, 0.0],
    )

    windows = rank_forecast_windows(
        forecast, _CHUNK, _CP_FIT, _MASS_KG, _CDA_M2, _CRR, min_hour=6, max_hour=21
    )

    assert len(windows) == 1
    assert windows[0].time.hour == 10


def test_rank_forecast_windows_sorts_fastest_first() -> None:
    # cap plein nord : vent qui vient du nord (0°) = face ; du sud (180°) = dos.
    forecast = _fake_forecast(
        ["10:00", "11:00"], wind_speed_kmh=[20.0, 20.0], wind_direction_deg=[0.0, 180.0]
    )

    windows = rank_forecast_windows(forecast, _CHUNK, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windows[0].time.hour == 11  # vent de dos : le plus rapide, en premier
    assert windows[1].time.hour == 10
    assert windows[0].predicted_time_s < windows[1].predicted_time_s


def test_rank_forecast_windows_records_wind_and_temperature() -> None:
    forecast = _fake_forecast(["10:00"], wind_speed_kmh=[18.0], wind_direction_deg=[90.0])

    windows = rank_forecast_windows(forecast, _CHUNK, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windows[0].wind_speed_ms == pytest.approx(18.0 / 3.6)
    assert windows[0].temperature_k == pytest.approx(15.0 + 273.15)


def test_rank_forecast_windows_returns_empty_when_all_slots_outside_range() -> None:
    forecast = _fake_forecast(
        ["02:00", "23:00"], wind_speed_kmh=[0.0, 0.0], wind_direction_deg=[0.0, 0.0]
    )

    windows = rank_forecast_windows(forecast, _CHUNK, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windows == []


def test_rank_forecast_windows_skips_slots_with_no_feasible_speed() -> None:
    """Un vent de face extrême ne suffit pas à rendre la vitesse insoluble
    (cyclist_speed_from_power, T-11) : à v->0, la puissance requise tend
    vers 0 quel que soit le vent, une solution existe presque toujours
    dans les bornes par défaut. Un vent de DOS extrême, en revanche, peut
    réduire la puissance requise au point que la vitesse nécessaire pour
    consommer exactement la puissance cible dépasse la borne haute
    (40 m/s) — vérifié en conditions réelles avant d'écrire ce test,
    pas supposé. Ce créneau doit être écarté, pas faire planter tout le
    classement."""
    extreme_tailwind_kmh = 216.0  # 60 m/s de dos (vient du sud, cap plein nord)
    forecast = _fake_forecast(
        ["10:00", "11:00"],
        wind_speed_kmh=[extreme_tailwind_kmh, 0.0],
        wind_direction_deg=[180.0, 0.0],
    )

    windows = rank_forecast_windows(forecast, _CHUNK, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert len(windows) == 1
    assert windows[0].time.hour == 11
