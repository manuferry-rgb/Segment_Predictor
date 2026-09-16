"""Tests du classement des créneaux horaires sur prévision (T-27)."""

import math

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
_CHUNKS = [SegmentChunk(0.0, 3500.0, 0.0, heading_rad=0.0)]  # plein nord
# 3500m plutôt que 2000m (valeur d'origine) : le temps prédit sans vent
# (~166s) tombait déjà sous le minimum calibré de _CP_FIT (180s, T-48a),
# ce qui aurait fait exclure les fenêtres de TOUS les tests existants dès
# l'ajout du garde-fou plage-calibrée — 3500m laisse assez de marge pour
# que les scénarios de vent existants (jusqu'à 20 km/h) restent dans
# [180, 1200]s sans changer ce que chaque test vérifie.


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
        forecast, _CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR, min_hour=6, max_hour=21
    )

    assert len(windows) == 1
    assert windows[0].time.hour == 10


def test_rank_forecast_windows_sorts_fastest_first() -> None:
    # cap plein nord : vent qui vient du nord (0°) = face ; du sud (180°) = dos.
    forecast = _fake_forecast(
        ["10:00", "11:00"], wind_speed_kmh=[20.0, 20.0], wind_direction_deg=[0.0, 180.0]
    )

    windows = rank_forecast_windows(forecast, _CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windows[0].time.hour == 11  # vent de dos : le plus rapide, en premier
    assert windows[1].time.hour == 10
    assert windows[0].predicted_time_s < windows[1].predicted_time_s


def test_rank_forecast_windows_records_wind_and_temperature() -> None:
    forecast = _fake_forecast(["10:00"], wind_speed_kmh=[18.0], wind_direction_deg=[90.0])

    windows = rank_forecast_windows(forecast, _CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windows[0].wind_speed_ms == pytest.approx(18.0 / 3.6)
    assert windows[0].wind_direction_rad == pytest.approx(math.pi / 2)
    assert windows[0].temperature_k == pytest.approx(15.0 + 273.15)


def test_rank_forecast_windows_uses_every_chunk_not_just_the_first() -> None:
    """Régression T-32 : `chunks` (pluriel, depuis segment_chunks_from_
    polyline) doit être transmis EN ENTIER à simulate_segment_time, pas
    seulement son premier élément — sinon le passage d'un `chunk` unique
    à une liste ne changerait rien en pratique.

    Deux tronçons de longueur égale, plein nord puis plein sud (un
    aller-retour), sous un vent qui vient du nord : face sur le premier
    tronçon, de dos sur le second. Face + dos ne s'annulent PAS
    symétriquement (terme aéro en v³, déjà vérifié isolément dans
    test_segment.py) : le temps sous ce vent doit être plus long que sans
    vent sur les deux mêmes tronçons. Si seul chunks[0] était utilisé, ce
    test échouerait aussi (un seul tronçon plein nord sous un vent de
    face reste plus lent que sans vent — mais un bug qui ignorerait
    chunks[1] resterait indétecté avec une seule assertion de ce type ;
    la valeur du test est de forcer à passer une VRAIE liste plutôt qu'un
    unique SegmentChunk, ce que l'ancienne signature ne permettait pas)."""
    two_leg_chunks = [
        SegmentChunk(0.0, 1750.0, 0.0, heading_rad=0.0),  # plein nord (aller)
        SegmentChunk(1750.0, 1750.0, 0.0, heading_rad=math.pi),  # plein sud (retour)
    ]
    # 1750m/jambe (3500m total) plutôt que 1000m d'origine, même raison
    # que _CHUNKS ci-dessus : rester dans [180, 1200]s une fois le
    # garde-fou T-48a en place.
    windy_forecast = _fake_forecast(["10:00"], wind_speed_kmh=[36.0], wind_direction_deg=[0.0])
    calm_forecast = _fake_forecast(["10:00"], wind_speed_kmh=[0.0], wind_direction_deg=[0.0])

    windy = rank_forecast_windows(windy_forecast, two_leg_chunks, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)
    calm = rank_forecast_windows(calm_forecast, two_leg_chunks, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windy[0].predicted_time_s > calm[0].predicted_time_s


def test_rank_forecast_windows_records_required_power() -> None:
    """La puissance requise (T-31) doit correspondre au TEMPS PRÉDIT de CE
    créneau (vent inclus) — pas à un CP+W'/t calculé sans vent, qui
    donnerait un nombre cohérent avec aucun des deux temps affichés."""
    forecast = _fake_forecast(["10:00"], wind_speed_kmh=[0.0], wind_direction_deg=[0.0])

    windows = rank_forecast_windows(forecast, _CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    expected_power_w = _CP_FIT.cp_watts + _CP_FIT.w_prime_joules / windows[0].predicted_time_s
    assert windows[0].required_power_w == pytest.approx(expected_power_w)


def test_rank_forecast_windows_returns_empty_when_all_slots_outside_range() -> None:
    forecast = _fake_forecast(
        ["02:00", "23:00"], wind_speed_kmh=[0.0, 0.0], wind_direction_deg=[0.0, 0.0]
    )

    windows = rank_forecast_windows(forecast, _CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windows == []


def test_rank_forecast_windows_excludes_slots_predicted_shorter_than_cp_range() -> None:
    """T-48a : un créneau dont le temps prédit tombe SOUS le minimum
    calibré de cp_fit.duration_range_s (180s par défaut) est écarté du
    classement plutôt que de laisser sustainable_power_w (models/power.py)
    extrapoler silencieusement CP+W'/t à une durée où le modèle n'a jamais
    été calibré — trouvé en prod (T-47) : une "meilleure fenêtre" plus
    rapide que le KOM du segment lui-même, avec une puissance requise
    (709W) physiologiquement intenable, sur un segment prédit à 66s."""
    short_chunks = [SegmentChunk(0.0, 800.0, 0.0, heading_rad=0.0)]  # ~55s sans vent
    forecast = _fake_forecast(["10:00"], wind_speed_kmh=[0.0], wind_direction_deg=[0.0])

    windows = rank_forecast_windows(forecast, short_chunks, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windows == []


def test_rank_forecast_windows_excludes_slots_predicted_longer_than_cp_range() -> None:
    """Symétrique du test précédent, côté durée trop LONGUE (>1200s par
    défaut) — CP+W'/t est tout aussi peu fiable en extrapolation au-delà
    de sa plage calibrée que en-deçà, pas seulement pour les durées
    courtes."""
    long_chunks = [SegmentChunk(0.0, 15_000.0, 0.0, heading_rad=0.0)]  # ~1400s sans vent
    forecast = _fake_forecast(["10:00"], wind_speed_kmh=[0.0], wind_direction_deg=[0.0])

    windows = rank_forecast_windows(forecast, long_chunks, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert windows == []


def test_rank_forecast_windows_keeps_in_range_slots_alongside_out_of_range_ones() -> None:
    """Le garde-fou T-48a écarte créneau par créneau, pas tout le
    classement dès qu'UN créneau est hors plage — même logique que le
    `except ValueError: continue` déjà présent pour une vitesse insoluble.
    Vent calme (306.6s, dans la plage) à 10h vs. vent de dos fort à 11h
    (175.0s avec _CHUNKS, sous les 180s calibrés) : seul 10h doit rester."""
    forecast = _fake_forecast(
        ["10:00", "11:00"], wind_speed_kmh=[0.0, 40.0], wind_direction_deg=[0.0, 180.0]
    )

    windows = rank_forecast_windows(forecast, _CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert len(windows) == 1
    assert windows[0].time.hour == 10


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

    windows = rank_forecast_windows(forecast, _CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert len(windows) == 1
    assert windows[0].time.hour == 11
