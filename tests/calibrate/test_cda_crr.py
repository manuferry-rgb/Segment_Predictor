"""Tests de la calibration CdA/Crr par minimisation d'écart (T-17).

Sur des efforts synthétiques générés depuis un CdA/Crr connu (via
simulate_segment_time, le même modèle que la calibration elle-même) :
le fit doit les retrouver. Nécessite au moins 2 pentes différentes pour
bien séparer l'aéro (∝v²) du roulement (constant) — un point que nos
vraies données actuelles (1 seul segment quasi plat) ne respectent pas,
documenté dans calibrate_cda_crr_from_db plutôt que caché.
"""

import duckdb
import pyarrow as pa
import pytest

from segment_predictor.calibrate.cda_crr import (
    DEFAULT_CDA_BOUNDS_M2,
    DEFAULT_CRR_BOUNDS,
    calibrate_cda_crr_from_db,
    fit_cda_crr,
)
from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time

_CP_FIT = CriticalPowerFit(
    cp_watts=280.0, w_prime_joules=18_000.0, r_squared=0.9, n_points=5, duration_range_s=(180, 1200)
)
_MASS_KG = 80.0


def _synthetic_effort(distance_m: float, average_grade: float, cda_m2: float, crr: float) -> tuple:
    """(distance_m, average_grade, actual_time_s) généré par le modèle lui-même."""
    actual_time_s = simulate_segment_time(
        [SegmentChunk(0.0, distance_m, average_grade, 0.0)],
        _CP_FIT.cp_watts,
        _CP_FIT.w_prime_joules,
        _MASS_KG,
        cda_m2,
        crr,
    )
    return (distance_m, average_grade, actual_time_s)


# ---- fit_cda_crr --------------------------------------------------------------------------


def test_fit_cda_crr_recovers_exact_values_from_noiseless_varied_grades() -> None:
    true_cda_m2, true_crr = 0.30, 0.0045
    efforts = [
        _synthetic_effort(2000.0, 0.0, true_cda_m2, true_crr),  # plat
        _synthetic_effort(1000.0, 0.08, true_cda_m2, true_crr),  # montée
        _synthetic_effort(5000.0, -0.02, true_cda_m2, true_crr),  # légère descente
    ]

    fit = fit_cda_crr(efforts, _CP_FIT, _MASS_KG)

    assert fit.cda_m2 == pytest.approx(true_cda_m2, abs=1e-4)
    assert fit.crr == pytest.approx(true_crr, abs=1e-4)
    assert fit.n_points == 3
    assert fit.converged is True
    assert fit.rmse_relative == pytest.approx(0.0, abs=1e-6)


def test_fit_cda_crr_raises_with_fewer_than_two_efforts() -> None:
    efforts = [_synthetic_effort(2000.0, 0.0, 0.30, 0.005)]
    with pytest.raises(ValueError, match="2 efforts"):
        fit_cda_crr(efforts, _CP_FIT, _MASS_KG)


def test_fit_cda_crr_stays_within_bounds() -> None:
    """Même avec des données bruitées/mal posées, le résultat ne sort jamais
    des bornes physiques."""
    efforts = [
        (2000.0, 0.0, 100.0),  # temps irréaliste (beaucoup trop rapide)
        (1000.0, 0.08, 900.0),  # temps irréaliste (beaucoup trop lent)
    ]

    fit = fit_cda_crr(efforts, _CP_FIT, _MASS_KG)

    assert DEFAULT_CDA_BOUNDS_M2[0] <= fit.cda_m2 <= DEFAULT_CDA_BOUNDS_M2[1]
    assert DEFAULT_CRR_BOUNDS[0] <= fit.crr <= DEFAULT_CRR_BOUNDS[1]


# ---- calibrate_cda_crr_from_db -------------------------------------------------------------


def _make_db(activities, segments, efforts):
    conn = duckdb.connect(":memory:")
    for name, rows in (
        ("activities", activities),
        ("segments", segments),
        ("segment_efforts", efforts),
    ):
        conn.register("_rows", pa.Table.from_pylist(rows))
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM _rows")
        conn.unregister("_rows")
    return conn


def test_calibrate_cda_crr_from_db_uses_only_solo_tagged_efforts(tmp_path) -> None:
    true_cda_m2, true_crr = 0.28, 0.004
    segments = [
        # distances choisies pour que les temps simulés restent dans la
        # plage de validité du modèle CP (180-1200s, cf _CP_FIT) — sinon
        # exclus par le filtre testé séparément plus bas
        {"id": 100, "name": "Plat", "distance_m": 2500.0, "average_grade": 0.0},
        {"id": 200, "name": "Montee", "distance_m": 1000.0, "average_grade": 0.08},
    ]
    _, _, t1 = _synthetic_effort(2500.0, 0.0, true_cda_m2, true_crr)
    _, _, t2 = _synthetic_effort(1000.0, 0.08, true_cda_m2, true_crr)
    efforts = [
        {
            "id": 1,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-01T10:00:00",
            "elapsed_time_s": t1,
            "pr_rank": 1,
        },
        {
            "id": 2,
            "segment_id": 200,
            "activity_id": 1,
            "start_date": "2024-01-02T10:00:00",
            "elapsed_time_s": t2,
            "pr_rank": 1,
        },
        # effort en groupe : ne doit PAS influencer le fit malgré un temps très différent
        {
            "id": 3,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-03T10:00:00",
            "elapsed_time_s": 200.0,
            "pr_rank": 1,
        },
    ]
    activities = [{"id": 1, "type": "Ride", "device_watts": True, "name": "Sortie"}]
    conn = _make_db(activities, segments, efforts)

    csv_path = tmp_path / "draft_status.csv"
    csv_path.write_text("effort_id,draft_status\n1,solo\n2,solo\n3,drafted\n")

    fit = calibrate_cda_crr_from_db(conn, csv_path, _MASS_KG, cp_fit=_CP_FIT)

    assert fit.cda_m2 == pytest.approx(true_cda_m2, abs=1e-3)
    assert fit.crr == pytest.approx(true_crr, abs=1e-3)
    assert fit.n_points == 2  # pas 3 : l'effort drafted est exclu


def test_calibrate_cda_crr_from_db_excludes_efforts_outside_cp_validity_range(tmp_path) -> None:
    """Trouvé en conditions réelles (T-17, 320 efforts solo) : les segments
    de moins de 180s (hors de cp_fit.duration_range_s) subissent un biais
    énorme, le modèle CP+W'/T divergeant vers l'infini quand T->0 — les
    inclure faisait plafonner CdA/Crr à leurs bornes maximales plutôt que
    de converger vers une valeur plausible. On les exclut plutôt que de
    laisser un biais de durée se faire passer pour un mauvais CdA/Crr."""
    true_cda_m2, true_crr = 0.28, 0.004
    segments = [
        {"id": 100, "name": "Plat", "distance_m": 2500.0, "average_grade": 0.0},
        {"id": 200, "name": "Montee", "distance_m": 1000.0, "average_grade": 0.08},
        {"id": 300, "name": "Sprint court", "distance_m": 300.0, "average_grade": 0.02},
    ]
    _, _, t1 = _synthetic_effort(2500.0, 0.0, true_cda_m2, true_crr)
    _, _, t2 = _synthetic_effort(1000.0, 0.08, true_cda_m2, true_crr)
    efforts = [
        {
            "id": 1,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-01T10:00:00",
            "elapsed_time_s": t1,
            "pr_rank": 1,
        },
        {
            "id": 2,
            "segment_id": 200,
            "activity_id": 1,
            "start_date": "2024-01-02T10:00:00",
            "elapsed_time_s": t2,
            "pr_rank": 1,
        },
        # sprint court, taggé solo, mais < 180s : doit être exclu du fit
        {
            "id": 3,
            "segment_id": 300,
            "activity_id": 1,
            "start_date": "2024-01-03T10:00:00",
            "elapsed_time_s": 40.0,
            "pr_rank": 1,
        },
    ]
    activities = [{"id": 1, "type": "Ride", "device_watts": True, "name": "Sortie"}]
    conn = _make_db(activities, segments, efforts)

    csv_path = tmp_path / "draft_status.csv"
    csv_path.write_text("effort_id,draft_status\n1,solo\n2,solo\n3,solo\n")

    fit = calibrate_cda_crr_from_db(conn, csv_path, _MASS_KG, cp_fit=_CP_FIT)

    assert fit.n_points == 2  # pas 3 : le sprint court (40s) est exclu
    assert fit.cda_m2 == pytest.approx(true_cda_m2, abs=1e-3)
    assert fit.crr == pytest.approx(true_crr, abs=1e-3)


def test_calibrate_cda_crr_from_db_excludes_efforts_without_a_pr_rank(tmp_path) -> None:
    """Trouvé en conditions réelles (T-17, 320 efforts solo, filtre durée
    déjà appliqué) : un effort solo "normal" (rythme d'entraînement, pas un
    effort quasi maximal) reste comparé à un modèle qui prédit le temps
    ATTEIGNABLE en effort maximal — un biais que CdA/Crr ne peuvent pas
    corriger non plus. pr_rank (position au classement perso Strava sur ce
    segment) est un proxy simple pour "effort proche du maximum" : NULL
    veut dire que Strava n'a même pas classé cet effort parmi les records
    perso sur ce segment. Approximation documentée, pas une vraie mesure
    d'intensité (cf README)."""
    true_cda_m2, true_crr = 0.28, 0.004
    segments = [
        {"id": 100, "name": "Plat", "distance_m": 2500.0, "average_grade": 0.0},
        {"id": 200, "name": "Montee", "distance_m": 1000.0, "average_grade": 0.08},
    ]
    _, _, t1 = _synthetic_effort(2500.0, 0.0, true_cda_m2, true_crr)
    _, _, t2 = _synthetic_effort(1000.0, 0.08, true_cda_m2, true_crr)
    efforts = [
        {
            "id": 1,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-01T10:00:00",
            "elapsed_time_s": t1,
            "pr_rank": 1,
        },
        {
            "id": 2,
            "segment_id": 200,
            "activity_id": 1,
            "start_date": "2024-01-02T10:00:00",
            "elapsed_time_s": t2,
            "pr_rank": 3,
        },
        # même segment que l'effort 1, taggé solo, mais jamais classé dans
        # les records perso (rythme d'entraînement) : doit être exclu
        {
            "id": 3,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-03T10:00:00",
            "elapsed_time_s": t1 * 1.3,
            "pr_rank": None,
        },
    ]
    activities = [{"id": 1, "type": "Ride", "device_watts": True, "name": "Sortie"}]
    conn = _make_db(activities, segments, efforts)

    csv_path = tmp_path / "draft_status.csv"
    csv_path.write_text("effort_id,draft_status\n1,solo\n2,solo\n3,solo\n")

    fit = calibrate_cda_crr_from_db(conn, csv_path, _MASS_KG, cp_fit=_CP_FIT)

    assert fit.n_points == 2  # pas 3 : l'effort sans pr_rank est exclu
    assert fit.n_excluded_not_a_pr == 1
    assert fit.cda_m2 == pytest.approx(true_cda_m2, abs=1e-3)
    assert fit.crr == pytest.approx(true_crr, abs=1e-3)


def test_calibrate_cda_crr_from_db_raises_when_no_solo_efforts(tmp_path) -> None:
    segments = [{"id": 100, "name": "Plat", "distance_m": 2000.0, "average_grade": 0.0}]
    efforts = [
        {
            "id": 1,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-01T10:00:00",
            "elapsed_time_s": 300,
        }
    ]
    activities = [{"id": 1, "type": "Ride", "device_watts": True, "name": "Sortie"}]
    conn = _make_db(activities, segments, efforts)

    csv_path = tmp_path / "draft_status.csv"
    csv_path.write_text("effort_id,draft_status\n1,unknown\n")

    with pytest.raises(ValueError, match="solo"):
        calibrate_cda_crr_from_db(conn, csv_path, _MASS_KG, cp_fit=_CP_FIT)
