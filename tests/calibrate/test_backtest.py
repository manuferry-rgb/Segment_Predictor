"""Tests du backtest CdA/Crr par split temporel (T-18).

Split chronologique délibérément, pas aléatoire : la question posée est
"avec ce que je savais alors, est-ce que je prédis bien un effort
futur ?" — un split aléatoire mélangerait passé et futur et donnerait
une erreur artificiellement optimiste.
"""

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from segment_predictor.calibrate.backtest import backtest_cda_crr, backtest_cda_crr_from_db
from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time

_CP_FIT = CriticalPowerFit(
    cp_watts=280.0, w_prime_joules=18_000.0, r_squared=0.9, n_points=5, duration_range_s=(180, 1200)
)
_MASS_KG = 80.0
_TRUE_CDA_M2 = 0.30
_TRUE_CRR = 0.004
# 3 pentes différentes, répétées, pour que train ET test restent
# identifiables (cf fit_cda_crr : il faut de la variété de pente, pas
# juste des points).
_GRADES = [0.0, 0.03, -0.01]
_DISTANCES_M = [2000.0, 1200.0, 3000.0]


def _synthetic_efforts(n: int) -> list[tuple[str, float, float, float]]:
    """n efforts synthétiques, un par jour croissant, générés par le
    modèle lui-même (pas de bruit) — le backtest doit donc les retrouver
    quasi exactement."""
    efforts = []
    for i in range(n):
        grade = _GRADES[i % len(_GRADES)]
        distance_m = _DISTANCES_M[i % len(_DISTANCES_M)]
        actual_time_s = simulate_segment_time(
            [SegmentChunk(0.0, distance_m, grade, 0.0)],
            _CP_FIT.cp_watts,
            _CP_FIT.w_prime_joules,
            _MASS_KG,
            _TRUE_CDA_M2,
            _TRUE_CRR,
        )
        efforts.append((f"2025-01-{i + 1:02d}", distance_m, grade, actual_time_s))
    return efforts


def test_backtest_cda_crr_recovers_low_error_on_noiseless_data() -> None:
    efforts = _synthetic_efforts(10)

    result = backtest_cda_crr(efforts, _CP_FIT, _MASS_KG, test_fraction=0.3)

    assert result.n_train == 7
    assert result.n_test == 3
    assert result.median_absolute_error_s == pytest.approx(0.0, abs=1e-3)
    assert len(result.predictions) == 3


def test_backtest_cda_crr_splits_chronologically_not_randomly() -> None:
    """Mélanger l'ordre des efforts en entrée ne doit rien changer : le
    tri se fait sur la date, pas sur l'ordre de la liste."""
    efforts = _synthetic_efforts(10)
    shuffled = [efforts[i] for i in [3, 0, 9, 5, 1, 8, 2, 7, 4, 6]]

    result_sorted = backtest_cda_crr(efforts, _CP_FIT, _MASS_KG, test_fraction=0.3)
    result_shuffled = backtest_cda_crr(shuffled, _CP_FIT, _MASS_KG, test_fraction=0.3)

    assert result_shuffled.cda_crr_fit.cda_m2 == pytest.approx(result_sorted.cda_crr_fit.cda_m2)
    assert result_shuffled.predictions == result_sorted.predictions


def test_backtest_cda_crr_raises_when_train_set_too_small() -> None:
    efforts = _synthetic_efforts(2)  # test_fraction=0.5 -> 1 train, 1 test : pas assez pour fit
    with pytest.raises(ValueError, match="2 efforts"):
        backtest_cda_crr(efforts, _CP_FIT, _MASS_KG, test_fraction=0.5)


def test_backtest_cda_crr_median_error_reflects_outlier_test_point() -> None:
    """Un seul effort de test très mal prédit doit se voir dans l'erreur
    médiane si le set de test est petit (ici 3 points, la médiane retombe
    sur l'un des trois) — vérifie qu'on calcule bien une médiane sur les
    écarts ABSOLUS du set de TEST, pas sur le train."""
    efforts = _synthetic_efforts(10)
    # Le dernier effort (dans le set de test, le plus récent) devient un
    # gros outlier : un effort drafté qui aurait échappé au tri T-16, par ex.
    date, distance_m, grade, actual_time_s = efforts[-1]
    efforts[-1] = (date, distance_m, grade, actual_time_s * 0.5)

    result = backtest_cda_crr(efforts, _CP_FIT, _MASS_KG, test_fraction=0.3)

    errors_s = [abs(p - a) for a, p in result.predictions]
    assert result.median_absolute_error_s == pytest.approx(float(np.median(errors_s)))
    assert max(errors_s) > result.median_absolute_error_s  # l'outlier tire le max, pas la médiane


# ---- backtest_cda_crr_from_db --------------------------------------------------------------


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


def test_backtest_cda_crr_from_db_applies_same_filters_as_calibration(tmp_path) -> None:
    """Ne re-teste pas les filtres durée/pr_rank en détail (déjà couverts
    par test_cda_crr.py sur load_filtered_solo_efforts, partagé) — vérifie
    juste que backtest_cda_crr_from_db les applique bien avant de passer
    la main à backtest_cda_crr."""
    segments = [
        {"id": 100, "name": "Plat", "distance_m": 2500.0, "average_grade": 0.0},
        {"id": 200, "name": "Montee", "distance_m": 1000.0, "average_grade": 0.08},
    ]
    t1 = simulate_segment_time(
        [SegmentChunk(0.0, 2500.0, 0.0, 0.0)],
        _CP_FIT.cp_watts,
        _CP_FIT.w_prime_joules,
        _MASS_KG,
        _TRUE_CDA_M2,
        _TRUE_CRR,
    )
    t2 = simulate_segment_time(
        [SegmentChunk(0.0, 1000.0, 0.08, 0.0)],
        _CP_FIT.cp_watts,
        _CP_FIT.w_prime_joules,
        _MASS_KG,
        _TRUE_CDA_M2,
        _TRUE_CRR,
    )

    efforts = []
    for i in range(8):  # 8 efforts valides : de quoi splitter proprement
        segment_id, time_s = (100, t1) if i % 2 == 0 else (200, t2)
        efforts.append(
            {
                "id": i + 1,
                "segment_id": segment_id,
                "activity_id": 1,
                "start_date": f"2024-01-{i + 1:02d}T10:00:00",
                "elapsed_time_s": time_s,
                "pr_rank": 1,
            }
        )
    # doit être exclu : pas taggé solo
    efforts.append(
        {
            "id": 9,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-09T10:00:00",
            "elapsed_time_s": t1,
            "pr_rank": 1,
        }
    )
    # doit être exclu : pr_rank NULL
    efforts.append(
        {
            "id": 10,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-10T10:00:00",
            "elapsed_time_s": t1,
            "pr_rank": None,
        }
    )
    # doit être exclu : hors plage de validité CP (< 180s)
    efforts.append(
        {
            "id": 11,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-11T10:00:00",
            "elapsed_time_s": 40.0,
            "pr_rank": 1,
        }
    )
    activities = [{"id": 1, "type": "Ride", "device_watts": True, "name": "Sortie"}]
    conn = _make_db(activities, segments, efforts)

    csv_path = tmp_path / "draft_status.csv"
    csv_path.write_text(
        "effort_id,draft_status\n"
        + "\n".join(f"{i + 1},solo" for i in range(8))
        + "\n9,drafted\n10,solo\n11,solo\n"
    )

    result = backtest_cda_crr_from_db(conn, csv_path, _MASS_KG, cp_fit=_CP_FIT, test_fraction=0.25)

    assert result.n_train + result.n_test == 8  # pas 11 : les 3 exclus n'entrent pas dans le split
    assert result.n_test == 2
    assert result.n_train == 6
