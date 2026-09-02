"""Backtest de la calibration CdA/Crr par split temporel (T-18).

Split chronologique, pas aléatoire : la question qui compte en usage
réel est "avec ce que je savais alors (efforts passés), est-ce que je
prédis bien un effort futur ?" — un split aléatoire mélangerait passé et
futur dans le train et donnerait une erreur artificiellement optimiste,
puisque le modèle aurait alors "vu" des efforts postérieurs à ceux qu'il
prédit.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from segment_predictor.calibrate.cda_crr import (
    CdaCrrFit,
    fit_cda_crr,
    load_filtered_solo_efforts,
)
from segment_predictor.calibrate.draft_tagging import fit_current_cp
from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time


@dataclass(frozen=True)
class BacktestResult:
    """`predictions` : paires (actual_time_s, predicted_time_s) sur le set
    de test, dans l'ordre chronologique — de quoi tracer prédit vs réel
    (T-18, critère de fin du ticket)."""

    cda_crr_fit: CdaCrrFit
    n_train: int
    n_test: int
    median_absolute_error_s: float
    predictions: list[tuple[float, float]]


def backtest_cda_crr(
    efforts: list[tuple[Any, float, float, float]],
    cp_fit: CriticalPowerFit,
    mass_kg: float,
    test_fraction: float = 0.2,
) -> BacktestResult:
    """`efforts` : (start_date, distance_m, average_grade, actual_time_s),
    dans n'importe quel ordre — triés ici par date. Calibre CdA/Crr (T-17)
    sur les plus anciens (`fit_cda_crr`), évalue sur les plus récents.

    `test_fraction` : proportion visée pour le set de test, arrondie à
    l'entier le plus proche avec un minimum d'1 point.
    """
    sorted_efforts = sorted(efforts, key=lambda effort: effort[0])
    n_test = max(1, round(len(sorted_efforts) * test_fraction))
    n_train = len(sorted_efforts) - n_test
    if n_train < 2:
        raise ValueError(
            f"il faut au moins 2 efforts dans le train après split (n_train={n_train}, "
            f"total={len(sorted_efforts)}, test_fraction={test_fraction}) — plus de données "
            "ou un test_fraction plus petit"
        )

    train, test = sorted_efforts[:n_train], sorted_efforts[n_train:]

    train_for_fit = [
        (distance_m, average_grade, actual_time_s)
        for _, distance_m, average_grade, actual_time_s in train
    ]
    cda_crr_fit = fit_cda_crr(train_for_fit, cp_fit, mass_kg)

    predictions = []
    for _, distance_m, average_grade, actual_time_s in test:
        predicted_time_s = simulate_segment_time(
            [SegmentChunk(0.0, distance_m, average_grade, 0.0)],
            cp_fit.cp_watts,
            cp_fit.w_prime_joules,
            mass_kg,
            cda_crr_fit.cda_m2,
            cda_crr_fit.crr,
        )
        predictions.append((actual_time_s, predicted_time_s))

    absolute_errors_s = [abs(predicted - actual) for actual, predicted in predictions]
    median_absolute_error_s = float(np.median(absolute_errors_s))

    return BacktestResult(
        cda_crr_fit=cda_crr_fit,
        n_train=n_train,
        n_test=len(test),
        median_absolute_error_s=median_absolute_error_s,
        predictions=predictions,
    )


def backtest_cda_crr_from_db(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    mass_kg: float,
    cp_fit: CriticalPowerFit | None = None,
    test_fraction: float = 0.2,
) -> BacktestResult:
    """Backtest en conditions réelles : mêmes efforts, mêmes filtres
    (durée de validité CP, `pr_rank` non nul) que `calibrate_cda_crr_from_db`
    (T-17) — voir son docstring pour le détail. Le backtest doit s'évaluer
    sur exactement les efforts que la calibration officielle utiliserait,
    sinon le chiffre d'erreur ne dit rien sur cette calibration-là.

    `cp_fit` est injectable pour les tests ; laissé à None en usage normal
    pour être recalculé depuis `conn` (voir fit_current_cp, T-16).
    """
    if cp_fit is None:
        cp_fit = fit_current_cp(conn)

    efforts, _, _ = load_filtered_solo_efforts(conn, csv_path, cp_fit)
    return backtest_cda_crr(efforts, cp_fit, mass_kg, test_fraction=test_fraction)
