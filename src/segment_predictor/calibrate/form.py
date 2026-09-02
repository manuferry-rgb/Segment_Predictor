"""Série temporelle de l'indice de performance, calculée sur l'historique
réel des activités (T-23).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import numpy as np

from segment_predictor.calibrate.draft_tagging import DEFAULT_CP_FIT_DURATIONS_S, fit_current_cp
from segment_predictor.calibrate.training_load import compute_training_load_from_db
from segment_predictor.models.form import (
    DEFAULT_MAXIMAL_EFFORT_THRESHOLD,
    is_near_maximal_effort,
    performance_index,
)
from segment_predictor.models.power import (
    CriticalPowerFit,
    mean_maximal_power_curve,
    resample_to_uniform_seconds,
)


@dataclass(frozen=True)
class PerformanceIndexPoint:
    date: date
    duration_s: int
    actual_power_w: float
    index: float


def compute_performance_index_series(
    conn: duckdb.DuckDBPyConnection,
    cp_fit: CriticalPowerFit | None = None,
    durations_s: Iterable[int] = DEFAULT_CP_FIT_DURATIONS_S,
    threshold: float = DEFAULT_MAXIMAL_EFFORT_THRESHOLD,
) -> list[PerformanceIndexPoint]:
    """Parcourt les activités Ride/VirtualRide à capteur de puissance, par
    date croissante, et calcule l'indice de performance (T-23) à chaque
    durée de référence où l'effort est jugé "proche du maximum" (T-23,
    is_near_maximal_effort) — le record glissant est mis à jour au fil de
    l'eau, jamais avec une donnée future.

    `cp_fit` est injectable pour les tests ; laissé à None en usage
    normal pour être recalculé depuis `conn` (voir fit_current_cp, T-16).
    """
    if cp_fit is None:
        cp_fit = fit_current_cp(conn)
    durations_s = list(durations_s)

    activities = conn.execute(
        "SELECT id, start_date FROM activities "
        "WHERE type IN ('Ride', 'VirtualRide') AND device_watts = true "
        "ORDER BY start_date"
    ).fetchall()

    best_so_far_w: dict[int, float] = {}
    points: list[PerformanceIndexPoint] = []
    for activity_id, start_date in activities:
        stream_rows = conn.execute(
            "SELECT t_s, watts FROM streams WHERE activity_id = ? ORDER BY sample_index",
            [activity_id],
        ).fetchall()
        if not stream_rows:
            continue
        t_s = np.array([row[0] for row in stream_rows])
        watts = np.array([row[1] if row[1] is not None else np.nan for row in stream_rows])
        try:
            uniform = resample_to_uniform_seconds(t_s, watts)
        except ValueError:
            continue  # stream malformé/trop court : on saute cette activité (précédent T-16)

        curve = mean_maximal_power_curve(uniform, durations_s)
        for duration_s, mmp_w in curve.items():
            if np.isnan(mmp_w):
                continue

            previous_best_w = best_so_far_w.get(duration_s)
            is_maximal = previous_best_w is None or is_near_maximal_effort(
                mmp_w, previous_best_w, threshold
            )
            if is_maximal:
                try:
                    index = performance_index(mmp_w, cp_fit, duration_s)
                    points.append(
                        PerformanceIndexPoint(
                            date=start_date.date(),
                            duration_s=duration_s,
                            actual_power_w=mmp_w,
                            index=index,
                        )
                    )
                except ValueError:
                    pass  # duration_s hors de cp_fit.duration_range_s : pas d'indice défini ici

            best_so_far_w[duration_s] = max(previous_best_w or 0.0, mmp_w)

    return points


# Colonnes de build_form_regression_dataset, dans l'ordre. Volontairement
# SANS les 4 champs wellness (T-22 : hrv, sleep_s, resting_heart_rate_bpm,
# weight_kg) : en conditions réelles ils sont quasi tous vides (0 hrv,
# 0 sleep_s, resting_heart_rate_bpm constant sur 49 jours, 1 seule valeur
# weight_kg — voir le README, section T-22). Les inclure demanderait une
# imputation qui inventerait des données absentes plutôt que de refléter
# un vrai signal de forme.
FORM_REGRESSION_FEATURE_NAMES = ("ctl", "atl", "tsb", "duration_s")


def build_form_regression_dataset(
    conn: duckdb.DuckDBPyConnection,
    cp_fit: CriticalPowerFit | None = None,
    durations_s: Iterable[int] = DEFAULT_CP_FIT_DURATIONS_S,
    threshold: float = DEFAULT_MAXIMAL_EFFORT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[date]]:
    """Joint la série de l'indice de performance (T-23,
    compute_performance_index_series) au CTL/ATL/TSB de CHAQUE DATE EXACTE
    (T-21, compute_training_load_from_db) — jamais un CTL/ATL/TSB futur ou
    passé approché, l'égalité de date est stricte. `duration_s` est ajouté
    comme 4e feature : le modèle CP n'aligne pas parfaitement les courbes
    par durée (visible dans le graphique T-23), l'inclure sépare ce biais
    résiduel du modèle de l'effet "forme" qu'on veut isoler.

    Retourne (X, y, dates) prêts pour fit_ridge/temporal_cross_validate_
    ridge (T-24) — X dans l'ordre de FORM_REGRESSION_FEATURE_NAMES.
    """
    if cp_fit is None:
        cp_fit = fit_current_cp(conn)

    index_points = compute_performance_index_series(
        conn, cp_fit=cp_fit, durations_s=durations_s, threshold=threshold
    )
    if not index_points:
        raise ValueError("aucun point d'indice de performance disponible (T-23) : rien à joindre")

    load_dates, load_points = compute_training_load_from_db(conn, cp_fit=cp_fit)
    load_by_date = dict(zip(load_dates, load_points, strict=True))

    rows: list[list[float]] = []
    dates: list[date] = []
    y: list[float] = []
    for point in index_points:
        load = load_by_date.get(point.date)
        if load is None:
            # ne devrait pas arriver : compute_training_load_from_db couvre toute la plage
            # d'activités, dont celle-ci fait partie. Pas de valeur inventée si ça arrive
            # quand même — on saute juste ce point plutôt que d'approcher une date voisine.
            continue
        rows.append([load.ctl, load.atl, load.tsb, float(point.duration_s)])
        dates.append(point.date)
        y.append(point.index)

    return np.array(rows), np.array(y), dates


DEFAULT_RECENT_WINDOW_DAYS = 90


def recent_performance_index_values(
    conn: duckdb.DuckDBPyConnection,
    cp_fit: CriticalPowerFit | None = None,
    window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
    reference_date: date | None = None,
    durations_s: Iterable[int] = DEFAULT_CP_FIT_DURATIONS_S,
) -> list[float]:
    """Valeurs de l'indice de performance (T-23) des `window_days` derniers
    jours jusqu'à `reference_date` (aujourd'hui par défaut) — pas tout
    l'historique : l'indice dérive fortement dans le temps (T-23/T-24,
    conséquence du CP *fixe* utilisé comme référence), la distribution
    récente reflète la forme ACTUELLE, pas celle d'il y a plusieurs années.
    Sert de distribution empirique à échantillonner pour l'incertitude de
    forme du Monte-Carlo (T-28, `models.uncertainty.propagate_uncertainty`).

    Liste vide si rien dans la fenêtre — pas une erreur ici (c'est
    `propagate_uncertainty` qui refuse une liste vide, pas cette fonction
    de lecture).
    """
    if window_days <= 0:
        raise ValueError(f"window_days doit être positif, reçu {window_days}")
    if reference_date is None:
        reference_date = date.today()

    cutoff_date = reference_date - timedelta(days=window_days)
    points = compute_performance_index_series(conn, cp_fit=cp_fit, durations_s=durations_s)
    return [point.index for point in points if cutoff_date <= point.date <= reference_date]
