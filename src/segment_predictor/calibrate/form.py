"""Série temporelle de l'indice de performance, calculée sur l'historique
réel des activités (T-23).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import duckdb
import numpy as np

from segment_predictor.calibrate.draft_tagging import DEFAULT_CP_FIT_DURATIONS_S, fit_current_cp
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
