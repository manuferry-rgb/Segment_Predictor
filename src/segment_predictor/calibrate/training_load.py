"""Charge d'entraînement réelle : TSS quotidien depuis la DB, puis CTL/ATL/TSB
(modèle de Banister, T-21).
"""

from datetime import date, timedelta

import duckdb
import numpy as np

from segment_predictor.calibrate.draft_tagging import fit_current_cp
from segment_predictor.models.power import (
    CriticalPowerFit,
    normalized_power,
    resample_to_uniform_seconds,
    training_stress_score,
)
from segment_predictor.models.training_load import (
    DEFAULT_ATL_TIME_CONSTANT_DAYS,
    DEFAULT_CTL_TIME_CONSTANT_DAYS,
    TrainingLoadPoint,
    compute_training_load,
)


def compute_daily_tss(
    conn: duckdb.DuckDBPyConnection, threshold_power_w: float
) -> dict[date, float]:
    """TSS (T-21) sommé par jour civil, sur toutes les activités
    Ride/VirtualRide à capteur de puissance — plusieurs sorties le même
    jour s'additionnent (charge cumulée, pas remplacée).

    Une activité trop courte ou trop trouée pour calculer une puissance
    normalisée (voir normalized_power) est exclue plutôt que de deviner
    un TSS pour elle — même logique que compute_aggregate_mmp_curve
    (T-16) pour un stream malformé.
    """
    rows = conn.execute(
        "SELECT id, start_date, moving_time_s FROM activities "
        "WHERE type IN ('Ride', 'VirtualRide') AND device_watts = true"
    ).fetchall()

    daily_tss: dict[date, float] = {}
    for activity_id, start_date, moving_time_s in rows:
        stream_rows = conn.execute(
            "SELECT t_s, watts FROM streams WHERE activity_id = ? ORDER BY sample_index",
            [activity_id],
        ).fetchall()
        if not stream_rows:
            continue
        t_s = np.array([r[0] for r in stream_rows])
        watts = np.array([r[1] if r[1] is not None else np.nan for r in stream_rows])
        try:
            uniform = resample_to_uniform_seconds(t_s, watts)
            np_w = normalized_power(uniform)
        except ValueError:
            continue  # trop court/troué pour une NP : on saute cette activité

        tss = training_stress_score(moving_time_s, np_w, threshold_power_w)
        day = start_date.date()
        daily_tss[day] = daily_tss.get(day, 0.0) + tss

    return daily_tss


def fill_daily_tss_series(daily_tss: dict[date, float], start: date, end: date) -> list[float]:
    """{jour: TSS} (potentiellement troué) -> série dense jour par jour de
    `start` à `end` inclus, 0.0 les jours absents — compute_training_load
    (T-21) a besoin d'une séquence sans trou pour avancer d'un jour exact
    à chaque pas de sa récursion.
    """
    if end < start:
        raise ValueError(f"end ({end}) doit être postérieur ou égal à start ({start})")
    n_days = (end - start).days + 1
    return [daily_tss.get(start + timedelta(days=i), 0.0) for i in range(n_days)]


def compute_training_load_from_db(
    conn: duckdb.DuckDBPyConnection,
    cp_fit: CriticalPowerFit | None = None,
    ctl_time_constant_days: float = DEFAULT_CTL_TIME_CONSTANT_DAYS,
    atl_time_constant_days: float = DEFAULT_ATL_TIME_CONSTANT_DAYS,
) -> tuple[list[date], list[TrainingLoadPoint]]:
    """TSS quotidien (compute_daily_tss) depuis la première à la dernière
    activité connue, converti en CTL/ATL/TSB (compute_training_load, T-21).

    `cp_fit` est injectable pour les tests ; laissé à None en usage normal
    pour être recalculé depuis `conn` (voir fit_current_cp, T-16). Le CP
    sert de proxy de FTP pour le TSS — voir training_stress_score.
    """
    if cp_fit is None:
        cp_fit = fit_current_cp(conn)

    daily_tss = compute_daily_tss(conn, threshold_power_w=cp_fit.cp_watts)
    if not daily_tss:
        raise ValueError("aucune activité à puissance exploitable : rien à calculer")

    start, end = min(daily_tss), max(daily_tss)
    series = fill_daily_tss_series(daily_tss, start, end)
    points = compute_training_load(series, ctl_time_constant_days, atl_time_constant_days)
    dates = [start + timedelta(days=i) for i in range(len(series))]
    return dates, points
