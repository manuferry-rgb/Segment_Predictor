"""Calibration de CdA et Crr par minimisation de l'écart temps prédit/réel (T-17).

Uniquement sur les efforts tagués `solo` (T-16) : un effort en groupe
fausserait CdA (traînée réduite par le drafting, pas par un CdA
réellement plus faible) — c'est tout l'intérêt du tri fait en T-16.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
from scipy.optimize import least_squares

from segment_predictor.calibrate.draft_tagging import fit_current_cp, load_existing_annotations
from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time

DEFAULT_CDA_BOUNDS_M2 = (0.15, 0.50)
DEFAULT_CRR_BOUNDS = (0.002, 0.010)
DEFAULT_INITIAL_CDA_M2 = 0.32
DEFAULT_INITIAL_CRR = 0.005
# Pénalité (erreur relative) utilisée quand simulate_segment_time ne converge
# pas pour un couple (CdA, Crr) testé par l'optimiseur — un coin de
# l'espace de recherche physiquement incohérent (déjà vu en T-13), pas une
# raison de faire planter toute la calibration pour un point d'essai que
# l'optimiseur écartera de toute façon.
_NON_CONVERGENCE_PENALTY = 10.0


@dataclass(frozen=True)
class CdaCrrFit:
    """Résultat d'une calibration. `rmse_relative` : racine de l'erreur
    quadratique moyenne RELATIVE (0.05 = 5% d'écart type entre prédit et
    réel) — pas en secondes, pour rester comparable entre segments de
    longueurs différentes.
    """

    cda_m2: float
    crr: float
    n_points: int
    rmse_relative: float
    converged: bool


def fit_cda_crr(
    efforts: list[tuple[float, float, float]],
    cp_fit: CriticalPowerFit,
    mass_kg: float,
    cda_bounds_m2: tuple[float, float] = DEFAULT_CDA_BOUNDS_M2,
    crr_bounds: tuple[float, float] = DEFAULT_CRR_BOUNDS,
) -> CdaCrrFit:
    """Trouve (CdA, Crr) qui minimisent l'écart relatif entre le temps
    simulé (T-13) et le temps réel, sur `efforts` (déjà filtrés aux
    efforts solo par l'appelant — cette fonction ne le vérifie pas).

    `efforts` : liste de (distance_m, average_grade, actual_time_s).
    2 paramètres à ajuster : il faut au moins 2 points, mais avec peu de
    points sur des pentes/vitesses proches, l'aéro (∝v²) et le roulement
    (constant) restent mal séparés — une variété de pentes est ce qui
    identifie vraiment les deux, pas juste le nombre de points.
    """
    if len(efforts) < 2:
        raise ValueError(
            f"il faut au moins 2 efforts solo pour ajuster 2 paramètres, reçu {len(efforts)}"
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        cda_m2, crr = params
        errors = []
        for distance_m, average_grade, actual_time_s in efforts:
            try:
                predicted_time_s = simulate_segment_time(
                    [SegmentChunk(0.0, distance_m, average_grade, 0.0)],
                    cp_fit.cp_watts,
                    cp_fit.w_prime_joules,
                    mass_kg,
                    cda_m2,
                    crr,
                )
                errors.append((predicted_time_s - actual_time_s) / actual_time_s)
            except ValueError:
                errors.append(_NON_CONVERGENCE_PENALTY)
        return np.array(errors)

    result = least_squares(
        residuals,
        x0=[DEFAULT_INITIAL_CDA_M2, DEFAULT_INITIAL_CRR],
        bounds=([cda_bounds_m2[0], crr_bounds[0]], [cda_bounds_m2[1], crr_bounds[1]]),
    )

    rmse_relative = float(np.sqrt(np.mean(result.fun**2)))
    return CdaCrrFit(
        cda_m2=float(result.x[0]),
        crr=float(result.x[1]),
        n_points=len(efforts),
        rmse_relative=rmse_relative,
        converged=bool(result.success),
    )


def calibrate_cda_crr_from_db(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    mass_kg: float,
    cp_fit: CriticalPowerFit | None = None,
) -> CdaCrrFit:
    """Charge les efforts tagués `solo` dans `csv_path`, récupère leur
    distance/pente via `main.segments`, et calibre CdA/Crr dessus.

    `cp_fit` est injectable pour les tests ; laissé à None en usage
    normal pour être recalculé depuis `conn` (voir fit_current_cp, T-16).
    """
    if cp_fit is None:
        cp_fit = fit_current_cp(conn)

    annotations = load_existing_annotations(csv_path)
    solo_effort_ids = [effort_id for effort_id, status in annotations.items() if status == "solo"]
    if not solo_effort_ids:
        raise ValueError(
            f"aucun effort tagué 'solo' dans {csv_path} — rien à calibrer. "
            "Tague au moins 2 lignes du CSV avant de relancer (voir README)."
        )

    placeholders = ",".join("?" * len(solo_effort_ids))
    rows = conn.execute(
        f"SELECT s.distance_m, s.average_grade, se.elapsed_time_s "
        f"FROM segment_efforts se JOIN segments s ON s.id = se.segment_id "
        f"WHERE se.id IN ({placeholders})",
        solo_effort_ids,
    ).fetchall()

    efforts = list(rows)
    return fit_cda_crr(efforts, cp_fit, mass_kg)
