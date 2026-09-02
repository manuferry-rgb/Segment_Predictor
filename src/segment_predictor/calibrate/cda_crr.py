"""Calibration de CdA et Crr par minimisation de l'écart temps prédit/réel (T-17).

Uniquement sur les efforts tagués `solo` (T-16) : un effort en groupe
fausserait CdA (traînée réduite par le drafting, pas par un CdA
réellement plus faible) — c'est tout l'intérêt du tri fait en T-16.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
from scipy.optimize import least_squares

from segment_predictor.calibrate.draft_tagging import fit_current_cp, load_existing_annotations
from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time

DEFAULT_CDA_BOUNDS_M2 = (0.15, 0.50)
# Borne haute à 0.012 plutôt que la valeur "manuel" habituelle (~0.005 sur
# bitume lisse) : sur les 77 efforts réels retenus (T-17), l'optimum SANS
# contrainte converge vers Crr≈0.0105, quelle que soit la largeur des
# bornes testées (vérifié jusqu'à 0.03) — pas un artefact de plafonnement,
# une vraie valeur portée par les données (revêtement réel des segments,
# et/ou un résidu de vent non modélisé absorbé par Crr plutôt que CdA,
# le vent n'étant pas encore branché dans cette calibration, cf T-15/T-19).
DEFAULT_CRR_BOUNDS = (0.002, 0.012)
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
    longueurs différentes. `n_excluded_outside_duration_range` : efforts
    solo écartés parce que hors de la plage de validité du modèle CP
    (voir calibrate_cda_crr_from_db). `n_excluded_not_a_pr` : efforts solo
    écartés parce que jamais classés dans les records perso Strava sur
    leur segment (pr_rank NULL) — voir calibrate_cda_crr_from_db pour le
    raisonnement. Les deux valent 0 si le fit vient de fit_cda_crr
    directement, sans passer par ces filtres.
    """

    cda_m2: float
    crr: float
    n_points: int
    rmse_relative: float
    converged: bool
    n_excluded_outside_duration_range: int = 0
    n_excluded_not_a_pr: int = 0


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


def load_filtered_solo_efforts(
    conn: duckdb.DuckDBPyConnection, csv_path: Path, cp_fit: CriticalPowerFit
) -> tuple[list[tuple[Any, float, float, float]], int, int]:
    """Efforts tagués `solo` dans `csv_path`, dans la plage de validité du
    modèle CP et avec un `pr_rank` non nul — voir `calibrate_cda_crr_from_db`
    pour le raisonnement complet des deux filtres. Chaque ligne retournée :
    (start_date, distance_m, average_grade, elapsed_time_s).

    Retourne aussi les deux compteurs d'exclusion (durée, pr_rank), dans cet
    ordre. Partagé avec `calibrate/backtest.py` (T-18) pour ne pas dupliquer
    ces filtres — le backtest doit s'évaluer sur exactement les mêmes
    efforts que la calibration officielle.
    """
    annotations = load_existing_annotations(csv_path)
    solo_effort_ids = [effort_id for effort_id, status in annotations.items() if status == "solo"]
    if not solo_effort_ids:
        raise ValueError(
            f"aucun effort tagué 'solo' dans {csv_path} — rien à calibrer. "
            "Tague au moins 2 lignes du CSV avant de relancer (voir README)."
        )

    placeholders = ",".join("?" * len(solo_effort_ids))
    rows = conn.execute(
        f"SELECT se.start_date, s.distance_m, s.average_grade, se.elapsed_time_s, se.pr_rank "
        f"FROM segment_efforts se JOIN segments s ON s.id = se.segment_id "
        f"WHERE se.id IN ({placeholders})",
        solo_effort_ids,
    ).fetchall()

    range_min_s, range_max_s = cp_fit.duration_range_s
    in_range = [row for row in rows if range_min_s <= row[3] <= range_max_s]
    excluded_duration_count = len(rows) - len(in_range)

    with_pr_rank = [row for row in in_range if row[4] is not None]
    excluded_not_a_pr_count = len(in_range) - len(with_pr_rank)

    efforts = [
        (start_date, distance_m, average_grade, time_s)
        for start_date, distance_m, average_grade, time_s, _ in with_pr_rank
    ]
    return efforts, excluded_duration_count, excluded_not_a_pr_count


def calibrate_cda_crr_from_db(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    mass_kg: float,
    cp_fit: CriticalPowerFit | None = None,
) -> CdaCrrFit:
    """Charge les efforts tagués `solo` dans `csv_path`, récupère leur
    distance/pente via `main.segments`, et calibre CdA/Crr dessus.

    Exclut les efforts dont la durée tombe hors de `cp_fit.duration_range_s`
    (par défaut 180-1200s) — trouvé en conditions réelles (320 efforts
    solo) : sous 180s, `CP + W'/T` diverge vers l'infini quand T->0 (déjà
    documenté en T-09), donc surestime largement la puissance soutenable
    sur les segments courts. Les inclure ne calibre pas un mauvais CdA/Crr,
    ça fait plafonner l'optimiseur à ses bornes maximales pour compenser un
    biais de durée qu'aucun CdA/Crr ne peut corriger.

    Exclut aussi les efforts solo dont `pr_rank` est NULL — trouvé en
    conditions réelles (320 efforts solo, filtre durée déjà appliqué,
    n=131 restants) : la plupart de ces efforts sont des passages à
    rythme d'entraînement, pas des efforts proches du maximum, or le
    modèle prédit le temps ATTEIGNABLE en effort maximal (CP + W'/T). Les
    comparer biaise le fit dans la même direction que le biais de durée,
    peu importe le vrai CdA/Crr. `pr_rank` (position au classement perso
    Strava sur ce segment) est un proxy simple pour "effort quasi
    maximal" — approximation documentée, pas une vraie mesure d'intensité
    (ex. VO2/FTP) : un effort peut être un record perso sans être un
    effort "à bloc", et inversement sur un segment rarement emprunté.

    `cp_fit` est injectable pour les tests ; laissé à None en usage
    normal pour être recalculé depuis `conn` (voir fit_current_cp, T-16).
    """
    if cp_fit is None:
        cp_fit = fit_current_cp(conn)

    efforts_with_dates, excluded_duration_count, excluded_not_a_pr_count = (
        load_filtered_solo_efforts(conn, csv_path, cp_fit)
    )
    efforts = [
        (distance_m, average_grade, time_s)
        for _, distance_m, average_grade, time_s in efforts_with_dates
    ]
    fit = fit_cda_crr(efforts, cp_fit, mass_kg)
    return replace(
        fit,
        n_excluded_outside_duration_range=excluded_duration_count,
        n_excluded_not_a_pr=excluded_not_a_pr_count,
    )
