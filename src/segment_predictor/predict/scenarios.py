"""Comparaison d'un même segment sous plusieurs scénarios de draft (T-20).

Combine les modèles déjà construits (simulate_segment_time, T-13 ;
draft_ratio_for_preset, T-19) avec un segment réel : le temps prédit
sous chaque preset de draft, et le gain par rapport au solo. Premier
morceau de la couche `predict` (segment + météo + forme -> temps prédit,
cf CLAUDE.md) — météo et forme viendront avec les tickets suivants.
"""

from dataclasses import dataclass

import duckdb

from segment_predictor.models.draft import draft_ratio_for_preset
from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time

ALL_DRAFT_PRESETS = ("solo", "roue_collee", "un_metre", "groupe")


@dataclass(frozen=True)
class DraftScenarioResult:
    """`gain_vs_solo_s` positif = plus rapide qu'en solo (temps solo moins
    temps de ce scénario)."""

    preset: str
    predicted_time_s: float
    gain_vs_solo_s: float
    gain_vs_solo_fraction: float


def compare_draft_scenarios(
    chunks: list[SegmentChunk],
    cp_fit: CriticalPowerFit,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    presets: tuple[str, ...] = ALL_DRAFT_PRESETS,
) -> list[DraftScenarioResult]:
    """Temps prédit (T-13) sous chaque preset de `presets` (T-19) : le CdA
    solo est réduit par le ratio du preset avant simulation — le drafting
    ne change que le terme aéro, pas la masse ni le roulement.

    `presets` doit inclure "solo" : c'est la référence du gain calculé
    pour chaque scénario, sinon `ValueError` explicite plutôt qu'un gain
    calculé contre rien.
    """
    if "solo" not in presets:
        raise ValueError(
            f"'solo' doit être dans presets pour calculer un gain de référence, reçu {presets}"
        )

    times_by_preset_s = {
        preset: simulate_segment_time(
            chunks,
            cp_fit.cp_watts,
            cp_fit.w_prime_joules,
            mass_kg,
            cda_m2 * draft_ratio_for_preset(preset),
            crr,
        )
        for preset in presets
    }
    solo_time_s = times_by_preset_s["solo"]

    return [
        DraftScenarioResult(
            preset=preset,
            predicted_time_s=times_by_preset_s[preset],
            gain_vs_solo_s=solo_time_s - times_by_preset_s[preset],
            gain_vs_solo_fraction=(solo_time_s - times_by_preset_s[preset]) / solo_time_s,
        )
        for preset in presets
    ]


def compare_draft_scenarios_for_segment(
    conn: duckdb.DuckDBPyConnection,
    segment_id: int,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    cp_fit: CriticalPowerFit,
    presets: tuple[str, ...] = ALL_DRAFT_PRESETS,
) -> list[DraftScenarioResult]:
    """Charge distance/pente de `segment_id` dans `main.segments` et compare
    les scénarios de draft dessus.

    Même approximation qu'ailleurs (T-16, T-17) : le segment est traité
    comme UN SEUL tronçon à pente moyenne constante
    (`main.segments.average_grade`), pas son profil réel tronçon par
    tronçon — pas extrait au niveau segment (T-07b).
    """
    row = conn.execute(
        "SELECT distance_m, average_grade FROM segments WHERE id = ?", [segment_id]
    ).fetchone()
    if row is None:
        raise ValueError(f"segment {segment_id} introuvable dans main.segments (404)")
    distance_m, average_grade = row

    chunk = SegmentChunk(
        start_distance_m=0.0, length_m=distance_m, grade=average_grade, heading_rad=0.0
    )
    return compare_draft_scenarios([chunk], cp_fit, mass_kg, cda_m2, crr, presets=presets)
