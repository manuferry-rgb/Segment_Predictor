"""Tests de l'optimisation du profil de puissance par programmation
dynamique (T-26).

Fonctions pures, aucun I/O. Le critère de fin du ticket — un profil
optimal plus rapide qu'un profil à puissance constante — est vérifié en
comparant directement à simulate_segment_time (T-13), sur un segment
avec une vraie variation de pente (sur un segment plat, la puissance
constante est déjà quasi optimale, il n'y aurait rien à démontrer).
"""

import pytest

from segment_predictor.models.pacing import DEFAULT_N_W_BAL_LEVELS, optimize_pacing
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time
from segment_predictor.models.wbal import w_prime_balance_step

_CP_W = 250.0
_W_PRIME_J = 20_000.0
_MASS_KG = 75.0
_CDA_M2 = 0.30
_CRR = 0.005


def _speed_ms(power_w: float, grade: float) -> float:
    from segment_predictor.models.physics import cyclist_speed_from_power
    from segment_predictor.models.segment import STANDARD_AIR_DENSITY_KG_M3

    return cyclist_speed_from_power(
        power_w, grade, 0.0, _MASS_KG, _CDA_M2, _CRR, STANDARD_AIR_DENSITY_KG_M3
    )


# ---- critère de fin du ticket : plus rapide qu'un profil à puissance constante ---------------


def test_optimal_pacing_beats_constant_power_on_a_hilly_segment() -> None:
    chunks = [
        SegmentChunk(0.0, 500.0, 0.08, 0.0),  # montée à 8%
        SegmentChunk(500.0, 1500.0, -0.02, 0.0),  # légère descente
    ]

    result = optimize_pacing(chunks, _CP_W, _W_PRIME_J, _MASS_KG, _CDA_M2, _CRR)
    constant_power_time_s = simulate_segment_time(
        chunks, _CP_W, _W_PRIME_J, _MASS_KG, _CDA_M2, _CRR
    )

    assert result.total_time_s < constant_power_time_s
    assert len(result.power_profile_w) == len(chunks)


def test_optimal_pacing_pushes_harder_on_the_climb_than_the_descent() -> None:
    """Signature attendue du gain physique : plus de puissance en montée
    (où la puissance se traduit presque directement en vitesse) que sur
    le plat/la descente (où le coût aéro cubique rend chaque watt
    supplémentaire beaucoup moins rentable en vitesse)."""
    chunks = [
        SegmentChunk(0.0, 500.0, 0.08, 0.0),
        SegmentChunk(500.0, 1500.0, -0.02, 0.0),
    ]

    result = optimize_pacing(chunks, _CP_W, _W_PRIME_J, _MASS_KG, _CDA_M2, _CRR)

    climb_power_w, descent_power_w = result.power_profile_w
    assert climb_power_w > descent_power_w


# ---- contrainte W' >= 0 -----------------------------------------------------------------------


def test_optimal_pacing_never_lets_w_prime_go_negative() -> None:
    chunks = [
        SegmentChunk(0.0, 500.0, 0.08, 0.0),
        SegmentChunk(500.0, 1500.0, -0.02, 0.0),
        SegmentChunk(2000.0, 500.0, 0.05, 0.0),
    ]

    result = optimize_pacing(chunks, _CP_W, _W_PRIME_J, _MASS_KG, _CDA_M2, _CRR)

    # La DP ne raisonne qu'aux niveaux de sa grille W' (100 par défaut,
    # ~200J d'écart ici) : un profil "faisable" dans la DP peut, reconstruit
    # en continu (comme ci-dessous), dériver légèrement sous 0 — l'erreur
    # de discrétisation attendue est au plus un pas de grille, pas un bug.
    grid_step_j = _W_PRIME_J / DEFAULT_N_W_BAL_LEVELS

    w_bal = _W_PRIME_J
    for chunk, power_w in zip(chunks, result.power_profile_w, strict=True):
        speed_ms = _speed_ms(power_w, chunk.grade)
        dt_s = chunk.length_m / speed_ms
        w_bal = w_prime_balance_step(w_bal, power_w, _CP_W, _W_PRIME_J, dt_s)
        assert w_bal >= -grid_step_j


# ---- cas simple : un seul tronçon plat -------------------------------------------------------


def test_optimal_pacing_matches_constant_power_on_a_single_flat_chunk() -> None:
    """Avec un seul tronçon, il n'y a rien à arbitrer : la DP doit
    retrouver (à la discrétisation près) le même temps que
    simulate_segment_time."""
    chunks = [SegmentChunk(0.0, 2000.0, 0.0, 0.0)]

    result = optimize_pacing(chunks, _CP_W, _W_PRIME_J, _MASS_KG, _CDA_M2, _CRR)
    constant_power_time_s = simulate_segment_time(
        chunks, _CP_W, _W_PRIME_J, _MASS_KG, _CDA_M2, _CRR
    )

    assert result.total_time_s == pytest.approx(constant_power_time_s, rel=0.05)


# ---- erreurs ------------------------------------------------------------------------------


def test_optimize_pacing_raises_on_empty_chunks() -> None:
    with pytest.raises(ValueError, match="tronçon"):
        optimize_pacing([], _CP_W, _W_PRIME_J, _MASS_KG, _CDA_M2, _CRR)
