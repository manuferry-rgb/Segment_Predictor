"""Tests du découpage de segment en tronçons ~50m (T-12) et de la simulation
d'un segment complet (T-13).

Fonctions pures, aucun I/O. `smooth_altitude` est testée seule (sur un
profil synthétique bruité de vérité connue) avant `chunk_segment`.
"""

import math

import numpy as np
import pytest

from segment_predictor.models.physics import cyclist_speed_from_power
from segment_predictor.models.segment import (
    STANDARD_AIR_DENSITY_KG_M3,
    SegmentChunk,
    chunk_segment,
    simulate_segment_time,
    smooth_altitude,
)

METERS_PER_DEGREE = 111_320.0  # approximation usuelle à l'équateur

# ---- smooth_altitude ------------------------------------------------------------------


def test_smooth_altitude_leaves_a_noiseless_constant_profile_unchanged() -> None:
    distance_m = np.arange(0, 200, 5.0)
    altitude_m = np.full_like(distance_m, 150.0)

    smoothed = smooth_altitude(distance_m, altitude_m, window_m=30.0)

    np.testing.assert_allclose(smoothed, 150.0)


def test_smooth_altitude_reduces_noise_on_a_known_linear_profile() -> None:
    """Bruit GPS synthétique (+/-3m, centré) sur une pente constante connue :
    le profil lissé doit se rapprocher nettement plus du vrai profil que le
    profil brut — c'est la justification quantitative du choix de méthode."""
    rng = np.random.default_rng(seed=7)
    distance_m = np.arange(0, 500, 2.0)  # échantillonnage dense, comme un vrai stream GPS
    true_altitude_m = 100.0 + 0.04 * distance_m  # pente constante 4%
    noisy_altitude_m = true_altitude_m + rng.normal(0.0, 3.0, size=distance_m.shape)

    smoothed = smooth_altitude(distance_m, noisy_altitude_m, window_m=30.0)

    raw_mean_error_m = np.abs(noisy_altitude_m - true_altitude_m).mean()
    smoothed_mean_error_m = np.abs(smoothed - true_altitude_m).mean()
    assert smoothed_mean_error_m < raw_mean_error_m * 0.5


def test_smooth_altitude_blurs_a_sharp_step() -> None:
    """Limite assumée, pas cachée : une vraie cassure nette est adoucie par
    la fenêtre, pas préservée — le prix à payer pour annuler le bruit GPS."""
    distance_m = np.arange(0, 200, 2.0)
    altitude_m = np.where(distance_m < 100, 0.0, 20.0)  # marche nette de 20m à 100m

    smoothed = smooth_altitude(distance_m, altitude_m, window_m=30.0)

    idx_at_step = np.argmin(np.abs(distance_m - 100))
    assert 0.0 < smoothed[idx_at_step] < 20.0


def test_smooth_altitude_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="longueur"):
        smooth_altitude(np.array([0.0, 10.0, 20.0]), np.array([100.0, 101.0]))


def test_smooth_altitude_raises_on_decreasing_distance() -> None:
    with pytest.raises(ValueError, match="croissante"):
        smooth_altitude(np.array([0.0, 10.0, 5.0]), np.array([100.0, 101.0, 102.0]))


# ---- chunk_segment ----------------------------------------------------------------------


def test_chunk_segment_recovers_exact_constant_grade_without_noise() -> None:
    distance_m = np.arange(0, 201, 10.0)  # 0..200m, 200m total, multiple exact de 50
    true_grade = 0.05
    altitude_m = distance_m * true_grade
    # chemin plein nord : longitude constante, latitude qui avance avec la distance
    lat = distance_m / METERS_PER_DEGREE
    lng = np.zeros_like(distance_m)

    chunks = chunk_segment(distance_m, altitude_m, lat, lng, chunk_length_m=50.0)

    assert len(chunks) == 4
    assert [c.start_distance_m for c in chunks] == [0.0, 50.0, 100.0, 150.0]
    for chunk in chunks:
        assert chunk.length_m == pytest.approx(50.0)
        assert chunk.grade == pytest.approx(true_grade)
        assert chunk.heading_rad == pytest.approx(0.0, abs=1e-6)  # plein nord


def test_chunk_segment_handles_a_remainder_shorter_last_chunk() -> None:
    distance_m = np.arange(0, 121, 10.0)  # 120m total, pas un multiple de 50
    altitude_m = distance_m * 0.02
    lat = distance_m / METERS_PER_DEGREE
    lng = np.zeros_like(distance_m)

    chunks = chunk_segment(distance_m, altitude_m, lat, lng, chunk_length_m=50.0)

    assert len(chunks) == 3
    assert [c.length_m for c in chunks] == pytest.approx([50.0, 50.0, 20.0])
    assert sum(c.length_m for c in chunks) == pytest.approx(120.0)


def test_chunk_segment_headings_match_known_compass_directions() -> None:
    """1er tronçon plein nord, 2e plein est — vérifiable à la main via la
    définition même du cap (0 = nord, croissant vers l'est)."""
    distance_m = np.array([0.0, 50.0, 100.0])
    lat = np.array([0.0, 50.0 / METERS_PER_DEGREE, 50.0 / METERS_PER_DEGREE])
    lng = np.array([0.0, 0.0, 50.0 / METERS_PER_DEGREE])
    altitude_m = np.array([0.0, 0.0, 0.0])

    chunks = chunk_segment(distance_m, altitude_m, lat, lng, chunk_length_m=50.0)

    assert len(chunks) == 2
    assert chunks[0].heading_rad == pytest.approx(0.0, abs=1e-6)  # nord
    assert chunks[1].heading_rad == pytest.approx(math.pi / 2, abs=1e-3)  # est


def test_chunk_segment_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="longueur"):
        chunk_segment(
            np.array([0.0, 50.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.1]),
            np.array([0.0, 0.0]),
        )


def test_chunk_segment_raises_on_non_positive_chunk_length() -> None:
    distance_m = np.array([0.0, 50.0, 100.0])
    zeros = np.zeros_like(distance_m)
    with pytest.raises(ValueError, match="chunk_length_m"):
        chunk_segment(distance_m, zeros, zeros, zeros, chunk_length_m=0.0)


# ---- simulate_segment_time (T-13) --------------------------------------------------------

_CP_WATTS = 250.0
_W_PRIME_JOULES = 20_000.0
_MASS_KG = 75.0
_CDA_M2 = 0.30
_CRR = 0.005


def _cp_only_time_s(chunks: list[SegmentChunk]) -> float:
    """Temps si on roulait à CP constant (sans le supplément W'/T) — calculable
    directement, sans boucle : sert de référence indépendante pour les tests."""
    total_time_s = 0.0
    for chunk in chunks:
        speed_ms = cyclist_speed_from_power(
            _CP_WATTS, chunk.grade, 0.0, _MASS_KG, _CDA_M2, _CRR, STANDARD_AIR_DENSITY_KG_M3
        )
        total_time_s += chunk.length_m / speed_ms
    return total_time_s


def test_simulate_segment_time_is_faster_than_riding_at_cp_alone() -> None:
    """W' > 0 => la puissance soutenable (CP + W'/T) est toujours > CP pour un T
    fini => le temps simulé doit être strictement inférieur au temps à CP seul."""
    chunks = [SegmentChunk(0.0, 1000.0, 0.0, 0.0)]

    simulated_time_s = simulate_segment_time(
        chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )

    assert simulated_time_s < _cp_only_time_s(chunks)
    # ordre de grandeur : ~250W sur plat, CdA/Crr routiers -> autour de 30km/h,
    # donc 1km en 100-150s environ (cf. sanity check T-10 : 250W ~ 302W/40km/h)
    assert 60.0 < simulated_time_s < 180.0


def test_simulate_segment_time_w_prime_contribution_shrinks_for_longer_segments() -> None:
    """W'/T devient négligeable quand T grandit : l'écart relatif avec le temps
    à CP seul doit être plus petit sur un segment long que sur un segment court."""
    short_chunks = [SegmentChunk(0.0, 500.0, 0.0, 0.0)]
    long_chunks = [SegmentChunk(0.0, 20_000.0, 0.0, 0.0)]

    def gap_ratio(chunks: list[SegmentChunk]) -> float:
        cp_only = _cp_only_time_s(chunks)
        simulated = simulate_segment_time(
            chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
        )
        return (cp_only - simulated) / cp_only

    assert gap_ratio(long_chunks) < gap_ratio(short_chunks)


def test_simulate_segment_time_is_self_consistent_at_the_fixed_point() -> None:
    """Au point fixe, la puissance soutenable qu'implique le temps trouvé doit
    redonner (à la tolérance de convergence près) ce même temps."""
    chunks = [SegmentChunk(0.0, 3000.0, 0.02, 0.0)]  # légère montée

    predicted_time_s = simulate_segment_time(
        chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )

    sustainable_power_w = _CP_WATTS + _W_PRIME_JOULES / predicted_time_s
    speed_ms = cyclist_speed_from_power(
        sustainable_power_w, 0.02, 0.0, _MASS_KG, _CDA_M2, _CRR, STANDARD_AIR_DENSITY_KG_M3
    )
    recomputed_time_s = 3000.0 / speed_ms

    assert recomputed_time_s == pytest.approx(predicted_time_s, abs=0.1)


def test_simulate_segment_time_climbing_takes_longer_than_flat() -> None:
    flat_chunks = [SegmentChunk(0.0, 2000.0, 0.0, 0.0)]
    climb_chunks = [SegmentChunk(0.0, 2000.0, 0.06, 0.0)]

    flat_time_s = simulate_segment_time(
        flat_chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )
    climb_time_s = simulate_segment_time(
        climb_chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )

    assert climb_time_s > flat_time_s


def test_simulate_segment_time_is_consistent_across_chunking_granularity() -> None:
    """Pente constante : découper le même tronçon en 10 morceaux plutôt qu'1
    ne doit rien changer au temps total (vitesse constante à pente constante)."""
    one_chunk = [SegmentChunk(0.0, 1000.0, 0.03, 0.0)]
    ten_chunks = [SegmentChunk(i * 100.0, 100.0, 0.03, 0.0) for i in range(10)]

    time_one = simulate_segment_time(one_chunk, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)
    time_ten = simulate_segment_time(
        ten_chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )

    assert time_one == pytest.approx(time_ten, rel=1e-6)


def test_simulate_segment_time_raises_when_not_converged() -> None:
    chunks = [SegmentChunk(0.0, 1000.0, 0.0, 0.0)]
    with pytest.raises(ValueError, match="convergence"):
        simulate_segment_time(
            chunks,
            _CP_WATTS,
            _W_PRIME_JOULES,
            _MASS_KG,
            _CDA_M2,
            _CRR,
            max_iterations=5,
            convergence_tolerance_s=-1.0,  # tolérance impossible à satisfaire
        )


def test_simulate_segment_time_raises_on_empty_chunks() -> None:
    with pytest.raises(ValueError, match="tronçon"):
        simulate_segment_time([], _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)
