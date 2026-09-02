"""Tests de la courbe de puissance (T-08) et du modèle Critical Power (T-09).

Fonctions pures, aucun I/O. Chaque cas T-08 est vérifiable à la main ;
les cas T-09 partent d'un CP/W' connu et vérifient que le fit les retrouve.
"""

import math

import numpy as np
import pytest

from segment_predictor.models.power import (
    fit_critical_power,
    mean_maximal_power,
    mean_maximal_power_curve,
    resample_to_uniform_seconds,
)

# ---- mean_maximal_power --------------------------------------------------------------


def test_mmp_constant_power_equals_that_power_for_any_duration() -> None:
    watts = np.full(100, 200.0)
    assert mean_maximal_power(watts, 1) == 200.0
    assert mean_maximal_power(watts, 50) == 200.0
    assert mean_maximal_power(watts, 100) == 200.0


def test_mmp_ramp_duration_1_is_the_single_max_sample() -> None:
    watts = np.arange(100, dtype=float)  # 0, 1, ..., 99
    assert mean_maximal_power(watts, 1) == 99.0


def test_mmp_ramp_duration_10_is_mean_of_last_10_samples() -> None:
    watts = np.arange(100, dtype=float)  # 0, 1, ..., 99
    # meilleure fenêtre = les 10 dernières valeurs [90..99], moyenne = 94.5
    assert mean_maximal_power(watts, 10) == pytest.approx(94.5)


def test_mmp_isolated_peak_block_duration_matches_block_size() -> None:
    watts = np.concatenate([np.full(50, 100.0), np.full(10, 500.0), np.full(50, 100.0)])
    assert mean_maximal_power(watts, 10) == pytest.approx(500.0)


def test_mmp_isolated_peak_block_duration_wider_than_block() -> None:
    watts = np.concatenate([np.full(50, 100.0), np.full(10, 500.0), np.full(50, 100.0)])
    # meilleure fenêtre de 20 = les 10 valeurs à 500 + 10 voisines à 100
    # somme = 10*500 + 10*100 = 6000, moyenne = 300
    assert mean_maximal_power(watts, 20) == pytest.approx(300.0)


def test_mmp_window_overlapping_a_gap_is_excluded() -> None:
    watts = np.array([100.0, 100.0, 100.0, np.nan, 300.0, 300.0, 300.0])
    # fenêtres de 3 valides : [0:3] (moyenne 100) et [4:7] (moyenne 300)
    # toute fenêtre chevauchant l'indice 3 (NaN) est exclue
    assert mean_maximal_power(watts, 3) == pytest.approx(300.0)


def test_mmp_returns_nan_when_duration_exceeds_signal_length() -> None:
    watts = np.full(5, 100.0)
    assert math.isnan(mean_maximal_power(watts, 10))


def test_mmp_returns_nan_when_all_values_are_gaps() -> None:
    watts = np.full(10, np.nan)
    assert math.isnan(mean_maximal_power(watts, 3))


def test_mmp_curve_is_not_guaranteed_monotonic_across_arbitrary_durations() -> None:
    """La MMP(d) >= MMP(D) pour d < D n'est garantie mathématiquement QUE si
    d divise D (argument de partition en blocs égaux) — pas en général.
    Contre-exemple minimal : [10, 0, 10] a une moyenne de 6.67 sur 3s, mais
    ses deux fenêtres de 2s ([10,0] et [0,10]) ne font que 5 chacune :
    aucune fenêtre plus courte ne "domine" la plus longue ici. Documenté
    plutôt que "corrigé" — forcer la monotonie inventerait une valeur.
    """
    watts = np.array([10.0, 0.0, 10.0])
    assert mean_maximal_power(watts, 3) == pytest.approx(20 / 3)
    assert mean_maximal_power(watts, 2) == pytest.approx(5.0)


def test_mmp_raises_on_non_positive_duration() -> None:
    watts = np.full(10, 100.0)
    with pytest.raises(ValueError, match="duration_s"):
        mean_maximal_power(watts, 0)
    with pytest.raises(ValueError, match="duration_s"):
        mean_maximal_power(watts, -5)


# ---- resample_to_uniform_seconds -------------------------------------------------------


def test_resample_no_gap_returns_watts_unchanged() -> None:
    t_s = np.array([0, 1, 2, 3])
    watts = np.array([100.0, 150.0, 200.0, 180.0])
    result = resample_to_uniform_seconds(t_s, watts)
    np.testing.assert_array_equal(result, watts)


def test_resample_fills_pause_gap_with_nan() -> None:
    # pause auto entre t=2 et t=5 : les secondes 3 et 4 n'ont pas été enregistrées
    t_s = np.array([0, 1, 2, 5, 6])
    watts = np.array([100.0, 110.0, 120.0, 200.0, 210.0])

    result = resample_to_uniform_seconds(t_s, watts)

    assert len(result) == 7
    np.testing.assert_array_equal(result[0:3], [100.0, 110.0, 120.0])
    assert math.isnan(result[3])
    assert math.isnan(result[4])
    np.testing.assert_array_equal(result[5:7], [200.0, 210.0])


def test_resample_normalizes_start_time_to_zero() -> None:
    t_s = np.array([10, 11, 12])
    watts = np.array([50.0, 60.0, 70.0])
    result = resample_to_uniform_seconds(t_s, watts)
    np.testing.assert_array_equal(result, [50.0, 60.0, 70.0])


def test_resample_raises_on_empty_stream() -> None:
    with pytest.raises(ValueError, match="vide"):
        resample_to_uniform_seconds(np.array([]), np.array([]))


def test_resample_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="longueur"):
        resample_to_uniform_seconds(np.array([0, 1, 2]), np.array([100.0, 110.0]))


def test_resample_raises_on_non_increasing_time() -> None:
    with pytest.raises(ValueError, match="croissant"):
        resample_to_uniform_seconds(np.array([0, 1, 1, 2]), np.array([100.0, 110.0, 120.0, 130.0]))
    with pytest.raises(ValueError, match="croissant"):
        resample_to_uniform_seconds(np.array([0, 2, 1]), np.array([100.0, 110.0, 120.0]))


# ---- mean_maximal_power_curve ----------------------------------------------------------


def test_mmp_curve_matches_individual_calls() -> None:
    watts = np.arange(100, dtype=float)
    curve = mean_maximal_power_curve(watts, [1, 10, 100])

    assert curve == {
        1: mean_maximal_power(watts, 1),
        10: mean_maximal_power(watts, 10),
        100: mean_maximal_power(watts, 100),
    }


def test_mmp_curve_includes_nan_for_infeasible_durations_without_raising() -> None:
    watts = np.full(5, 100.0)
    curve = mean_maximal_power_curve(watts, [1, 3, 10])

    assert curve[1] == 100.0
    assert curve[3] == 100.0
    assert math.isnan(curve[10])


# ---- fit_critical_power (T-09) ---------------------------------------------------------


def _hyperbola(durations_s: np.ndarray, cp: float, w_prime: float) -> np.ndarray:
    """P(t) = CP + W'/t — génère des points MMP synthétiques exacts."""
    return cp + w_prime / durations_s


def test_fit_recovers_exact_cp_and_w_prime_from_noiseless_data() -> None:
    true_cp, true_w_prime = 250.0, 20_000.0
    durations = np.array([180.0, 300.0, 600.0, 900.0, 1200.0])
    mmp = _hyperbola(durations, true_cp, true_w_prime)

    fit = fit_critical_power(durations, mmp)

    assert fit.cp_watts == pytest.approx(true_cp, abs=1e-6)
    assert fit.w_prime_joules == pytest.approx(true_w_prime, abs=1e-3)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.n_points == 5
    assert fit.duration_range_s == (180, 1200)


def test_fit_excludes_points_outside_the_duration_range() -> None:
    true_cp, true_w_prime = 250.0, 20_000.0
    # 60s et 3600s sont hors plage par défaut [180, 1200] ; s'ils étaient
    # inclus avec une valeur incohérente avec l'hyperbole, ils fausseraient le fit.
    durations = np.array([60.0, 180.0, 300.0, 600.0, 900.0, 1200.0, 3600.0])
    mmp = _hyperbola(durations, true_cp, true_w_prime)
    mmp[0] = 900.0  # incohérent avec l'hyperbole, doit être ignoré (hors plage)
    mmp[-1] = 50.0  # idem

    fit = fit_critical_power(durations, mmp)

    assert fit.n_points == 5  # seulement les 5 points dans [180, 1200]
    assert fit.cp_watts == pytest.approx(true_cp, abs=1e-6)
    assert fit.w_prime_joules == pytest.approx(true_w_prime, abs=1e-3)


def test_fit_ignores_nan_points_within_range() -> None:
    true_cp, true_w_prime = 250.0, 20_000.0
    durations = np.array([180.0, 300.0, 600.0, 900.0, 1200.0])
    mmp = _hyperbola(durations, true_cp, true_w_prime)
    mmp[2] = np.nan  # trou dans la courbe MMP à 600s

    fit = fit_critical_power(durations, mmp)

    assert fit.n_points == 4
    assert fit.cp_watts == pytest.approx(true_cp, abs=1e-6)


def test_fit_recovers_approximate_cp_and_w_prime_with_noise() -> None:
    true_cp, true_w_prime = 280.0, 18_000.0
    durations = np.array([180.0, 240.0, 300.0, 420.0, 600.0, 900.0, 1200.0])
    mmp = _hyperbola(durations, true_cp, true_w_prime)

    rng = np.random.default_rng(seed=42)
    noisy_mmp = mmp + rng.normal(loc=0.0, scale=3.0, size=mmp.shape)  # +/- 3W de bruit

    fit = fit_critical_power(durations, noisy_mmp)

    assert fit.cp_watts == pytest.approx(true_cp, rel=0.02)
    assert fit.w_prime_joules == pytest.approx(true_w_prime, rel=0.05)
    assert fit.r_squared > 0.95


def test_fit_accepts_custom_duration_range() -> None:
    true_cp, true_w_prime = 250.0, 20_000.0
    durations = np.array([60.0, 120.0, 180.0])
    mmp = _hyperbola(durations, true_cp, true_w_prime)

    fit = fit_critical_power(durations, mmp, duration_range_s=(60, 180))

    assert fit.n_points == 3
    assert fit.cp_watts == pytest.approx(true_cp, abs=1e-6)
    assert fit.duration_range_s == (60, 180)


def test_fit_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="longueur"):
        fit_critical_power(np.array([180.0, 300.0]), np.array([250.0]))


def test_fit_raises_on_invalid_duration_range() -> None:
    durations = np.array([180.0, 300.0, 600.0])
    mmp = _hyperbola(durations, 250.0, 20_000.0)
    with pytest.raises(ValueError, match="duration_range_s"):
        fit_critical_power(durations, mmp, duration_range_s=(0, 100))
    with pytest.raises(ValueError, match="duration_range_s"):
        fit_critical_power(durations, mmp, duration_range_s=(600, 300))


def test_fit_raises_when_fewer_than_two_valid_points_in_range() -> None:
    durations = np.array([180.0, 5000.0])  # un seul point dans [180, 1200]
    mmp = _hyperbola(durations, 250.0, 20_000.0)
    with pytest.raises(ValueError, match="point"):
        fit_critical_power(durations, mmp)
