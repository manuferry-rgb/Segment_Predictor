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
    normalized_power,
    resample_to_uniform_seconds,
    sustainable_power_w,
    training_stress_score,
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


def test_fit_reports_near_zero_uncertainty_on_a_noiseless_fit() -> None:
    """Un ajustement parfait (tous les points pile sur l'hyperbole) a une
    covariance quasi nulle — vérifié directement (pas de NaN, pas de
    valeur inventée quand les résidus sont nuls)."""
    durations = np.array([180.0, 300.0, 600.0, 900.0, 1200.0])
    mmp = _hyperbola(durations, 250.0, 20_000.0)

    fit = fit_critical_power(durations, mmp)

    assert fit.cp_watts_std == pytest.approx(0.0, abs=1e-6)
    assert fit.w_prime_joules_std == pytest.approx(0.0, abs=1e-3)


def test_fit_reports_uncertainty_matching_independent_polyfit_covariance() -> None:
    """T-28 : l'écart-type de CP/W' vient de la covariance de la régression
    déjà utilisée pour le fit (np.polyfit(..., cov=True)) — recalculé ici
    indépendamment de fit_critical_power, pas re-dérivé de son
    implémentation."""
    true_cp, true_w_prime = 280.0, 18_000.0
    durations = np.array([180.0, 240.0, 300.0, 420.0, 600.0, 900.0, 1200.0])
    mmp = _hyperbola(durations, true_cp, true_w_prime)
    rng = np.random.default_rng(7)
    noisy_mmp = mmp + rng.normal(loc=0.0, scale=3.0, size=mmp.shape)

    fit = fit_critical_power(durations, noisy_mmp)

    total_work = noisy_mmp * durations
    _, cov = np.polyfit(durations, total_work, 1, cov=True)
    expected_cp_std = float(np.sqrt(cov[0, 0]))
    expected_w_prime_std = float(np.sqrt(cov[1, 1]))

    assert fit.cp_watts_std == pytest.approx(expected_cp_std)
    assert fit.w_prime_joules_std == pytest.approx(expected_w_prime_std)
    assert fit.cp_watts_std > 0


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


# ---- sustainable_power_w (T-31) ---------------------------------------------------------------


def test_sustainable_power_matches_the_cp_hyperbola_by_construction() -> None:
    """P = CP + W'/t, exactement la formule ajustée par fit_critical_power —
    pas une nouvelle méthode, juste le même modèle évalué en avant plutôt
    qu'ajusté en arrière."""
    power_w = sustainable_power_w(cp_watts=250.0, w_prime_joules=20_000.0, duration_s=1000.0)
    assert power_w == pytest.approx(250.0 + 20_000.0 / 1000.0)


def test_sustainable_power_approaches_cp_for_very_long_durations() -> None:
    """Le terme W'/t s'annule quand t -> l'infini : l'asymptote du modèle
    à 2 paramètres est CP elle-même."""
    power_w = sustainable_power_w(cp_watts=250.0, w_prime_joules=20_000.0, duration_s=1_000_000.0)
    assert power_w == pytest.approx(250.0, abs=0.1)


def test_sustainable_power_raises_on_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        sustainable_power_w(cp_watts=250.0, w_prime_joules=20_000.0, duration_s=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        sustainable_power_w(cp_watts=250.0, w_prime_joules=20_000.0, duration_s=-10.0)


# ---- normalized_power (T-21) ----------------------------------------------------------------


def test_normalized_power_equals_constant_for_steady_power() -> None:
    """Puissance parfaitement constante : la moyenne glissante 30s vaut
    tout le temps cette constante, donc NP = cette constante (le cas où
    NP et puissance moyenne coïncident)."""
    watts = np.full(120, 200.0)
    assert normalized_power(watts) == pytest.approx(200.0, abs=1e-6)


def test_normalized_power_exceeds_simple_average_when_power_varies() -> None:
    """C'est tout l'intérêt de NP par rapport à une moyenne simple : la
    puissance^4 pénalise les pics, donc une puissance très variable (même
    moyenne simple) a un coût physiologique plus élevé, reflété par
    NP > moyenne arithmétique."""
    watts = np.tile([50.0] * 30 + [400.0] * 30, 5)  # alterne repos/sprint, 5 fois

    assert normalized_power(watts) > float(np.mean(watts))


def test_normalized_power_ignores_windows_overlapping_a_gap() -> None:
    """Un trou (NaN, ex. perte de capteur) ne doit pas se propager dans
    tout le résultat : seules les fenêtres de 30s qui touchent le trou
    sont exclues, pas le stream entier."""
    watts = np.full(90, 200.0)
    watts[40:50] = np.nan

    assert normalized_power(watts) == pytest.approx(200.0, abs=1e-6)


def test_normalized_power_raises_when_stream_shorter_than_window() -> None:
    watts = np.full(10, 200.0)  # < 30s, aucune fenêtre complète possible
    with pytest.raises(ValueError, match="30"):
        normalized_power(watts)


def test_normalized_power_raises_when_no_window_fully_valid() -> None:
    """Contrairement à mean_maximal_power (qui renvoie NaN pour une courbe
    MMP agrégée sur plein d'activités, où un trou isolé est acceptable),
    NP alimente directement un TSS par activité : un NaN silencieux ici
    corromprait toute la récursion de Banister (T-21) en aval sans qu'on
    s'en aperçoive. Erreur explicite plutôt qu'une valeur invalide qui se
    propage."""
    watts = np.full(60, np.nan)
    with pytest.raises(ValueError, match="fenêtre"):
        normalized_power(watts)


# ---- training_stress_score (T-21) ------------------------------------------------------------


def test_tss_at_threshold_power_for_one_hour_is_100_by_definition() -> None:
    """Définition de Coggan : 1h pile à la puissance seuil = 100 TSS, le
    point d'ancrage de toute l'échelle."""
    tss = training_stress_score(
        duration_s=3600.0, normalized_power_w=250.0, threshold_power_w=250.0
    )
    assert tss == pytest.approx(100.0, abs=1e-6)


def test_tss_scales_linearly_with_duration_at_constant_intensity() -> None:
    tss_1h = training_stress_score(3600.0, 250.0, 250.0)
    tss_30min = training_stress_score(1800.0, 250.0, 250.0)
    assert tss_30min == pytest.approx(tss_1h / 2)


def test_tss_scales_with_square_of_intensity_factor() -> None:
    """TSS = durée × IF² × 100/3600 : doubler l'intensité (IF) quadruple le
    TSS à durée égale — l'effet dominant qui fait qu'une sortie courte et
    intense peut peser plus lourd qu'une longue sortie tranquille."""
    tss_at_threshold = training_stress_score(3600.0, 250.0, 250.0)
    tss_double_intensity = training_stress_score(3600.0, 500.0, 250.0)
    assert tss_double_intensity == pytest.approx(tss_at_threshold * 4)


def test_tss_raises_for_nonpositive_threshold_power() -> None:
    with pytest.raises(ValueError, match="threshold_power_w"):
        training_stress_score(3600.0, 250.0, 0.0)
