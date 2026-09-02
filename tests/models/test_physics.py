"""Tests de l'équation de puissance du cycliste, de la densité de l'air (T-10)
et de la résolution inverse puissance -> vitesse (T-11).

Fonctions pures, aucun I/O. Chaque valeur attendue est recalculée
indépendamment dans le test (pas juste comparée à elle-même) — c'est ce
qui rend les cas "vérifiables à la main".
"""

import math

import numpy as np
import pytest

from segment_predictor.models.physics import (
    STANDARD_GRAVITY_MS2,
    air_density,
    cyclist_power_required,
    cyclist_speed_from_power,
)

# ---- cyclist_power_required ------------------------------------------------------------


def test_power_flat_no_wind_gravity_is_exactly_zero() -> None:
    """Plat (grade=0) : sin(0)=0 exactement, la composante gravité est nulle,
    pas juste négligeable — c'est le cas dégénéré demandé."""
    speed_ms = 10.0
    mass_kg = 80.0
    crr = 0.004
    cda_m2 = 0.30
    air_density_kg_m3 = 1.2

    expected_rolling_n = crr * mass_kg * STANDARD_GRAVITY_MS2  # cos(0) = 1
    expected_aero_n = 0.5 * air_density_kg_m3 * cda_m2 * speed_ms**2
    expected_power_w = (expected_rolling_n + expected_aero_n) * speed_ms

    power_w = cyclist_power_required(speed_ms, 0.0, 0.0, mass_kg, cda_m2, crr, air_density_kg_m3)

    assert power_w == pytest.approx(expected_power_w)


def test_power_steep_climb_low_speed_aero_is_negligible() -> None:
    """Montée à 15%, 1.5 m/s (~5.4 km/h) : la gravité domine largement,
    l'aéro (en v²) devient minuscule à cette vitesse — vérifié quantitativement,
    pas juste "petit à l'œil"."""
    grade = 0.15
    speed_ms = 1.5
    mass_kg = 85.0
    crr = 0.005
    cda_m2 = 0.30
    air_density_kg_m3 = 1.2

    sin_theta = grade / math.sqrt(1 + grade**2)
    cos_theta = 1 / math.sqrt(1 + grade**2)
    expected_gravity_n = mass_kg * STANDARD_GRAVITY_MS2 * sin_theta
    expected_rolling_n = crr * mass_kg * STANDARD_GRAVITY_MS2 * cos_theta
    expected_aero_n = 0.5 * air_density_kg_m3 * cda_m2 * speed_ms**2
    expected_power_w = (expected_gravity_n + expected_rolling_n + expected_aero_n) * speed_ms

    power_w = cyclist_power_required(speed_ms, grade, 0.0, mass_kg, cda_m2, crr, air_density_kg_m3)

    assert power_w == pytest.approx(expected_power_w)
    assert (expected_aero_n * speed_ms) / power_w < 0.01  # < 1% de la puissance totale


def test_power_uses_exact_trig_not_small_angle_approximation() -> None:
    """sin(θ) = pente/√(1+pente²), pas l'approximation sin(θ) ≈ pente."""
    grade = 0.25  # pente raide : l'écart entre les deux formules devient visible
    speed_ms = 5.0
    mass_kg = 80.0
    crr = 0.0
    cda_m2 = 0.0  # isole la composante gravité

    exact_sin_theta = grade / math.sqrt(1 + grade**2)
    approx_sin_theta = grade  # ce que donnerait l'approximation petits angles
    assert exact_sin_theta != pytest.approx(approx_sin_theta, rel=1e-3)  # l'écart existe bien

    power_w = cyclist_power_required(speed_ms, grade, 0.0, mass_kg, cda_m2, crr, 1.2)
    expected_power_w = mass_kg * STANDARD_GRAVITY_MS2 * exact_sin_theta * speed_ms

    assert power_w == pytest.approx(expected_power_w)


def test_power_headwind_increases_required_power() -> None:
    base_kwargs = dict(
        speed_ms=8.0, grade=0.0, mass_kg=80.0, cda_m2=0.30, crr=0.004, air_density_kg_m3=1.2
    )
    power_no_wind = cyclist_power_required(headwind_speed_ms=0.0, **base_kwargs)
    power_headwind = cyclist_power_required(headwind_speed_ms=3.0, **base_kwargs)

    assert power_headwind > power_no_wind


def test_power_tailwind_decreases_required_power() -> None:
    base_kwargs = dict(
        speed_ms=8.0, grade=0.0, mass_kg=80.0, cda_m2=0.30, crr=0.004, air_density_kg_m3=1.2
    )
    power_no_wind = cyclist_power_required(headwind_speed_ms=0.0, **base_kwargs)
    power_tailwind = cyclist_power_required(headwind_speed_ms=-3.0, **base_kwargs)

    assert power_tailwind < power_no_wind


def test_power_downhill_is_less_than_flat() -> None:
    base_kwargs = dict(
        speed_ms=8.0,
        headwind_speed_ms=0.0,
        mass_kg=80.0,
        cda_m2=0.30,
        crr=0.004,
        air_density_kg_m3=1.2,
    )
    power_flat = cyclist_power_required(grade=0.0, **base_kwargs)
    power_downhill = cyclist_power_required(grade=-0.05, **base_kwargs)

    assert power_downhill < power_flat


def test_power_strong_tailwind_can_make_aero_force_assist() -> None:
    """Si le vent arrière est plus fort que la vitesse du vélo, l'air pousse
    plutôt qu'il ne freine : la force aéro change de signe (physiquement correct).
    """
    speed_ms = 2.0
    headwind_speed_ms = -10.0  # vent arrière à 10 m/s, largement > vitesse vélo
    power_w = cyclist_power_required(speed_ms, 0.0, headwind_speed_ms, 80.0, 0.30, 0.004, 1.2)

    # roulement seul (positif) moins une aéro qui pousse maintenant : la puissance
    # nécessaire doit être inférieure à la puissance de roulement seule
    rolling_only_w = 0.004 * 80.0 * STANDARD_GRAVITY_MS2 * speed_ms
    assert power_w < rolling_only_w


# ---- air_density -------------------------------------------------------------------------


def test_air_density_matches_isa_sea_level_reference() -> None:
    """0 m, 15°C (288.15 K) : valeur de référence ISA bien connue, 1.225 kg/m³."""
    density = air_density(altitude_m=0.0, temperature_k=288.15)
    assert density == pytest.approx(1.225, abs=1e-3)


def test_air_density_decreases_with_altitude() -> None:
    density_sea_level = air_density(altitude_m=0.0, temperature_k=288.15)
    density_at_2000m = air_density(altitude_m=2000.0, temperature_k=288.15)
    assert density_at_2000m < density_sea_level


def test_air_density_decreases_when_warmer() -> None:
    density_cold = air_density(altitude_m=500.0, temperature_k=273.15)  # 0°C
    density_hot = air_density(altitude_m=500.0, temperature_k=308.15)  # 35°C
    assert density_hot < density_cold


# ---- cyclist_speed_from_power (T-11) ----------------------------------------------------


def test_speed_from_power_round_trip_on_random_cases() -> None:
    """vitesse -> puissance -> vitesse doit retomber sur ses pieds, sur des cas
    tirés aléatoirement (graine fixe, plages physiquement réalistes).

    Sur une pente négative, P(v) peut être négative à basse vitesse (roue
    libre, hors périmètre documenté de cyclist_speed_from_power) : un premier
    essai avec ces bornes l'a révélé concrètement (grade=-2.6%, v=2m/s ->
    P≈-61W). On filtre donc les tirages où la puissance directe n'est pas
    positive plutôt que de deviner des bornes de pente qui l'excluent a
    priori — c'est le périmètre réel de la fonction qui décide, pas une
    borne arbitraire choisie sans vérifier.
    """
    rng = np.random.default_rng(seed=123)
    n_candidates = 500

    speeds_ms = rng.uniform(2.0, 15.0, n_candidates)
    grades = rng.uniform(-0.03, 0.12, n_candidates)
    headwinds_ms = rng.uniform(-3.0, 3.0, n_candidates)
    masses_kg = rng.uniform(60.0, 100.0, n_candidates)
    cdas_m2 = rng.uniform(0.25, 0.40, n_candidates)
    crrs = rng.uniform(0.003, 0.008, n_candidates)
    air_densities = rng.uniform(1.0, 1.25, n_candidates)

    n_valid = 0
    for i in range(n_candidates):
        power_w = cyclist_power_required(
            speeds_ms[i],
            grades[i],
            headwinds_ms[i],
            masses_kg[i],
            cdas_m2[i],
            crrs[i],
            air_densities[i],
        )
        if power_w <= 0:
            continue  # roue libre pour ce tirage : hors périmètre, on saute

        n_valid += 1
        recovered_speed_ms = cyclist_speed_from_power(
            power_w, grades[i], headwinds_ms[i], masses_kg[i], cdas_m2[i], crrs[i], air_densities[i]
        )
        assert recovered_speed_ms == pytest.approx(speeds_ms[i], rel=1e-6), (
            f"trial {i}: v={speeds_ms[i]}, grade={grades[i]}, headwind={headwinds_ms[i]}"
        )

    assert n_valid >= 200  # assez de cas valides pour que le test soit significatif


def test_speed_from_power_flat_no_wind_matches_hand_check() -> None:
    """Cas simple, plat sans vent : P = Crr·m·g·v + 0.5·ρ·CdA·v³ (cubique en v)."""
    mass_kg, crr, cda_m2, air_density_kg_m3 = 80.0, 0.005, 0.30, 1.2
    true_speed_ms = 9.0
    power_w = cyclist_power_required(
        true_speed_ms, 0.0, 0.0, mass_kg, cda_m2, crr, air_density_kg_m3
    )

    recovered_speed_ms = cyclist_speed_from_power(
        power_w, 0.0, 0.0, mass_kg, cda_m2, crr, air_density_kg_m3
    )

    assert recovered_speed_ms == pytest.approx(true_speed_ms, rel=1e-8)


def test_speed_from_power_raises_when_no_root_in_bounds() -> None:
    """Puissance hors de portée des bornes de recherche : erreur explicite, pas
    une vitesse fausse."""
    with pytest.raises(ValueError, match="vitesse"):
        cyclist_speed_from_power(
            power_w=100_000.0,  # bien au-delà de ce qu'atteint 40 m/s
            grade=0.0,
            headwind_speed_ms=0.0,
            mass_kg=80.0,
            cda_m2=0.30,
            crr=0.005,
            air_density_kg_m3=1.2,
        )
