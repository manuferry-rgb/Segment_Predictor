"""Équation de puissance du cycliste et densité de l'air.

Fonctions pures — aucun I/O, aucune dépendance à ingest/storage.
"""

import math

# Gravité standard, valeur exacte par définition internationale.
STANDARD_GRAVITY_MS2 = 9.80665

# Constantes de l'atmosphère standard internationale (ISA), utilisées pour
# la relation pression <-> altitude dans air_density().
_SEA_LEVEL_PRESSURE_PA = 101_325.0
_SEA_LEVEL_TEMPERATURE_K = 288.15
_TEMPERATURE_LAPSE_RATE_K_PER_M = 0.0065
_MOLAR_MASS_AIR_KG_PER_MOL = 0.0289644
_UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K = 8.31446
# Constante spécifique de l'air, dérivée des deux précédentes plutôt que
# codée séparément : évite un écart d'arrondi entre deux constantes qui
# devraient être exactement cohérentes entre elles.
_SPECIFIC_GAS_CONSTANT_AIR_J_PER_KG_K = (
    _UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K / _MOLAR_MASS_AIR_KG_PER_MOL
)


def air_density(altitude_m: float, temperature_k: float) -> float:
    """Densité de l'air (kg/m³) à une altitude et une température données.

    Approximation assumée : la pression à l'altitude donnée est calculée
    via la formule barométrique de l'atmosphère standard (ISA), qui
    suppose son propre profil de température standard pour la relation
    pression/altitude — pas la température réelle passée en paramètre.
    C'est cette température réelle qui sert ensuite, via la loi des gaz
    parfaits, à convertir cette pression en densité. On n'a pas de profil
    atmosphérique complet mesuré (seulement une température au sol), donc
    ce mélange ISA-pour-la-pression / mesure-réelle-pour-la-température
    est l'approximation standard des calculateurs de puissance cycliste,
    pas une simulation atmosphérique complète.
    """
    pressure_pa = _SEA_LEVEL_PRESSURE_PA * (
        1 - _TEMPERATURE_LAPSE_RATE_K_PER_M * altitude_m / _SEA_LEVEL_TEMPERATURE_K
    ) ** (
        STANDARD_GRAVITY_MS2
        * _MOLAR_MASS_AIR_KG_PER_MOL
        / (_UNIVERSAL_GAS_CONSTANT_J_PER_MOL_K * _TEMPERATURE_LAPSE_RATE_K_PER_M)
    )
    return pressure_pa / (_SPECIFIC_GAS_CONSTANT_AIR_J_PER_KG_K * temperature_k)


def cyclist_power_required(
    speed_ms: float,
    grade: float,
    headwind_speed_ms: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float,
) -> float:
    """Puissance (W) nécessaire pour maintenir `speed_ms` sur une pente `grade`.

    P = (F_gravité + F_roulement + F_aéro) · vitesse. `grade` est une pente
    (rise/run, ex. 0.10 pour 10%), pas un angle : sin(θ) et cos(θ) sont
    calculés exactement (sin θ = grade/√(1+grade²)), pas via l'approximation
    petits angles sin θ ≈ grade — même coût de calcul, plus juste sur les
    pentes raides.

    `headwind_speed_ms` : positif = vent de face (freine), négatif = vent
    arrière (aide). Pas de rendement de transmission ici (hors périmètre) :
    c'est la puissance à la roue, pas au pédalier.
    """
    sin_theta = grade / math.sqrt(1 + grade**2)
    cos_theta = 1 / math.sqrt(1 + grade**2)

    gravity_force_n = mass_kg * STANDARD_GRAVITY_MS2 * sin_theta
    rolling_force_n = crr * mass_kg * STANDARD_GRAVITY_MS2 * cos_theta

    # La traînée dépend de la vitesse relative à l'air, pas de la vitesse
    # sol : un vent arrière plus fort que le vélo fait que l'air pousse au
    # lieu de freiner (d'où le signe explicite plutôt qu'un simple carré).
    relative_air_speed_ms = speed_ms + headwind_speed_ms
    aero_force_n = (
        0.5 * air_density_kg_m3 * cda_m2 * relative_air_speed_ms**2 * _sign(relative_air_speed_ms)
    )

    total_force_n = gravity_force_n + rolling_force_n + aero_force_n
    return total_force_n * speed_ms


def _sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0
