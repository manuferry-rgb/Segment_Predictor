"""Équation de puissance du cycliste, densité de l'air, et résolution inverse.

Fonctions pures — aucun I/O, aucune dépendance à ingest/storage.
"""

import math

from scipy.optimize import brentq

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


def cyclist_speed_from_power(
    power_w: float,
    grade: float,
    headwind_speed_ms: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float,
    speed_bounds_ms: tuple[float, float] = (0.01, 40.0),
) -> float:
    """Résout cyclist_power_required(v, ...) = power_w pour v, par la méthode de Brent.

    Pas de solution analytique simple : quand headwind=0, P(v) est cubique
    en v (le terme aéro en v³) ; avec du vent c'est pire. Brent encadre la
    racine par dichotomie/interpolation plutôt que d'inverser l'équation
    symboliquement.

    Suppose P(v) - power_w change de signe une seule fois sur
    `speed_bounds_ms`. Vrai pour `power_w > 0` (le cas d'usage : combien de
    vitesse pour pédaler à X watts), y compris en légère descente — à v->0,
    P(v)->0 toujours (P = F(v)·v, et c'est v qui l'emporte), donc pour
    power_w > 0 il n'y a qu'un franchissement. Pas garanti pour power_w <= 0
    (vitesse de roue libre en forte descente) : hors périmètre, une
    `ValueError` remonte plutôt qu'une racine non pertinente.
    """

    def residual(speed_ms: float) -> float:
        return (
            cyclist_power_required(
                speed_ms, grade, headwind_speed_ms, mass_kg, cda_m2, crr, air_density_kg_m3
            )
            - power_w
        )

    lo, hi = speed_bounds_ms
    try:
        return brentq(residual, lo, hi)
    except ValueError as error:
        raise ValueError(
            f"pas de vitesse solution dans {speed_bounds_ms} m/s pour {power_w}W "
            f"(pente={grade}, vent={headwind_speed_ms}m/s) : {error}"
        ) from error


def _sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def effective_headwind_speed_ms(
    wind_speed_ms: float, wind_direction_rad: float, heading_rad: float
) -> float:
    """Projette le vent sur le cap de déplacement : composante de vent de face.

    Deux conventions à ne pas confondre (piégeuses, d'où ce commentaire) :
    - `wind_direction_rad` (Open-Meteo, T-14) : direction D'OÙ VIENT le vent
      (convention météo standard) — 0 = vient du nord, croît vers l'est.
    - `heading_rad` (T-12) : direction OÙ JE VAIS (cap de déplacement) —
      mêmes axes (0 = nord, croît vers l'est), mais l'un est une origine,
      l'autre une destination. Les confondre inverse le signe du résultat.

    Un vent qui VIENT DE la direction où je vais souffle droit dans ma
    figure : vent de face pur quand wind_direction_rad == heading_rad,
    d'où cos(wind_direction_rad - heading_rad) plutôt qu'une autre
    combinaison de signes.

    Résultat positif = vent de face, négatif = vent arrière — même
    convention que `headwind_speed_ms` dans cyclist_power_required (T-10),
    donc branchable directement dessus. Un vent arrière plus fort que la
    vitesse au sol donne une vitesse d'air relative négative une fois
    sommé à `speed_ms` côté cyclist_power_required, qui la gère déjà
    (le terme aéro bascule en poussée via `_sign`) — rien de spécial à
    faire ici, cette fonction ne fait que projeter, pas de vitesse au sol
    en jeu.
    """
    if wind_speed_ms < 0:
        raise ValueError(f"wind_speed_ms doit être positif ou nul, reçu {wind_speed_ms}")

    return wind_speed_ms * math.cos(wind_direction_rad - heading_rad)
