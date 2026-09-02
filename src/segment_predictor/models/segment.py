"""Découpage d'un profil (distance, altitude, lat/lng) en tronçons ~50m,
et simulation du temps pour parcourir ces tronçons (T-13).

Fonctions pures — aucun I/O. Trois étapes séparées et composables :
`smooth_altitude` (débruite l'altitude GPS) -> `chunk_segment` (découpe
en tronçons, pente + cap par tronçon) -> `simulate_segment_time` (temps
prédit, à la puissance soutenable par le modèle CP).
"""

import math
from dataclasses import dataclass

import numpy as np

from .physics import air_density, cyclist_speed_from_power

# ISA, niveau de la mer, 15°C — dérivé de air_density() plutôt qu'un
# littéral séparé, pour rester cohérent avec T-10 par construction.
STANDARD_AIR_DENSITY_KG_M3 = air_density(altitude_m=0.0, temperature_k=288.15)

DEFAULT_MAX_ITERATIONS = 50
DEFAULT_CONVERGENCE_TOLERANCE_S = 0.1


def smooth_altitude(
    distance_m: np.ndarray, altitude_m: np.ndarray, window_m: float = 30.0
) -> np.ndarray:
    """Moyenne glissante en DISTANCE (pas en nombre d'échantillons).

    Le GPS échantillonne à ~1Hz en temps, donc l'espacement en mètres
    varie avec la vitesse — une moyenne sur un nombre fixe d'échantillons
    n'aurait pas une largeur physique constante. Ici, pour chaque point,
    on moyenne tous les relevés bruts dans une fenêtre de ±window_m/2
    autour de lui. Le bruit d'altitude GPS est à peu près centré et peu
    corrélé d'un relevé au suivant, alors que le vrai relief ne change
    pas significativement sur `window_m` : la moyenne annule le bruit
    sans effacer la pente réelle. Limite assumée : une vraie cassure
    nette du terrain est adoucie sur la largeur de la fenêtre, pas
    préservée (voir test_smooth_altitude_blurs_a_sharp_step).

    Implémentation : mêmes sommes cumulées que mean_maximal_power (T-08),
    mais indexées par distance (via recherche binaire) plutôt que par
    position fixe — O(n log n), pas de boucle Python sur chaque fenêtre.
    """
    distance_m = np.asarray(distance_m, dtype=float)
    altitude_m = np.asarray(altitude_m, dtype=float)
    if len(distance_m) != len(altitude_m):
        raise ValueError(
            f"distance_m et altitude_m n'ont pas la même longueur "
            f"({len(distance_m)} vs {len(altitude_m)})"
        )
    if len(distance_m) == 0:
        raise ValueError("stream vide : rien à lisser")
    if np.any(np.diff(distance_m) < 0):
        raise ValueError("distance_m doit être croissante (ou constante), pas décroissante")

    half_window_m = window_m / 2
    lo_idx = np.searchsorted(distance_m, distance_m - half_window_m, side="left")
    hi_idx = np.searchsorted(distance_m, distance_m + half_window_m, side="right")

    cumsum = np.concatenate(([0.0], np.cumsum(altitude_m)))
    window_sums = cumsum[hi_idx] - cumsum[lo_idx]
    window_counts = hi_idx - lo_idx
    return window_sums / window_counts


def _bearing_rad(lat1_deg: float, lng1_deg: float, lat2_deg: float, lng2_deg: float) -> float:
    """Cap initial (relèvement) du point 1 vers le point 2 : 0 = nord, croît vers l'est."""
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    delta_lng = math.radians(lng2_deg - lng1_deg)
    x = math.sin(delta_lng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lng)
    return math.atan2(x, y) % (2 * math.pi)


@dataclass(frozen=True)
class SegmentChunk:
    """Un tronçon d'environ `chunk_length_m` (le dernier peut être plus court)."""

    start_distance_m: float
    length_m: float
    grade: float  # rise/run, comme le paramètre `grade` de physics.cyclist_power_required
    heading_rad: float  # 0 = nord, croît vers l'est


def chunk_segment(
    distance_m: np.ndarray,
    smoothed_altitude_m: np.ndarray,
    lat: np.ndarray,
    lng: np.ndarray,
    chunk_length_m: float = 50.0,
) -> list[SegmentChunk]:
    """Découpe un profil en tronçons de ~`chunk_length_m`, pente + cap par tronçon.

    `smoothed_altitude_m` doit déjà être lissée (voir smooth_altitude) —
    cette fonction ne lisse rien, elle découpe et calcule des pentes/caps.
    Le découpage se fait sur `distance_m` (fourni par Strava, pas
    recalculé depuis lat/lng — évite d'ajouter notre propre bruit de
    reconstruction). Aux frontières de tronçon, qui ne tombent
    généralement pas exactement sur un échantillon brut, altitude et
    lat/lng sont interpolées linéairement plutôt que d'utiliser
    l'échantillon le plus proche.
    """
    distance_m = np.asarray(distance_m, dtype=float)
    smoothed_altitude_m = np.asarray(smoothed_altitude_m, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)

    lengths = {len(distance_m), len(smoothed_altitude_m), len(lat), len(lng)}
    if len(lengths) != 1:
        raise ValueError(
            "distance_m, smoothed_altitude_m, lat, lng doivent avoir la même longueur "
            f"(reçu {len(distance_m)}, {len(smoothed_altitude_m)}, {len(lat)}, {len(lng)})"
        )
    if len(distance_m) < 2:
        raise ValueError("il faut au moins 2 points pour découper un tronçon")
    if chunk_length_m <= 0:
        raise ValueError(f"chunk_length_m doit être positif, reçu {chunk_length_m}")

    total_distance_m = distance_m[-1] - distance_m[0]
    n_chunks = math.ceil(total_distance_m / chunk_length_m)
    if n_chunks == 0:
        return []

    boundaries_m = distance_m[0] + np.arange(n_chunks + 1) * chunk_length_m
    boundaries_m[-1] = distance_m[
        -1
    ]  # dernier tronçon : jusqu'à la fin exacte (peut être plus court)

    boundary_altitudes = np.interp(boundaries_m, distance_m, smoothed_altitude_m)
    boundary_lats = np.interp(boundaries_m, distance_m, lat)
    boundary_lngs = np.interp(boundaries_m, distance_m, lng)

    chunks = []
    for i in range(n_chunks):
        length_m = boundaries_m[i + 1] - boundaries_m[i]
        if length_m <= 0:
            continue  # cas dégénéré : total_distance_m multiple exact de chunk_length_m
        elevation_change_m = boundary_altitudes[i + 1] - boundary_altitudes[i]
        chunks.append(
            SegmentChunk(
                start_distance_m=float(boundaries_m[i]),
                length_m=float(length_m),
                grade=float(elevation_change_m / length_m),
                heading_rad=_bearing_rad(
                    boundary_lats[i], boundary_lngs[i], boundary_lats[i + 1], boundary_lngs[i + 1]
                ),
            )
        )
    return chunks


def _simulate_at_constant_power(
    chunks: list[SegmentChunk],
    power_w: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float,
) -> float:
    """Temps total pour parcourir tous les tronçons à une puissance CONSTANTE.

    Pas de vent (headwind=0.0) ni de draft à ce stade : "conditions
    standard", ajoutés en T-14/T-19.
    """
    total_time_s = 0.0
    for chunk in chunks:
        speed_ms = cyclist_speed_from_power(
            power_w, chunk.grade, 0.0, mass_kg, cda_m2, crr, air_density_kg_m3
        )
        total_time_s += chunk.length_m / speed_ms
    return total_time_s


def simulate_segment_time(
    chunks: list[SegmentChunk],
    cp_watts: float,
    w_prime_joules: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float = STANDARD_AIR_DENSITY_KG_M3,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    convergence_tolerance_s: float = DEFAULT_CONVERGENCE_TOLERANCE_S,
) -> float:
    """Temps prédit pour parcourir `chunks` à la puissance soutenable par le
    modèle CP (T-09), sans vent ni draft ("conditions standard").

    Boucle de convergence (point fixe) : la puissance soutenable
    `CP + W'/T` dépend du temps T du segment, qui dépend lui-même de la
    puissance qu'on peut tenir sur ce temps. On itère : estimer T, en
    déduire la puissance soutenable, simuler le tour à cette puissance
    constante pour obtenir un nouveau T, répéter jusqu'à ce que T se
    stabilise. Jamais de valeur retournée en silence si ça ne converge
    pas : ValueError explicite au-delà de `max_iterations`.
    """
    if not chunks:
        raise ValueError("aucun tronçon à simuler")

    total_length_m = sum(chunk.length_m for chunk in chunks)
    predicted_time_s = total_length_m / 8.0  # amorce grossière (~29 km/h) ; la boucle corrige

    for _ in range(max_iterations):
        sustainable_power_w = cp_watts + w_prime_joules / predicted_time_s
        new_time_s = _simulate_at_constant_power(
            chunks, sustainable_power_w, mass_kg, cda_m2, crr, air_density_kg_m3
        )
        if abs(new_time_s - predicted_time_s) < convergence_tolerance_s:
            return new_time_s
        predicted_time_s = new_time_s

    raise ValueError(
        f"pas de convergence après {max_iterations} itérations "
        f"(dernier temps prédit : {predicted_time_s:.1f}s, tolérance : {convergence_tolerance_s}s)"
    )
