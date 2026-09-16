"""Découpage d'un profil (distance, altitude, lat/lng) en tronçons ~50m,
et simulation du temps pour parcourir ces tronçons (T-13).

Fonctions pures — aucun I/O. Trois étapes séparées et composables :
`smooth_altitude` (débruite l'altitude GPS) -> `chunk_segment` (découpe
en tronçons, pente + cap par tronçon) -> `simulate_segment_time` (temps
prédit, à la puissance soutenable par le modèle CP).
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .physics import air_density, cyclist_speed_from_power, effective_headwind_speed_ms
from .power import interpolate_mmp_curve, sustainable_power_w

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


# Rayon terrestre moyen (m), sphère IUGG — approximation standard pour une
# distance grand cercle à l'échelle d'un segment (quelques km) : l'écart
# avec un modèle ellipsoïdal (WGS84) est de l'ordre de 0.3%, largement
# sous le bruit d'arrondi du polyline lui-même (précision 1e-5°, T-32).
_EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_m(
    lat1_deg: float, lng1_deg: float, lat2_deg: float, lng2_deg: float
) -> float:
    """Distance grand cercle (m) entre deux points GPS, formule de haversine.

    Publique (T-32) comme bearing_rad : réutilisée par
    segment_chunks_from_polyline pour la longueur de chaque tronçon,
    faute de `distance_m` fourni (contrairement à chunk_segment, qui le
    reçoit d'un vrai stream Strava plutôt que de le recalculer).
    """
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    delta_lat = math.radians(lat2_deg - lat1_deg)
    delta_lng = math.radians(lng2_deg - lng1_deg)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lng / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_rad(lat1_deg: float, lng1_deg: float, lat2_deg: float, lng2_deg: float) -> float:
    """Cap initial (relèvement) du point 1 vers le point 2 : 0 = nord, croît
    vers l'est. Publique (pas seulement interne à chunk_segment) : réutilisée
    pour donner un cap global à un segment entier depuis ses start/end
    latlng (T-27, storage/segments.py), là où l'approximation "un seul
    tronçon" n'a pas de heading calculé chunk par chunk.
    """
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
                heading_rad=bearing_rad(
                    boundary_lats[i], boundary_lngs[i], boundary_lats[i + 1], boundary_lngs[i + 1]
                ),
            )
        )
    return chunks


def segment_chunks_from_polyline(
    points: list[tuple[float, float]], average_grade: float
) -> list[SegmentChunk]:
    """Un SegmentChunk par paire de points consécutifs d'un polyline décodé
    (T-32, `models.polyline.decode_polyline`) — un cap RÉEL par tronçon
    (bearing_rad) au lieu de l'unique cap moyen start->end de
    `heading_rad` (T-27). C'est tout l'objet du ticket : un vent projeté
    correctement le long d'un segment qui tourne ou boucle, pas un seul
    vent de face/dos appliqué à toute la distance (vérifié sur HBFH, un
    segment en boucle, voir test_segment_chunks_from_polyline_matches_
    real_strava_segment : le cap y varie sur presque tout le cercle).

    Limite ASSUMÉE, pas cachée : `average_grade` (pente moyenne Strava du
    segment ENTIER) est appliquée telle quelle à CHAQUE tronçon — aucune
    source d'altitude par tronçon n'est disponible depuis un polyline
    (lat/lng seulement, contrairement à chunk_segment qui reçoit un vrai
    stream d'altitude). Seul le cap varie réellement ici, pas la pente
    (dénivelé hors périmètre de ce ticket, voir ROADMAP.md T-32).

    Deux points consécutifs identiques (arrondi à 1e-5° du polyline, le
    tracé peut repasser exactement par un point déjà visité) sont
    ignorés plutôt que de produire un tronçon de longueur nulle et un
    cap indéfini (atan2(0, 0)).
    """
    if len(points) < 2:
        raise ValueError(f"il faut au moins 2 points pour découper un tronçon, reçu {len(points)}")

    chunks = []
    cumulative_distance_m = 0.0
    for (lat1, lng1), (lat2, lng2) in itertools.pairwise(points):
        length_m = haversine_distance_m(lat1, lng1, lat2, lng2)
        if length_m <= 0:
            continue  # points consécutifs identiques : rien à découper ici
        chunks.append(
            SegmentChunk(
                start_distance_m=cumulative_distance_m,
                length_m=length_m,
                grade=average_grade,
                heading_rad=bearing_rad(lat1, lng1, lat2, lng2),
            )
        )
        cumulative_distance_m += length_m

    if not chunks:
        raise ValueError("tous les points du polyline sont identiques : aucun tronçon à produire")

    return chunks


def segment_chunks_from_profile(
    distance_m: np.ndarray,
    altitude_m: np.ndarray,
    lat: np.ndarray,
    lng: np.ndarray,
    chunk_length_m: float = 50.0,
) -> list[SegmentChunk]:
    """Tronçons ~50m avec une VRAIE pente par tronçon (T-51c), depuis le
    profil OFFICIEL d'un segment (`segment_streams`, T-51a/T-51b :
    distance/altitude/lat/lng le long du tracé) — remplace
    `segment_chunks_from_polyline` (T-32) pour les calculs où une pente
    moyenne unique appliquée à tout le segment n'est pas assez fidèle
    (trouvé en testant KomInfo.power_w en vrai, T-51 : un segment à pente
    moyenne 4.8% mais probablement très irrégulière donnait une
    puissance requise fausse dans les deux sens).

    Pas une nouvelle mécanique : enchaîne simplement `smooth_altitude`
    (lisse le bruit GPS avant de dériver une pente, T-12) puis
    `chunk_segment` (T-12, déjà utilisée pour chunker le stream d'UNE
    activité en calibration) — cette fonction-ci n'existait pas encore
    pour le profil d'un SEGMENT lui-même.
    """
    smoothed_altitude_m = smooth_altitude(distance_m, altitude_m)
    return chunk_segment(distance_m, smoothed_altitude_m, lat, lng, chunk_length_m)


def average_tailwind_speed_ms(
    chunks: list[SegmentChunk], wind_speed_ms: float, wind_direction_rad: float
) -> float:
    """Vitesse de vent favorable MOYENNE sur le segment (m/s), pondérée
    par la longueur de chaque tronçon — filtre géométrique "quels
    segments tenter aujourd'hui" (T-33/T-34), indépendant de toute
    calibration (CP, CdA, Crr) : seulement le tracé (T-32) contre un
    vent, pas une prédiction de temps.

    Positive = vent de dos en moyenne (aide), négative = vent de face en
    moyenne (freine) — signe OPPOSÉ à `effective_headwind_speed_ms`
    (positive = face là-bas) : plus intuitif à lire "vent favorable de
    +8 km/h" que "vent de face de -8 km/h" dans l'UI.

    Une VRAIE grandeur physique (m/s), pas une fraction abstraite de
    distance (ancienne version T-33, `tailwind_fraction`) : un vent de
    dos à 2 m/s sur la moitié du trajet et un vent de dos à 15 m/s sur
    la même moitié comptaient PAREIL dans une fraction (0.5 dans les
    deux cas), alors qu'ils n'aident pas du tout autant — pondérer par
    la vitesse en plus de la longueur corrige ça (voir test_average_
    tailwind_speed_is_weighted_by_chunk_length_not_chunk_count).
    """
    if not chunks:
        raise ValueError("chunks vide : rien à évaluer")

    total_length_m = sum(chunk.length_m for chunk in chunks)
    weighted_headwind_ms = sum(
        chunk.length_m
        * effective_headwind_speed_ms(wind_speed_ms, wind_direction_rad, chunk.heading_rad)
        for chunk in chunks
    )
    return -weighted_headwind_ms / total_length_m


def average_wind_alignment_pct(chunks: list[SegmentChunk], wind_direction_rad: float) -> float:
    """Alignement PUR entre le tracé et la direction du vent, pondéré par
    la longueur de chaque tronçon (T-43) — même pondération que
    `average_tailwind_speed_ms` (T-34), mais sans `wind_speed_ms` : le
    résultat ne dépend que de la GÉOMÉTRIE (tracé + direction du vent),
    pas de sa force. Répond à "quelle fraction d'un vent, si j'en avais,
    serait dans le bon sens sur ce tracé ?", une question différente de
    "quelle vitesse de vent de dos j'ai aujourd'hui" (déjà
    `average_tailwind_speed_ms`) — les deux se complètent, aucune ne
    remplace l'autre (voir T-34 : un % seul avait déjà été abandonné une
    fois comme critère de CLASSEMENT, car il ne distingue pas 2 km/h de
    20 km/h de vent bien aligné — ce %-ci sert à AFFICHER l'alignement,
    pas à classer).

    +100 = vent de dos pur sur tout le tracé, -100 = vent de face pur,
    0 = vent de travers pur (en moyenne pondérée).
    """
    if not chunks:
        raise ValueError("chunks vide : rien à évaluer")

    total_length_m = sum(chunk.length_m for chunk in chunks)
    # -cos(...) : même signe que average_tailwind_speed_ms (positif = vent
    # de dos), en tirant `wind_speed_ms` de effective_headwind_speed_ms
    # (fixé à 1.0, pure projection angulaire) plutôt qu'en dupliquant le
    # cos() séparément.
    weighted_alignment = sum(
        chunk.length_m * -effective_headwind_speed_ms(1.0, wind_direction_rad, chunk.heading_rad)
        for chunk in chunks
    )
    return (weighted_alignment / total_length_m) * 100.0


def _simulate_at_constant_power(
    chunks: list[SegmentChunk],
    power_w: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float,
    wind_speed_ms: float,
    wind_direction_rad: float,
) -> float:
    """Temps total pour parcourir tous les tronçons à une puissance CONSTANTE.

    Le vent (T-15, `effective_headwind_speed_ms`) est absolu — un seul
    (wind_speed_ms, wind_direction_rad) pour tout le segment, réaliste sur
    la portée d'un segment (quelques minutes) — mais sa PROJECTION est
    recalculée à chaque tronçon via son propre `heading_rad` : un même
    vent absolu peut être de face sur un tronçon et de dos sur un autre
    si le tracé change de direction (ex. un aller-retour). Pas de draft à
    ce stade ("conditions standard" pour le draft, ajouté ailleurs T-19).
    """
    total_time_s = 0.0
    for chunk in chunks:
        headwind_speed_ms = effective_headwind_speed_ms(
            wind_speed_ms, wind_direction_rad, chunk.heading_rad
        )
        speed_ms = cyclist_speed_from_power(
            power_w, chunk.grade, headwind_speed_ms, mass_kg, cda_m2, crr, air_density_kg_m3
        )
        total_time_s += chunk.length_m / speed_ms
    return total_time_s


DEFAULT_POWER_BOUNDS_W = (1.0, 3000.0)


def power_required_for_target_time(
    chunks: list[SegmentChunk],
    target_time_s: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float = STANDARD_AIR_DENSITY_KG_M3,
    wind_speed_ms: float = 0.0,
    wind_direction_rad: float = 0.0,
    power_bounds_w: tuple[float, float] = DEFAULT_POWER_BOUNDS_W,
) -> float:
    """Puissance CONSTANTE qui, appliquée sur `chunks` (profil RÉEL, pente
    variable par tronçon, T-51c), donne exactement `target_time_s` au
    total (T-51d).

    Question inverse de `simulate_segment_time`/`simulate_segment_time_
    from_mmp_curve` : celles-ci partent d'un modèle de puissance
    soutenable pour EN DÉDUIRE un temps ; ici le temps est déjà CONNU
    (typiquement le temps du KOM, T-51) et on cherche la puissance qui
    l'explique — pas de boucle de convergence par point fixe nécessaire
    (`_simulate_at_constant_power` est déjà une fonction directe de la
    puissance), juste une recherche de racine par la méthode de Brent
    (même outil que `cyclist_speed_from_power`, physics.py) sur
    `_simulate_at_constant_power(power_w, ...) - target_time_s`,
    strictement décroissante en `power_w` (plus de puissance -> toujours
    plus vite, jamais l'inverse) : une seule racine dans `power_bounds_w`.
    """
    if target_time_s <= 0:
        raise ValueError(f"target_time_s doit être positif, reçu {target_time_s}")

    def residual(power_w: float) -> float:
        return (
            _simulate_at_constant_power(
                chunks,
                power_w,
                mass_kg,
                cda_m2,
                crr,
                air_density_kg_m3,
                wind_speed_ms,
                wind_direction_rad,
            )
            - target_time_s
        )

    lo, hi = power_bounds_w
    try:
        return brentq(residual, lo, hi)
    except ValueError as error:
        raise ValueError(
            f"pas de puissance solution dans {power_bounds_w}W pour tenir {target_time_s:.0f}s "
            f"sur ce profil : {error}"
        ) from error


def _simulate_time_with_power_curve(
    chunks: list[SegmentChunk],
    power_curve_fn: Callable[[float], float],
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float,
    wind_speed_ms: float,
    wind_direction_rad: float,
    max_iterations: int,
    convergence_tolerance_s: float,
    initial_guess_s: float | None,
) -> float:
    """Boucle de convergence par point fixe partagée par `simulate_segment_time`
    (T-13, courbe CP+W') et `simulate_segment_time_from_mmp_curve` (T-42,
    courbe MMP réellement mesurée) — seule la source de puissance en
    fonction de la durée change (`power_curve_fn`), la mécanique de
    convergence est identique : la puissance soutenable sur T dépend de T,
    qui dépend lui-même de cette puissance. On itère jusqu'à stabilisation.
    Jamais de valeur retournée en silence si ça ne converge pas : ValueError
    explicite au-delà de `max_iterations` — et `power_curve_fn` peut lever
    sa propre ValueError à tout moment (ex. `interpolate_mmp_curve` hors de
    sa plage mesurée), qui remonte telle quelle.

    `initial_guess_s` : amorce de la boucle. `None` retombe sur l'amorce
    grossière historique (longueur/8, ~29 km/h) ; un appelant qui dispose
    déjà d'une meilleure estimation (ex. le temps prédit par le modèle
    CP+W' pour ce même segment) peut la fournir pour réduire le risque de
    sortir de la plage mesurée d'une courbe MMP réelle avant d'avoir pu
    converger — le résultat au point fixe ne dépend pas de l'amorce,
    seule la trajectoire pour l'atteindre en dépend.
    """
    if not chunks:
        raise ValueError("aucun tronçon à simuler")

    if initial_guess_s is None:
        total_length_m = sum(chunk.length_m for chunk in chunks)
        initial_guess_s = total_length_m / 8.0  # amorce grossière (~29 km/h) ; la boucle corrige
    predicted_time_s = initial_guess_s

    for _ in range(max_iterations):
        power_w = power_curve_fn(predicted_time_s)
        new_time_s = _simulate_at_constant_power(
            chunks,
            power_w,
            mass_kg,
            cda_m2,
            crr,
            air_density_kg_m3,
            wind_speed_ms,
            wind_direction_rad,
        )
        if abs(new_time_s - predicted_time_s) < convergence_tolerance_s:
            return new_time_s
        predicted_time_s = new_time_s

    raise ValueError(
        f"pas de convergence après {max_iterations} itérations "
        f"(dernier temps prédit : {predicted_time_s:.1f}s, tolérance : {convergence_tolerance_s}s)"
    )


def simulate_segment_time(
    chunks: list[SegmentChunk],
    cp_watts: float,
    w_prime_joules: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float = STANDARD_AIR_DENSITY_KG_M3,
    wind_speed_ms: float = 0.0,
    wind_direction_rad: float = 0.0,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    convergence_tolerance_s: float = DEFAULT_CONVERGENCE_TOLERANCE_S,
) -> float:
    """Temps prédit pour parcourir `chunks` à la puissance soutenable par le
    modèle CP (T-09), sans draft ("conditions standard" pour le draft
    uniquement — le vent est pris en compte si fourni, T-27).

    `wind_speed_ms`/`wind_direction_rad` : vent absolu unique pour tout le
    segment (voir `_simulate_at_constant_power`) — `wind_direction_rad`
    dans la convention météo (direction D'OÙ VIENT le vent, T-15), pas à
    confondre avec `heading_rad` des tronçons (direction OÙ ON VA).
    Défaut (0.0, 0.0) : aucun effet, comportement identique à avant T-27.

    Boucle de convergence : voir `_simulate_time_with_power_curve` (T-42,
    factorisée là plutôt que dupliquée entre cette fonction et
    `simulate_segment_time_from_mmp_curve`). Ici la source de puissance est
    le modèle CP+W' (`sustainable_power_w`, models.power).
    """
    return _simulate_time_with_power_curve(
        chunks,
        lambda predicted_time_s: sustainable_power_w(cp_watts, w_prime_joules, predicted_time_s),
        mass_kg,
        cda_m2,
        crr,
        air_density_kg_m3,
        wind_speed_ms,
        wind_direction_rad,
        max_iterations,
        convergence_tolerance_s,
        initial_guess_s=None,
    )


def simulate_segment_time_from_mmp_curve(
    chunks: list[SegmentChunk],
    mmp_curve: dict[int, float],
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float = STANDARD_AIR_DENSITY_KG_M3,
    wind_speed_ms: float = 0.0,
    wind_direction_rad: float = 0.0,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    convergence_tolerance_s: float = DEFAULT_CONVERGENCE_TOLERANCE_S,
    initial_guess_s: float | None = None,
) -> float:
    """Variante de `simulate_segment_time` (T-13) qui lit la puissance
    soutenable directement sur la courbe MMP RÉELLEMENT MESURÉE (T-42a,
    `interpolate_mmp_curve`) plutôt que sur le modèle CP+W' lissé —
    répond à "si je donnais vraiment ma meilleure puissance déjà atteinte
    pour cette durée, avec le vent de ce créneau, quel temps ça donnerait ?"
    Ajoutée à côté de `simulate_segment_time`, pas à sa place (décision
    explicite) : le classement des créneaux continue de reposer sur le
    modèle CP+W', cette fonction n'alimente qu'un chiffre de comparaison.

    Peut lever la ValueError d'`interpolate_mmp_curve` si le temps converge
    hors de la plage mesurée par `mmp_curve` (pas d'extrapolation, T-42a) —
    contrairement à `simulate_segment_time`, qui répond toujours (le modèle
    CP+W' extrapole, moins fidèlement mais sans jamais lever pour ça).
    `initial_guess_s` : voir `_simulate_time_with_power_curve` — utile ici
    pour amorcer avec le temps déjà prédit par le modèle CP+W' plutôt que
    l'amorce générique, et réduire le risque de sortir de la plage mesurée
    avant d'avoir convergé.
    """
    return _simulate_time_with_power_curve(
        chunks,
        lambda predicted_time_s: interpolate_mmp_curve(mmp_curve, predicted_time_s),
        mass_kg,
        cda_m2,
        crr,
        air_density_kg_m3,
        wind_speed_ms,
        wind_direction_rad,
        max_iterations,
        convergence_tolerance_s,
        initial_guess_s,
    )
