"""Courbe de puissance : puissance maximale moyenne (MMP) par durée,
et modèle Critical Power à 2 paramètres ajusté dessus.

Fonctions pures — aucun I/O, aucune dépendance à ingest/storage. Le pont
entre le stream brut Strava (paires (t_s, watts) à résolution variable
à cause des pauses auto) et le calcul MMP se fait via
`resample_to_uniform_seconds`.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def resample_to_uniform_seconds(t_s: np.ndarray, watts: np.ndarray) -> np.ndarray:
    """Stream (t_s, watts) à résolution variable -> grille 1 Hz continue, trous en NaN.

    Une pause auto de 150 s dans l'enregistrement Strava ne devient PAS
    150 échantillons à 0 W (ce serait une puissance inventée) : elle
    devient 150 NaN, que `mean_maximal_power` exclut ensuite du calcul.
    """
    t_s = np.asarray(t_s)
    watts = np.asarray(watts, dtype=float)

    if len(t_s) == 0:
        raise ValueError("stream vide : rien à rééchantillonner")
    if len(t_s) != len(watts):
        raise ValueError(f"t_s et watts n'ont pas la même longueur ({len(t_s)} vs {len(watts)})")
    if np.any(np.diff(t_s) <= 0):
        raise ValueError(
            "t_s doit être strictement croissant (pas de doublon, pas de retour en arrière)"
        )

    start = t_s[0]
    span = int(t_s[-1] - start + 1)
    grid = np.full(span, np.nan, dtype=float)
    grid[(t_s - start).astype(np.int64)] = watts
    return grid


def mean_maximal_power(watts: np.ndarray, duration_s: int) -> float:
    """Puissance moyenne maximale sur une fenêtre continue de `duration_s` secondes.

    `watts` : grille 1 Hz (voir resample_to_uniform_seconds), NaN = trou.
    Une fenêtre contenant au moins un NaN est exclue du calcul plutôt
    qu'approximée. Renvoie NaN si aucune fenêtre valide n'existe
    (activité trop courte, ou trop trouée pour cette durée).

    Implémentation par somme cumulée : la somme de toute fenêtre
    [i, i+d) vaut cumsum[i+d] - cumsum[i], calculable en O(1). Recalculer
    chaque fenêtre depuis zéro (fenêtre glissante "naïve") coûterait
    O(n·d) pour cette seule durée ; ici c'est O(n), et entièrement
    vectorisé (une soustraction de deux tranches numpy, pas de boucle
    Python).

    Attention en lisant une courbe MMP(d) pour plusieurs d : elle n'est
    PAS garantie strictement décroissante quand d augmente. C'est vrai
    seulement quand une durée divise l'autre (partition en blocs égaux) ;
    en général, une petite irrégularité entre deux durées voisines est
    un résultat exact, pas un bug (voir test_mmp_curve_is_not_guaranteed_
    monotonic_across_arbitrary_durations).
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s doit être positif, reçu {duration_s}")

    watts = np.asarray(watts, dtype=float)
    n = len(watts)
    if duration_s > n:
        return float("nan")

    is_valid = ~np.isnan(watts)
    filled = np.where(is_valid, watts, 0.0)

    # Un 0 en tête simplifie l'indexation : sum(watts[i:i+d]) = cumsum[i+d] - cumsum[i].
    cumsum = np.concatenate(([0.0], np.cumsum(filled)))
    valid_cumsum = np.concatenate(([0], np.cumsum(is_valid.astype(np.int64))))

    window_sums = cumsum[duration_s:] - cumsum[:-duration_s]
    window_valid_counts = valid_cumsum[duration_s:] - valid_cumsum[:-duration_s]

    fully_valid = window_valid_counts == duration_s
    if not np.any(fully_valid):
        return float("nan")

    return float(np.max(window_sums[fully_valid] / duration_s))


def mean_maximal_power_curve(watts: np.ndarray, durations_s: Iterable[int]) -> dict[int, float]:
    """MMP pour plusieurs durées. Chaque durée reste O(n) (voir mean_maximal_power),
    donc la courbe complète coûte O(n · nombre de durées) — pas de raccourci
    supplémentaire au-delà de ça, chaque durée a sa propre fenêtre optimale.
    """
    return {duration_s: mean_maximal_power(watts, duration_s) for duration_s in durations_s}


# Fenêtre standard de l'algorithme de Coggan pour la puissance normalisée —
# pas un paramètre à ajuster au cas par cas, la définition elle-même fixe 30s.
NORMALIZED_POWER_WINDOW_S = 30


def normalized_power(watts: np.ndarray) -> float:
    """Puissance normalisée (NP, algorithme de Coggan) : moyenne glissante
    30s, chaque valeur élevée à la puissance 4, moyenne de ces valeurs,
    racine 4e. Le exposant 4 pénalise fortement les pics par rapport à une
    moyenne simple — deux sorties de même puissance moyenne mais l'une
    "en dents de scie" (sprints/récup) coûte physiologiquement plus cher
    que l'autre à rythme constant, et NP le capture alors qu'une moyenne
    simple ne le distingue pas.

    `watts` : grille 1 Hz (voir resample_to_uniform_seconds), NaN = trou.
    Une fenêtre de 30s touchant un NaN est exclue, comme pour
    mean_maximal_power. Contrairement à mean_maximal_power (qui renvoie
    NaN si rien n'est valide — pensé pour être agrégé sur plein
    d'activités où un trou isolé est tolérable), ici une ValueError est
    levée explicitement : NP alimente directement training_stress_score
    (T-21) par activité, un NaN silencieux corromprait toute la
    récursion de Banister en aval sans qu'on s'en aperçoive.
    """
    watts = np.asarray(watts, dtype=float)
    window_s = NORMALIZED_POWER_WINDOW_S
    if len(watts) < window_s:
        raise ValueError(
            f"stream de {len(watts)}s trop court pour une fenêtre de {window_s}s (NP non définie)"
        )

    is_valid = ~np.isnan(watts)
    filled = np.where(is_valid, watts, 0.0)

    cumsum = np.concatenate(([0.0], np.cumsum(filled)))
    valid_cumsum = np.concatenate(([0], np.cumsum(is_valid.astype(np.int64))))

    window_sums = cumsum[window_s:] - cumsum[:-window_s]
    window_valid_counts = valid_cumsum[window_s:] - valid_cumsum[:-window_s]
    fully_valid = window_valid_counts == window_s

    if not np.any(fully_valid):
        raise ValueError(f"aucune fenêtre de {window_s}s entièrement valide (stream trop troué)")

    rolling_avg_w = window_sums[fully_valid] / window_s
    return float(np.mean(rolling_avg_w**4) ** 0.25)


def training_stress_score(
    duration_s: float, normalized_power_w: float, threshold_power_w: float
) -> float:
    """TSS (Training Stress Score, définition de Coggan) : charge d'une
    séance, calibrée pour qu'1h pile à la puissance seuil vaille 100.

    `threshold_power_w` : ici le CP calibré (T-16/T-17), utilisé comme
    proxy de la FTP — CP et FTP ne sont pas rigoureusement identiques
    (CP est un seuil théorique du modèle à 2 paramètres, FTP un protocole
    de test standardisé), mais assez proches en pratique pour cet usage,
    et on n'a pas d'historique de tests FTP datés. Approximation
    documentée, pas cachée.

    IF (Intensity Factor) = normalized_power_w / threshold_power_w :
    intermédiaire nommé pour rester lisible, pas juste pour factoriser du
    calcul.
    """
    if threshold_power_w <= 0:
        raise ValueError(f"threshold_power_w doit être positif, reçu {threshold_power_w}")

    intensity_factor = normalized_power_w / threshold_power_w
    return duration_s * normalized_power_w * intensity_factor / (threshold_power_w * 3600) * 100


# Plage par défaut du modèle CP à 2 paramètres : en dessous, la puissance est
# dominée par des facteurs neuromusculaires/anaérobies (le modèle diverge vers
# l'infini quand t->0, ce qui est physiologiquement absurde) ; au-dessus, CP
# n'est plus vraiment constante (fatigue, déplétion glycogénique,
# thermorégulation font décliner la puissance soutenable plus vite que ne le
# prédit l'hyperbole). 3-20 min est la fenêtre usuelle en physiologie de
# l'exercice depuis Monod & Scherrer (1965).
DEFAULT_CP_DURATION_RANGE_S = (180, 1200)


@dataclass(frozen=True)
class CriticalPowerFit:
    """Résultat d'un ajustement du modèle P(t) = CP + W'/t.

    `r_squared` est calculé dans l'espace P(t) d'origine (watts), pas
    dans l'espace travail-temps utilisé pour le fit lui-même — sinon il
    mesurerait la qualité de l'ajustement sur une grandeur dérivée (le
    travail), pas sur ce qu'on veut vraiment évaluer (la puissance).

    `cp_watts_std`/`w_prime_joules_std` (T-28) : écarts-types issus de la
    matrice de covariance de la même régression linéaire (voir
    fit_critical_power) — pas une nouvelle méthode statistique, juste la
    covariance qu'un fit aux moindres carrés produit déjà. Par défaut à
    0.0 (pas d'incertitude connue) pour rester compatible avec les tests
    existants qui construisent un CriticalPowerFit à la main sans s'en
    soucier.
    """

    cp_watts: float
    w_prime_joules: float
    r_squared: float
    n_points: int
    duration_range_s: tuple[int, int]
    cp_watts_std: float = 0.0
    w_prime_joules_std: float = 0.0


def fit_critical_power(
    durations_s: np.ndarray,
    mmp_watts: np.ndarray,
    duration_range_s: tuple[int, int] = DEFAULT_CP_DURATION_RANGE_S,
) -> CriticalPowerFit:
    """Ajuste CP (W) et W' (J) sur les points (durée, MMP) dans `duration_range_s`.

    Linéarisation travail-temps (Monod & Scherrer) : P = CP + W'/t se
    réécrit W_total = P·t = CP·t + W', linéaire en t. Régresser sur le
    travail total plutôt que directement sur 1/t évite de sur-pondérer
    les durées courtes (1/t explose quand t est petit) — c'est la
    méthode historique de calcul de CP/W', fermée (pas d'itération, pas
    de sensibilité à une valeur initiale).

    Les points hors `duration_range_s`, ou avec un MMP manquant (NaN,
    ex. activité jamais assez longue), sont exclus plutôt qu'estimés.
    """
    durations_s = np.asarray(durations_s, dtype=float)
    mmp_watts = np.asarray(mmp_watts, dtype=float)
    if len(durations_s) != len(mmp_watts):
        raise ValueError(
            f"durations_s et mmp_watts n'ont pas la même longueur "
            f"({len(durations_s)} vs {len(mmp_watts)})"
        )

    range_min, range_max = duration_range_s
    if range_min <= 0 or range_max <= range_min:
        raise ValueError(f"duration_range_s invalide : {duration_range_s}")

    in_range = (durations_s >= range_min) & (durations_s <= range_max)
    valid = in_range & ~np.isnan(mmp_watts)
    t = durations_s[valid]
    p = mmp_watts[valid]

    if len(t) < 2:
        raise ValueError(
            f"seulement {len(t)} point(s) valide(s) dans [{range_min}, {range_max}]s : "
            "il en faut au moins 2 pour ajuster 2 paramètres (CP et W')"
        )

    total_work = p * t  # travail total (J) fourni pendant t secondes à puissance p
    # cov=True (T-28) : matrice de covariance de (CP, W') gratuite depuis
    # cette même régression — pas une méthode séparée, np.polyfit la
    # calcule déjà en interne pour son propre usage interne (résidus).
    (cp, w_prime), cov = np.polyfit(t, total_work, 1, cov=True)  # pente = CP, ordonnée = W'
    cp_watts_std = float(np.sqrt(cov[0, 0]))
    w_prime_joules_std = float(np.sqrt(cov[1, 1]))

    predicted_p = cp + w_prime / t
    residuals = p - predicted_p
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((p - np.mean(p)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return CriticalPowerFit(
        cp_watts=float(cp),
        w_prime_joules=float(w_prime),
        r_squared=r_squared,
        n_points=len(t),
        duration_range_s=(int(range_min), int(range_max)),
        cp_watts_std=cp_watts_std,
        w_prime_joules_std=w_prime_joules_std,
    )


def sustainable_power_w(cp_watts: float, w_prime_joules: float, duration_s: float) -> float:
    """Puissance soutenable par le modèle CP sur `duration_s` : P = CP + W'/t.

    Même formule que la boucle de convergence de `models.segment.
    simulate_segment_time` (T-13/T-27), extraite ici en fonction pure
    plutôt que dupliquée : sert à AFFICHER la puissance requise pour un
    temps prédit donné (T-31), sans changer la signature de
    `simulate_segment_time` (déjà utilisée telle quelle par la
    calibration, le backtest et la propagation d'incertitude).
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s doit être positif, reçu {duration_s}")
    return cp_watts + w_prime_joules / duration_s
