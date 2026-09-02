"""Courbe de puissance : puissance maximale moyenne (MMP) par durée.

Fonctions pures — aucun I/O, aucune dépendance à ingest/storage. Le pont
entre le stream brut Strava (paires (t_s, watts) à résolution variable
à cause des pauses auto) et le calcul MMP se fait via
`resample_to_uniform_seconds`.
"""

from collections.abc import Iterable

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
