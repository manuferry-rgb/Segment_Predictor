"""Indice de performance : ratio puissance réelle / puissance prédite par
le modèle CP, sur les efforts proches du maximum uniquement (T-23).

Fonctions pures — aucun I/O. `calibrate/form.py` fournit les données
réelles (MMP par activité, record glissant par durée) sur lesquelles ces
fonctions s'appliquent.
"""

from segment_predictor.models.power import CriticalPowerFit

# Seuil du filtre "effort maximal" (voir is_near_maximal_effort) : 95% du
# record glissant à date. Choisi plutôt que 100% pile pour ne pas exclure
# un effort à bloc légèrement moins bon que LE record absolu de l'athlète
# à cette durée — un athlète n'égale pas son propre record à chaque sortie
# à fond, ce serait un filtre bien trop strict pour avoir assez de points.
DEFAULT_MAXIMAL_EFFORT_THRESHOLD = 0.95


def performance_index(actual_power_w: float, cp_fit: CriticalPowerFit, duration_s: float) -> float:
    """actual_power_w / (CP + W'/duration_s). > 1 : au-dessus de ce que le
    modèle prédit comme soutenable à cette durée (bonne forme, ou modèle
    qui sous-estime) ; < 1 : en dessous (fatigue, mauvais jour, ou modèle
    qui surestime) — l'indice ne dit pas lequel, juste la direction.

    Valide seulement sur `cp_fit.duration_range_s` — même raison qu'ailleurs
    (T-09/T-17) : hors de cette plage, `CP + W'/T` diverge ou n'est plus
    fiable, comparer une puissance réelle à une prédiction déjà fausse ne
    dirait rien sur la vraie forme du jour.
    """
    range_min_s, range_max_s = cp_fit.duration_range_s
    if not range_min_s <= duration_s <= range_max_s:
        raise ValueError(
            f"duration_s={duration_s} hors de la plage de validité du modèle CP "
            f"{cp_fit.duration_range_s} — l'indice n'aurait pas de sens"
        )
    predicted_power_w = cp_fit.cp_watts + cp_fit.w_prime_joules / duration_s
    return actual_power_w / predicted_power_w


def is_near_maximal_effort(
    mmp_w: float,
    best_so_far_mmp_w: float,
    threshold: float = DEFAULT_MAXIMAL_EFFORT_THRESHOLD,
) -> bool:
    """Un effort est jugé "proche du maximum" (T-23) s'il atteint au moins
    `threshold` fois `best_so_far_mmp_w` — le record GLISSANT à cette
    date, pas le record absolu toutes dates confondues.

    La distinction compte : comparer au record absolu ferait
    disparaître rétroactivement de vrais efforts à bloc anciens dès
    qu'un record plus récent les dépasse, biaisant la série temporelle
    de l'indice vers les seules activités récentes (les plus proches du
    record du jour). Le record glissant (le meilleur jamais vu JUSQUE-LÀ,
    calculé par `calibrate/form.py`) évite ce biais.
    """
    if best_so_far_mmp_w <= 0:
        raise ValueError(f"best_so_far_mmp_w doit être positif, reçu {best_so_far_mmp_w}")
    return mmp_w >= threshold * best_so_far_mmp_w
