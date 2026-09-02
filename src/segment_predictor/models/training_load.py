"""Modèle de charge d'entraînement de Banister : CTL, ATL, TSB (T-21).

Fonction pure — aucun I/O. Prend une série de TSS quotidiens (un par
jour civil, 0.0 les jours sans sortie : la forme se décharge quand même,
ce n'est pas un jour "neutre" à ignorer) et produit, par récursion
exponentielle :

- CTL (Chronic Training Load, "forme" long terme) : moyenne mobile
  exponentielle du TSS, constante de temps ~42 jours.
- ATL (Acute Training Load, "fatigue" court terme) : même récursion,
  constante de temps ~7 jours — réagit beaucoup plus vite que CTL à une
  séance ponctuelle.
- TSB (Training Stress Balance, "forme du jour") = CTL - ATL, calculée
  avec les valeurs de la VEILLE : le TSB du jour J reflète la forme avec
  laquelle on aborde la séance de J, pas l'effet de cette séance
  elle-même (convention standard, ex. TrainingPeaks).
"""

from dataclasses import dataclass

# Constantes de temps standard (Banister 1975, reprises telles quelles par
# la plupart des outils type TrainingPeaks) : ~6 semaines pour la forme
# long terme, ~1 semaine pour la fatigue aiguë. Pas des valeurs qu'on
# calibre sur nos propres données ici — elles définissent CTL/ATL, ce
# n'en sont pas des paramètres libres.
DEFAULT_CTL_TIME_CONSTANT_DAYS = 42
DEFAULT_ATL_TIME_CONSTANT_DAYS = 7


@dataclass(frozen=True)
class TrainingLoadPoint:
    """`ctl`/`atl` : valeurs APRÈS absorption du TSS du jour. `tsb` : calculé
    AVANT, à partir des ctl/atl de la veille (voir docstring du module)."""

    ctl: float
    atl: float
    tsb: float


def compute_training_load(
    daily_tss: list[float],
    ctl_time_constant_days: float = DEFAULT_CTL_TIME_CONSTANT_DAYS,
    atl_time_constant_days: float = DEFAULT_ATL_TIME_CONSTANT_DAYS,
    initial_ctl: float = 0.0,
    initial_atl: float = 0.0,
) -> list[TrainingLoadPoint]:
    """Un point par entrée de `daily_tss`, dans l'ordre chronologique
    (l'appelant garantit un TSS par jour civil consécutif, y compris les
    jours à 0.0 — cette fonction ne connaît pas les dates, juste la
    séquence).

    `initial_ctl`/`initial_atl` : forme au tout début de la série. 0.0 par
    défaut (athlète "reposé" avant le premier jour connu) ; à renseigner
    explicitement si l'historique réel commence en cours de saison.
    """
    if not daily_tss:
        raise ValueError("daily_tss vide : rien à calculer")
    if ctl_time_constant_days <= 0:
        raise ValueError(f"ctl_time_constant_days doit être positif, reçu {ctl_time_constant_days}")
    if atl_time_constant_days <= 0:
        raise ValueError(f"atl_time_constant_days doit être positif, reçu {atl_time_constant_days}")

    points = []
    ctl, atl = initial_ctl, initial_atl
    for tss in daily_tss:
        tsb = ctl - atl  # forme héritée de la veille, avant d'absorber le TSS du jour
        ctl = ctl + (tss - ctl) / ctl_time_constant_days
        atl = atl + (tss - atl) / atl_time_constant_days
        points.append(TrainingLoadPoint(ctl=ctl, atl=atl, tsb=tsb))
    return points
