"""Optimisation du profil de puissance par programmation dynamique (T-26).

État = W' restant (discrétisé en grille), décision = puissance par
tronçon (choisie parmi une grille de candidats), objectif = temps total
minimal sous la contrainte W' >= 0 à tout instant. Fonction pure — aucun
I/O.

Optimisation clé : la durée d'un tronçon à une puissance donnée
(`cyclist_speed_from_power`, T-11) ne dépend PAS de l'état W' — elle est
donc précalculée une seule fois par (tronçon, puissance candidate), pas
recalculée pour chaque état de la grille W' (qui la réutilise). Sans ça,
le nombre d'appels au solveur de vitesse serait multiplié par le nombre
de niveaux W' de la grille, pour rien.
"""

from dataclasses import dataclass

import numpy as np

from segment_predictor.models.physics import cyclist_speed_from_power
from segment_predictor.models.segment import STANDARD_AIR_DENSITY_KG_M3, SegmentChunk
from segment_predictor.models.wbal import w_prime_balance_step

DEFAULT_N_W_BAL_LEVELS = 100
DEFAULT_N_POWER_CANDIDATES = 30
# Grille de puissances candidates, en fraction de CP. Assez bas pour
# "lever le pied" en descente/plat, assez haut pour permettre un vrai
# surplus (dépensant du W') sur les tronçons raides.
DEFAULT_MIN_POWER_FRACTION_OF_CP = 0.3
DEFAULT_MAX_POWER_FRACTION_OF_CP = 2.5


@dataclass(frozen=True)
class PacingResult:
    """`power_profile_w` : une puissance par tronçon, dans l'ordre de `chunks`."""

    total_time_s: float
    power_profile_w: list[float]


def optimize_pacing(
    chunks: list[SegmentChunk],
    cp_watts: float,
    w_prime_joules: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    air_density_kg_m3: float = STANDARD_AIR_DENSITY_KG_M3,
    n_w_bal_levels: int = DEFAULT_N_W_BAL_LEVELS,
    n_power_candidates: int = DEFAULT_N_POWER_CANDIDATES,
) -> PacingResult:
    """Programmation dynamique en avant : `dp_time[i, k]` = temps minimal
    pour atteindre le DÉBUT du tronçon `i` avec un W'bal correspondant au
    niveau `k` de la grille (grille linéaire de 0 à `w_prime_joules`,
    discrétisation approximative — plus `n_w_bal_levels` est grand, plus
    la solution est précise, au prix du temps de calcul). Le résultat
    final est le minimum sur tous les niveaux W' à la fin du dernier
    tronçon (peu importe ce qu'il reste dans le réservoir).

    Une puissance candidate qui ferait passer W'bal sous 0 sur un tronçon
    est simplement écartée pour cette transition — c'est la contrainte
    W' >= 0 du ticket, imposée directement dans la recherche plutôt que
    vérifiée après coup.
    """
    if not chunks:
        raise ValueError("chunks vide : aucun tronçon à optimiser")

    n_chunks = len(chunks)
    w_bal_grid = np.linspace(0.0, w_prime_joules, n_w_bal_levels)
    power_candidates_w = np.linspace(
        DEFAULT_MIN_POWER_FRACTION_OF_CP * cp_watts,
        DEFAULT_MAX_POWER_FRACTION_OF_CP * cp_watts,
        n_power_candidates,
    )

    # Précalcul : durée de CHAQUE tronçon à CHAQUE puissance candidate,
    # indépendant de l'état W' (voir docstring du module).
    chunk_duration_s = np.full((n_chunks, n_power_candidates), np.nan)
    for i, chunk in enumerate(chunks):
        for j, power_w in enumerate(power_candidates_w):
            try:
                speed_ms = cyclist_speed_from_power(
                    power_w, chunk.grade, 0.0, mass_kg, cda_m2, crr, air_density_kg_m3
                )
            except ValueError:
                continue  # pas de vitesse solution pour cette puissance sur ce tronçon
            chunk_duration_s[i, j] = chunk.length_m / speed_ms

    dp_time_s = np.full((n_chunks + 1, n_w_bal_levels), np.inf)
    dp_time_s[0, n_w_bal_levels - 1] = 0.0  # départ : réservoir plein, temps nul
    parent_state = np.full((n_chunks + 1, n_w_bal_levels), -1, dtype=int)
    parent_power_idx = np.full((n_chunks + 1, n_w_bal_levels), -1, dtype=int)

    for i in range(n_chunks):
        reachable_states = np.where(np.isfinite(dp_time_s[i]))[0]
        for k in reachable_states:
            current_w_bal = w_bal_grid[k]
            current_time_s = dp_time_s[i, k]
            for j, power_w in enumerate(power_candidates_w):
                dt_s = chunk_duration_s[i, j]
                if np.isnan(dt_s):
                    continue

                new_w_bal = w_prime_balance_step(
                    current_w_bal, power_w, cp_watts, w_prime_joules, dt_s
                )
                if new_w_bal < 0:
                    continue  # contrainte W' >= 0
                new_w_bal = min(new_w_bal, w_prime_joules)  # sécurité numérique (asymptote)

                new_k = round(float(new_w_bal / w_prime_joules) * (n_w_bal_levels - 1))
                new_time_s = current_time_s + dt_s
                if new_time_s < dp_time_s[i + 1, new_k]:
                    dp_time_s[i + 1, new_k] = new_time_s
                    parent_state[i + 1, new_k] = k
                    parent_power_idx[i + 1, new_k] = j

    final_states = np.where(np.isfinite(dp_time_s[n_chunks]))[0]
    if len(final_states) == 0:
        raise ValueError(
            "aucun profil de puissance ne permet de finir tous les tronçons sans épuiser W' "
            "— élargir la grille de puissances candidates (n_power_candidates) ou vérifier "
            "CP/W' par rapport à la difficulté du segment"
        )
    best_final_state = final_states[np.argmin(dp_time_s[n_chunks, final_states])]

    power_profile_w = []
    state = int(best_final_state)
    for i in range(n_chunks, 0, -1):
        power_idx = parent_power_idx[i, state]
        power_profile_w.append(float(power_candidates_w[power_idx]))
        state = int(parent_state[i, state])
    power_profile_w.reverse()

    return PacingResult(
        total_time_s=float(dp_time_s[n_chunks, best_final_state]),
        power_profile_w=power_profile_w,
    )
