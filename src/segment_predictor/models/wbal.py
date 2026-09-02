"""Bilan W' : modèle de Skiba (T-25).

Fonction pure — aucun I/O. Suit le "réservoir" de travail anaérobie W'
seconde par seconde : dépensé quand la puissance dépasse CP, rechargé
sinon avec une cinétique exponentielle dont la constante de temps dépend
du déficit sous CP (plus on est loin sous CP — repos complet — plus la
récupération est rapide).

Constantes de la formule de tau vérifiées contre une source indépendante
documentant l'algorithme de Skiba (pas recopiées de mémoire) :
https://github.com/Berg0162/RT-Critical-Power (algorithme "différentiel"
de Skiba, tau_W' = 546*exp(-0.01*DCP) + 316, DCP en watts).
"""

import numpy as np

_TAU_A_S = 546.0
_TAU_B = -0.01
_TAU_C_S = 316.0


def recovery_time_constant_s(deficit_below_cp_w: float) -> float:
    """tau_W'(DCP) = 546*exp(-0.01*DCP) + 316, DCP = CP - puissance (>= 0,
    seulement défini sous CP). Décroissante en DCP : la récupération est
    la plus LENTE pile à CP (DCP=0, tau=862s) et se rapproche de son
    plancher (316s) à mesure que la puissance chute vers 0.
    """
    if deficit_below_cp_w < 0:
        raise ValueError(
            f"deficit_below_cp_w doit être >= 0 (défini seulement sous CP), "
            f"reçu {deficit_below_cp_w}"
        )
    return _TAU_A_S * np.exp(_TAU_B * deficit_below_cp_w) + _TAU_C_S


def compute_w_prime_balance(
    power_w: np.ndarray, cp_watts: float, w_prime_joules: float, dt_s: float = 1.0
) -> np.ndarray:
    """W'bal(t) seconde par seconde (ou pas `dt_s`), même longueur que
    `power_w`. Réserve pleine (`w_prime_joules`) juste avant le premier
    échantillon.

    Au-dessus de CP : dépense linéaire, W'bal -= (puissance - CP)·dt_s —
    un débit constant, indépendant de tau (qui ne joue qu'à la
    récupération). En dessous : récupération exponentielle vers
    `w_prime_joules` avec la constante de temps de
    `recovery_time_constant_s`, recalculée à CHAQUE pas (elle dépend de
    la puissance du pas courant, pas figée sur tout l'intervalle sous CP).

    Aucun plancher à 0 : un bilan négatif est un résultat valide, pas une
    erreur — il signale que l'effort modélisé dépasse ce que W' permet de
    soutenir (contrainte à imposer explicitement en aval, ex. T-26, pas
    ici).
    """
    power_w = np.asarray(power_w, dtype=float)
    if len(power_w) == 0:
        raise ValueError("power_w vide : rien à calculer")
    if cp_watts <= 0:
        raise ValueError(f"cp_watts doit être positif, reçu {cp_watts}")
    if w_prime_joules <= 0:
        raise ValueError(f"w_prime_joules doit être positif, reçu {w_prime_joules}")
    if dt_s <= 0:
        raise ValueError(f"dt_s doit être positif, reçu {dt_s}")

    w_bal = np.empty(len(power_w))
    previous_w_bal = w_prime_joules
    for i, power in enumerate(power_w):
        if power > cp_watts:
            current_w_bal = previous_w_bal - (power - cp_watts) * dt_s
        else:
            tau_s = recovery_time_constant_s(cp_watts - power)
            current_w_bal = w_prime_joules - (w_prime_joules - previous_w_bal) * np.exp(
                -dt_s / tau_s
            )
        w_bal[i] = current_w_bal
        previous_w_bal = current_w_bal

    return w_bal
