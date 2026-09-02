"""Tests du bilan W' — modèle de Skiba (T-25).

Fonctions pures, aucun I/O. Profils synthétiques (le critère de fin du
ticket), pas de données réelles : chaque cas est vérifiable à la main.
"""

import numpy as np
import pytest

from segment_predictor.models.wbal import compute_w_prime_balance, recovery_time_constant_s

_CP_W = 250.0
_W_PRIME_J = 20_000.0


# ---- recovery_time_constant_s ----------------------------------------------------------------


def test_recovery_time_constant_at_zero_deficit() -> None:
    # tau = 546*exp(0) + 316 = 862 : à DCP=0 (pile à CP), la récupération
    # théorique est à sa plus LENTE (voir test_recovery_is_slower_near_cp).
    assert recovery_time_constant_s(0.0) == pytest.approx(862.0)


def test_recovery_time_constant_decreases_as_deficit_grows() -> None:
    """Plus on est loin en dessous de CP (repos complet), plus la
    récupération est rapide (tau plus petit)."""
    assert recovery_time_constant_s(200.0) < recovery_time_constant_s(50.0)
    assert recovery_time_constant_s(50.0) < recovery_time_constant_s(0.0)


def test_recovery_time_constant_raises_on_negative_deficit() -> None:
    with pytest.raises(ValueError, match="deficit_below_cp_w"):
        recovery_time_constant_s(-10.0)


# ---- compute_w_prime_balance ------------------------------------------------------------------


def test_w_prime_balance_starts_full() -> None:
    power_w = np.full(10, _CP_W)  # pile à CP : ni dépense ni vraie récupération à faire
    w_bal = compute_w_prime_balance(power_w, _CP_W, _W_PRIME_J)
    assert w_bal[0] == pytest.approx(_W_PRIME_J, rel=1e-6)


def test_w_prime_balance_depletes_linearly_above_cp() -> None:
    """Au-dessus de CP, la dépense est un simple débit constant
    (puissance - CP) par seconde — vérifiable analytiquement, sans
    dépendre du tout de tau (qui ne joue qu'en dessous de CP)."""
    power_above_cp = _CP_W + 50.0  # 50W au-dessus de CP
    power_w = np.full(100, power_above_cp)

    w_bal = compute_w_prime_balance(power_w, _CP_W, _W_PRIME_J, dt_s=1.0)

    expected = _W_PRIME_J - 50.0 * np.arange(1, 101)
    assert w_bal == pytest.approx(expected)


def test_w_prime_balance_can_go_negative_without_being_floored() -> None:
    """Pas de plancher artificiel à 0 : une puissance au-dessus de CP
    maintenue plus longtemps que W'/déficit ne le permet donne un bilan
    négatif, signal que l'effort tel que modélisé est intenable — une
    information utile (notamment pour T-26), pas une erreur à cacher."""
    power_w = np.full(1000, _CP_W + 100.0)  # épuiserait W' en 200s
    w_bal = compute_w_prime_balance(power_w, _CP_W, _W_PRIME_J, dt_s=1.0)
    assert w_bal[-1] < 0.0


def test_w_prime_balance_recovers_toward_full_at_rest() -> None:
    """Après une dépense partielle, un long repos (puissance nulle,
    déficit maximal) doit ramener le bilan près de sa valeur pleine."""
    depletion = np.full(60, _CP_W + 100.0)  # 60s à 100W au-dessus de CP : dépense 6000J
    rest = np.zeros(3600)  # 1h de repos complet
    power_w = np.concatenate([depletion, rest])

    w_bal = compute_w_prime_balance(power_w, _CP_W, _W_PRIME_J, dt_s=1.0)

    assert w_bal[59] == pytest.approx(_W_PRIME_J - 6000.0, rel=1e-6)  # juste après la dépense
    assert w_bal[-1] == pytest.approx(_W_PRIME_J, rel=1e-3)  # quasi rechargé après 1h de repos


def test_recovery_is_slower_near_cp_than_at_full_rest() -> None:
    """Même dépense initiale, même durée de récupération : rester juste
    sous CP doit récupérer MOINS qu'un repos complet (tau plus grand
    près de CP, cf recovery_time_constant_s)."""
    depletion = np.full(60, _CP_W + 100.0)
    recovery_duration = 300

    near_cp = np.concatenate([depletion, np.full(recovery_duration, _CP_W - 5.0)])
    full_rest = np.concatenate([depletion, np.zeros(recovery_duration)])

    w_bal_near_cp = compute_w_prime_balance(near_cp, _CP_W, _W_PRIME_J, dt_s=1.0)
    w_bal_full_rest = compute_w_prime_balance(full_rest, _CP_W, _W_PRIME_J, dt_s=1.0)

    assert w_bal_near_cp[-1] < w_bal_full_rest[-1]


def test_w_prime_balance_raises_on_empty_power() -> None:
    with pytest.raises(ValueError, match="vide"):
        compute_w_prime_balance(np.array([]), _CP_W, _W_PRIME_J)


def test_w_prime_balance_raises_on_non_positive_cp_or_w_prime() -> None:
    power_w = np.full(10, 200.0)
    with pytest.raises(ValueError, match="cp_watts"):
        compute_w_prime_balance(power_w, 0.0, _W_PRIME_J)
    with pytest.raises(ValueError, match="w_prime_joules"):
        compute_w_prime_balance(power_w, _CP_W, 0.0)


def test_w_prime_balance_raises_on_non_positive_dt() -> None:
    power_w = np.full(10, 200.0)
    with pytest.raises(ValueError, match="dt_s"):
        compute_w_prime_balance(power_w, _CP_W, _W_PRIME_J, dt_s=0.0)
