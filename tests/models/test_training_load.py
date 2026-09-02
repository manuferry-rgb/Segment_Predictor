"""Tests du modèle de charge d'entraînement de Banister : CTL/ATL/TSB (T-21).

Fonction pure, aucun I/O. Les cas ci-dessous vérifient les propriétés
mathématiques de la récursion exponentielle (convergence en régime
constant, décroissance sans TSS, TSB basé sur la veille) plutôt que des
valeurs de référence externes — pas de "vérité terrain" indépendante
pour ce modèle au-delà de sa propre définition.
"""

import pytest

from segment_predictor.models.training_load import (
    DEFAULT_ATL_TIME_CONSTANT_DAYS,
    DEFAULT_CTL_TIME_CONSTANT_DAYS,
    compute_training_load,
)


def test_constant_tss_makes_ctl_and_atl_converge_to_that_value() -> None:
    """En régime permanent (TSS identique tous les jours, assez longtemps),
    une moyenne mobile exponentielle converge vers l'entrée constante —
    propriété mathématique de la récursion, pas une valeur inventée."""
    daily_tss = [80.0] * 300  # largement plus que les deux constantes de temps

    points = compute_training_load(daily_tss)

    assert points[-1].ctl == pytest.approx(80.0, abs=0.5)
    assert points[-1].atl == pytest.approx(80.0, abs=0.5)
    assert points[-1].tsb == pytest.approx(0.0, abs=1.0)


def test_tsb_reflects_yesterdays_ctl_and_atl_not_todays_tss() -> None:
    """Le TSB du jour J doit refléter la forme avec laquelle on ABORDE la
    séance de J (CTL/ATL de J-1), pas ce qu'on vient de faire ce jour-là —
    convention standard (TrainingPeaks/Banister) : sinon un gros pic de
    TSS ferait chuter le TSB du jour même où il a lieu, avant que la
    fatigue n'ait eu le temps de s'accumuler."""
    points = compute_training_load(
        [500.0, 0.0], initial_ctl=50.0, initial_atl=50.0
    )  # jour 1 : TSS énorme

    assert points[0].tsb == pytest.approx(50.0 - 50.0)  # basé sur initial_ctl/atl, pas sur 500


def test_rest_day_decays_ctl_and_atl_rather_than_leaving_them_unchanged() -> None:
    """Un jour sans sortie (TSS=0) n'est pas neutre : CTL et ATL doivent
    tous les deux baisser vers 0, la forme se décharge même au repos."""
    points = compute_training_load([0.0, 0.0, 0.0], initial_ctl=50.0, initial_atl=50.0)

    for i in range(1, len(points)):
        assert points[i].ctl < points[i - 1].ctl
        assert points[i].atl < points[i - 1].atl


def test_atl_reacts_faster_than_ctl_to_a_tss_spike() -> None:
    """Constante de temps ATL (7j) << CTL (42j) par construction : une
    séance ponctuelle doit bouger l'ATL beaucoup plus que la CTL."""
    points = compute_training_load([300.0, 0.0], initial_ctl=50.0, initial_atl=50.0)

    atl_change = abs(points[0].atl - 50.0)
    ctl_change = abs(points[0].ctl - 50.0)
    assert atl_change > ctl_change


def test_default_time_constants_match_banister_convention() -> None:
    assert DEFAULT_CTL_TIME_CONSTANT_DAYS == 42
    assert DEFAULT_ATL_TIME_CONSTANT_DAYS == 7


def test_raises_on_empty_daily_tss() -> None:
    with pytest.raises(ValueError, match="vide"):
        compute_training_load([])


def test_raises_on_non_positive_time_constants() -> None:
    with pytest.raises(ValueError, match="ctl_time_constant_days"):
        compute_training_load([10.0], ctl_time_constant_days=0)
    with pytest.raises(ValueError, match="atl_time_constant_days"):
        compute_training_load([10.0], atl_time_constant_days=-1)
