"""Tests de l'indice de performance et du filtre "effort maximal" (T-23).

Fonctions pures, aucun I/O.
"""

import pytest

from segment_predictor.models.form import (
    DEFAULT_MAXIMAL_EFFORT_THRESHOLD,
    is_near_maximal_effort,
    performance_index,
)
from segment_predictor.models.power import CriticalPowerFit

_CP_FIT = CriticalPowerFit(
    cp_watts=250.0, w_prime_joules=20_000.0, r_squared=0.9, n_points=5, duration_range_s=(180, 1200)
)


# ---- performance_index --------------------------------------------------------------------


def test_performance_index_is_one_when_actual_matches_prediction_exactly() -> None:
    predicted_power_w = _CP_FIT.cp_watts + _CP_FIT.w_prime_joules / 300.0

    index = performance_index(predicted_power_w, _CP_FIT, duration_s=300.0)

    assert index == pytest.approx(1.0)


def test_performance_index_above_one_means_better_than_predicted() -> None:
    predicted_power_w = _CP_FIT.cp_watts + _CP_FIT.w_prime_joules / 300.0

    index = performance_index(predicted_power_w * 1.1, _CP_FIT, duration_s=300.0)

    assert index == pytest.approx(1.1)


def test_performance_index_raises_outside_cp_validity_range() -> None:
    with pytest.raises(ValueError, match="plage de validité"):
        performance_index(300.0, _CP_FIT, duration_s=60.0)  # < 180s
    with pytest.raises(ValueError, match="plage de validité"):
        performance_index(300.0, _CP_FIT, duration_s=2000.0)  # > 1200s


# ---- is_near_maximal_effort ----------------------------------------------------------------


def test_is_near_maximal_effort_true_at_the_record() -> None:
    assert is_near_maximal_effort(300.0, best_so_far_mmp_w=300.0) is True


def test_is_near_maximal_effort_true_just_above_threshold() -> None:
    assert is_near_maximal_effort(286.0, best_so_far_mmp_w=300.0) is True  # 95.3%


def test_is_near_maximal_effort_false_below_threshold() -> None:
    assert is_near_maximal_effort(200.0, best_so_far_mmp_w=300.0) is False  # 66.7%


def test_is_near_maximal_effort_respects_custom_threshold() -> None:
    assert is_near_maximal_effort(250.0, best_so_far_mmp_w=300.0, threshold=0.8) is True
    assert is_near_maximal_effort(250.0, best_so_far_mmp_w=300.0, threshold=0.9) is False


def test_default_maximal_effort_threshold_is_95_percent() -> None:
    assert DEFAULT_MAXIMAL_EFFORT_THRESHOLD == 0.95


def test_is_near_maximal_effort_raises_on_non_positive_reference() -> None:
    with pytest.raises(ValueError, match="best_so_far_mmp_w"):
        is_near_maximal_effort(100.0, best_so_far_mmp_w=0.0)
