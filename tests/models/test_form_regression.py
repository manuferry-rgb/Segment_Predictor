"""Tests de la régression Ridge et de la validation croisée temporelle (T-24).

Fonctions pures, aucun I/O — implémentées à la main (solution fermée de
Ridge, R², découpage temporel) plutôt qu'avec scikit-learn : pas de
nouvelle dépendance, et chaque étape reste explicable.
"""

import numpy as np
import pytest

from segment_predictor.models.form_regression import (
    fit_ridge,
    predict_ridge,
    r_squared,
    temporal_cross_validate_ridge,
)

# ---- fit_ridge / predict_ridge --------------------------------------------------------------


def test_fit_ridge_recovers_noiseless_linear_relationship_with_tiny_alpha() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, size=(200, 2))
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 5.0  # relation exacte, pas de bruit

    fit = fit_ridge(x, y, alpha=1e-8)
    predicted = predict_ridge(fit, x)

    assert predicted == pytest.approx(y, abs=1e-4)


def test_fit_ridge_shrinks_predictions_toward_the_mean_as_alpha_grows() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-10, 10, size=(200, 2))
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 5.0

    fit_low_alpha = fit_ridge(x, y, alpha=1e-6)
    fit_high_alpha = fit_ridge(x, y, alpha=1e8)

    variance_low = np.var(predict_ridge(fit_low_alpha, x))
    variance_high = np.var(predict_ridge(fit_high_alpha, x))

    # plus de régularisation -> des prédictions plus proches de la moyenne
    # -> moins de variance expliquée par les features
    assert variance_high < 0.01 * variance_low
    assert predict_ridge(fit_high_alpha, x) == pytest.approx(np.mean(y), abs=1.0)


def test_fit_ridge_raises_on_constant_feature() -> None:
    x = np.column_stack([np.ones(10), np.arange(10, dtype=float)])  # 1ere colonne constante
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="constante"):
        fit_ridge(x, y, alpha=1.0)


def test_fit_ridge_raises_on_non_positive_alpha() -> None:
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="alpha"):
        fit_ridge(x, y, alpha=0.0)


# ---- r_squared -------------------------------------------------------------------------------


def test_r_squared_is_one_for_a_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)


def test_r_squared_is_zero_when_predicting_the_mean() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.full_like(y, np.mean(y))
    assert r_squared(y, predicted) == pytest.approx(0.0)


def test_r_squared_can_be_negative_for_a_bad_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([10.0, -5.0, 20.0, -8.0])  # bien pire que la moyenne
    assert r_squared(y, predicted) < 0.0


# ---- temporal_cross_validate_ridge ------------------------------------------------------------


def test_temporal_cross_validate_ridge_returns_one_r_squared_per_fold() -> None:
    rng = np.random.default_rng(2)
    n = 100
    x = rng.uniform(-10, 10, size=(n, 2))
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + rng.normal(0, 0.1, size=n)
    dates = np.arange(n)  # déjà trié, un "jour" par ligne

    r2_scores = temporal_cross_validate_ridge(x, y, dates, alpha=1.0, n_splits=4)

    assert len(r2_scores) == 4


def test_temporal_cross_validate_ridge_never_trains_on_future_data() -> None:
    """Construit par blocs temporels croissants : vérifié indirectement en
    s'assurant qu'un signal qui n'existe QUE dans la seconde moitié
    chronologique donne un bon R² sur les folds tardifs (où le train
    couvre déjà cette moitié) et un R² proche de 0 sur les folds précoces
    (où le train n'a encore vu que du bruit sans signal). Bruit minuscule
    (pas un vrai zéro) dans la 1ère moitié : une colonne EXACTEMENT
    constante ferait lever fit_ridge (test_fit_ridge_raises_on_constant_
    feature), ce n'est pas ce qu'on veut tester ici."""
    rng = np.random.default_rng(3)
    n = 200
    dates = np.arange(n)
    x = np.zeros((n, 1))
    x[: n // 2, 0] = rng.normal(0, 0.01, size=n // 2)
    x[n // 2 :, 0] = np.arange(n // 2, dtype=float)  # signal net dans la 2e moitié
    y = np.zeros(n)
    y[: n // 2] = rng.normal(0, 0.01, size=n // 2)
    y[n // 2 :] = 3.0 * x[n // 2 :, 0]

    r2_scores = temporal_cross_validate_ridge(x, y, dates, alpha=1e-6, n_splits=4)

    assert r2_scores[0] < 0.5  # fold précoce : train encore sans signal
    assert r2_scores[-1] > 0.9  # fold tardif : train couvre déjà le signal


def test_temporal_cross_validate_ridge_raises_when_not_enough_data() -> None:
    x = np.zeros((3, 1))
    y = np.zeros(3)
    dates = np.arange(3)
    with pytest.raises(ValueError, match="n_splits"):
        temporal_cross_validate_ridge(x, y, dates, alpha=1.0, n_splits=5)
