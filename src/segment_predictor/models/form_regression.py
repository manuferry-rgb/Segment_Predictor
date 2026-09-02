"""Régression Ridge et validation croisée temporelle (T-24).

Fonctions pures — aucun I/O. Implémentées à la main (solution fermée de
Ridge, R², découpage temporel) plutôt qu'avec scikit-learn : pas de
nouvelle dépendance pour ce qui reste une résolution d'algèbre linéaire
de quelques lignes, et chaque étape reste explicable.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RidgeFit:
    """Coefficients sur les features STANDARDISÉES (moyenne 0, écart-type
    1) — c'est ce qui rend la pénalisation L2 comparable entre features
    d'échelles très différentes (ex. TSB ~ ±20, duration_s ~ 100-1000).
    `predict_ridge` refait la standardisation à partir de `feature_means`/
    `feature_stds`, jamais besoin de la refaire à la main.
    """

    coefficients: np.ndarray
    intercept: float
    feature_means: np.ndarray
    feature_stds: np.ndarray
    alpha: float


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> RidgeFit:
    """Ridge : minimise ||y - Xβ||² + alpha·||β||² (L2). Solution fermée
    β = (XᵀX + alpha·I)⁻¹Xᵀy sur les features centrées-réduites, target
    centrée (l'intercept n'est jamais pénalisé — seule sa valeur, pas sa
    magnitude, compte).

    `alpha` > 0 : à 0 ce serait une régression linéaire ordinaire, plus
    la peine de passer par Ridge (et le nom du paramètre perdrait son
    sens) — ValueError explicite plutôt qu'un alpha=0 qui marcherait
    par accident.
    """
    if alpha <= 0:
        raise ValueError(f"alpha doit être positif, reçu {alpha}")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    feature_means = x.mean(axis=0)
    feature_stds = x.std(axis=0)
    if np.any(feature_stds == 0):
        raise ValueError(
            "au moins une feature est constante (écart-type nul) : impossible de la "
            "standardiser, et elle n'apporterait de toute façon aucune information"
        )

    x_standardized = (x - feature_means) / feature_stds
    y_mean = y.mean()
    y_centered = y - y_mean

    n_features = x_standardized.shape[1]
    coefficients = np.linalg.solve(
        x_standardized.T @ x_standardized + alpha * np.eye(n_features),
        x_standardized.T @ y_centered,
    )

    return RidgeFit(
        coefficients=coefficients,
        intercept=float(y_mean),
        feature_means=feature_means,
        feature_stds=feature_stds,
        alpha=alpha,
    )


def predict_ridge(fit: RidgeFit, x: np.ndarray) -> np.ndarray:
    x_standardized = (np.asarray(x, dtype=float) - fit.feature_means) / fit.feature_stds
    return fit.intercept + x_standardized @ fit.coefficients


def r_squared(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """1 - (somme des carrés des résidus / somme des carrés totale) :
    1.0 = prédiction parfaite, 0.0 = pas mieux que prédire la moyenne de
    `y_true`, négatif = pire que cette moyenne (possible hors échantillon
    d'entraînement, ex. en validation croisée).

    `y_true` constant (variance nulle) rend le dénominateur nul : R²
    n'est alors mathématiquement pas défini, pas juste "1.0 par
    convention" — ValueError plutôt qu'un NaN/inf silencieux.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_predicted = np.asarray(y_predicted, dtype=float)

    total_sum_of_squares = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum_of_squares == 0:
        raise ValueError("y_true est constant : R² n'est pas défini (variance totale nulle)")

    residual_sum_of_squares = np.sum((y_true - y_predicted) ** 2)
    return float(1 - residual_sum_of_squares / total_sum_of_squares)


def temporal_cross_validate_ridge(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    alpha: float,
    n_splits: int,
) -> list[float]:
    """Découpe (x, y), triés par `dates`, en `n_splits + 1` blocs
    chronologiques contigus. Fold i : entraîne sur tous les blocs
    jusqu'à i inclus, teste sur le bloc i+1 — jamais de donnée future
    dans le train (le premier bloc ne sert donc jamais de test, juste
    d'amorce). Retourne un R² par fold.
    """
    order = np.argsort(np.asarray(dates))
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]

    n = len(y_sorted)
    n_blocks = n_splits + 1
    block_size = n // n_blocks
    if block_size < 2:
        raise ValueError(
            f"pas assez de points ({n}) pour n_splits={n_splits} : chaque bloc de test a "
            "besoin d'au moins 2 points"
        )

    boundaries = [round(i * n / n_blocks) for i in range(n_blocks + 1)]

    scores = []
    for i in range(n_splits):
        train_end = boundaries[i + 1]
        test_start, test_end = boundaries[i + 1], boundaries[i + 2]

        fit = fit_ridge(x_sorted[:train_end], y_sorted[:train_end], alpha)
        predicted = predict_ridge(fit, x_sorted[test_start:test_end])
        scores.append(r_squared(y_sorted[test_start:test_end], predicted))

    return scores
