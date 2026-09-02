"""T-24 : régression Ridge de l'indice de performance (T-23) sur les
features de forme (CTL/ATL/TSB, T-21, + duration_s), validation croisée
temporelle.

Le critère de fin du ticket est un R² honnête, même faible — ce script
l'affiche tel quel, il ne cherche pas à le maquiller.

Usage : uv run python scripts/fit_form_model.py
"""

from pathlib import Path

import duckdb
import numpy as np

from segment_predictor.calibrate.form import (
    FORM_REGRESSION_FEATURE_NAMES,
    build_form_regression_dataset,
)
from segment_predictor.models.form_regression import fit_ridge, temporal_cross_validate_ridge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"

ALPHA = 1.0
N_SPLITS = 5


def main() -> None:
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        x, y, dates = build_form_regression_dataset(conn)
    finally:
        conn.close()

    print(f"{len(y)} points, features = {FORM_REGRESSION_FEATURE_NAMES}")

    r2_scores = temporal_cross_validate_ridge(x, y, np.array(dates), alpha=ALPHA, n_splits=N_SPLITS)
    print(f"R² par fold temporel (alpha={ALPHA}) : {[f'{r2:.3f}' for r2 in r2_scores]}")
    print(f"R² moyen = {np.mean(r2_scores):.3f} (écart-type {np.std(r2_scores):.3f})")

    # Fit final sur tout l'historique, pour lire les coefficients (sens du
    # signal, pas une prédiction hors échantillon — ça, c'est le rôle du R²
    # de validation croisée ci-dessus).
    full_fit = fit_ridge(x, y, alpha=ALPHA)
    print("\nCoefficients (sur features standardisées, fit sur tout l'historique) :")
    for name, coefficient in zip(FORM_REGRESSION_FEATURE_NAMES, full_fit.coefficients, strict=True):
        print(f"  {name}: {coefficient:+.4f}")


if __name__ == "__main__":
    main()
