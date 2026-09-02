"""T-18 : backtest de la calibration CdA/Crr par split temporel.

Réutilise exactement les mêmes efforts que la calibration officielle
(T-17 : tagués solo, dans la plage de validité CP, avec un pr_rank non
nul), les trie par date, calibre CdA/Crr sur les plus anciens et évalue
sur les plus récents — jamais l'inverse. Affiche l'erreur absolue
médiane (en secondes, sur le set de test) et sauvegarde un nuage de
points prédit vs réel.

Usage : uv run python scripts/backtest_cda_crr.py
"""

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif : juste le PNG en sortie
import matplotlib.pyplot as plt

from segment_predictor.calibrate.backtest import backtest_cda_crr_from_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"
# data/ est gitignoré : le graphique est un artefact dérivé de mes
# données perso, régénérable, pas une donnée d'entrée (contrairement à
# annotations/draft_status.csv) — même logique que le .duckdb.
PNG_PATH = PROJECT_ROOT / "data" / "backtest_t18.png"

# Toi + vélo — ajuste si ton poids ou ton vélo change.
MASS_KG = 91.0


def main() -> None:
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        result = backtest_cda_crr_from_db(conn, CSV_PATH, mass_kg=MASS_KG)
    finally:
        conn.close()

    print(f"CdA = {result.cda_crr_fit.cda_m2:.4f} m², Crr = {result.cda_crr_fit.crr:.5f}")
    print(f"n_train = {result.n_train}, n_test = {result.n_test}")
    print(f"erreur absolue médiane (test) = {result.median_absolute_error_s:.1f} s")

    actual_s = [actual for actual, _ in result.predictions]
    predicted_s = [predicted for _, predicted in result.predictions]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual_s, predicted_s)
    lims = [0.0, max(actual_s + predicted_s) * 1.05]
    ax.plot(lims, lims, linestyle="--", color="gray")  # y=x : prédiction parfaite
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Temps réel (s)")
    ax.set_ylabel("Temps prédit (s)")
    ax.set_title(
        f"Backtest T-18 — erreur médiane {result.median_absolute_error_s:.1f}s (n={result.n_test})"
    )
    fig.savefig(PNG_PATH)
    print(f"Graphique : {PNG_PATH}")


if __name__ == "__main__":
    main()
