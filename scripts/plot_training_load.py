"""T-21 : courbes CTL/ATL/TSB (modèle de Banister) sur tout l'historique réel.

Le critère de fin du ticket ("courbes cohérentes avec mon vécu") est une
vérification visuelle : ce script produit le graphique, mais seul toi
peux juger si les pics/creux correspondent à ce que tu as vécu (blocs
d'entraînement, coupures, tapering avant une course...).

Usage : uv run python scripts/plot_training_load.py
"""

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_predictor.calibrate.draft_tagging import fit_current_cp
from segment_predictor.calibrate.training_load import compute_training_load_from_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
# Artefact dérivé, gitignoré (data/) — même logique que backtest_t18.png (T-18).
PNG_PATH = PROJECT_ROOT / "data" / "training_load_t21.png"


def main() -> None:
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        cp_fit = fit_current_cp(conn)
        dates, points = compute_training_load_from_db(conn, cp_fit=cp_fit)
    finally:
        conn.close()

    ctl = [p.ctl for p in points]
    atl = [p.atl for p in points]
    tsb = [p.tsb for p in points]

    print(f"{dates[0]} -> {dates[-1]} ({len(dates)} jours), CP={cp_fit.cp_watts:.0f} W (proxy FTP)")
    print(f"dernier point : CTL={ctl[-1]:.1f}  ATL={atl[-1]:.1f}  TSB={tsb[-1]:.1f}")

    fig, (ax_load, ax_tsb) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_load.plot(dates, ctl, label="CTL (forme)", color="tab:blue")
    ax_load.plot(dates, atl, label="ATL (fatigue)", color="tab:orange")
    ax_load.set_ylabel("TSS/jour (moyenne mobile)")
    ax_load.legend()
    ax_load.set_title("Charge d'entraînement — modèle de Banister (T-21)")

    ax_tsb.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax_tsb.plot(dates, tsb, color="tab:green")
    ax_tsb.set_ylabel("TSB (forme du jour)")
    ax_tsb.set_xlabel("Date")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    print(f"Graphique : {PNG_PATH}")


if __name__ == "__main__":
    main()
