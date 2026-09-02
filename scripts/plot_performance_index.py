"""T-23 : série temporelle de l'indice de performance (puissance réelle /
puissance prédite par le modèle CP, sur les efforts jugés proches du
maximum — voir calibrate/form.py et models/form.py pour la définition du
filtre).

Usage : uv run python scripts/plot_performance_index.py
"""

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_predictor.calibrate.form import compute_performance_index_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
# Artefact dérivé, gitignoré (data/) — même logique que les PNG de T-18/T-21.
PNG_PATH = PROJECT_ROOT / "data" / "performance_index_t23.png"


def main() -> None:
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        points = compute_performance_index_series(conn)
    finally:
        conn.close()

    if not points:
        raise SystemExit("Aucun effort jugé proche du maximum trouvé — rien à tracer.")

    durations = sorted({p.duration_s for p in points})
    print(f"{len(points)} points (efforts proches du maximum) sur {len(durations)} durées")
    print(f"index moyen = {sum(p.index for p in points) / len(points):.3f}")
    fig, ax = plt.subplots(figsize=(12, 5))
    for duration_s in durations:
        subset = [p for p in points if p.duration_s == duration_s]
        ax.plot(
            [p.date for p in subset],
            [p.index for p in subset],
            marker="o",
            linestyle="-",
            label=f"{duration_s}s",
        )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Indice de performance (réel / prédit par le modèle CP)")
    ax.set_xlabel("Date")
    ax.set_title("Indice de performance — efforts proches du maximum (T-23)")
    ax.legend(title="Durée")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    print(f"Graphique : {PNG_PATH}")


if __name__ == "__main__":
    main()
