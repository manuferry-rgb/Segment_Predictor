"""T-16 : (re)génère le CSV de tri pour le tag manuel des efforts en groupe.

Le CSV (annotations/draft_status.csv) est une donnée d'entrée VERSIONNÉE
dans Git, pas un artefact jetable : lance ce script, ouvre le CSV, remplis
la colonne `draft_status` (solo/drafted/unknown) pour les lignes qui
t'intéressent, commit. Relancer ce script ne touche jamais aux valeurs
déjà annotées — seuls le temps prédit et l'écart sont recalculés (ils
reflètent "le modèle actuel", qui évolue avec plus de données).

⚠️ Le temps prédit vient d'un modèle non calibré (CdA/Crr génériques) et
d'une approximation supplémentaire (segment = un seul tronçon à pente
moyenne). L'écart est une aide au tri, pas une détection — voir
draft_tagging.py et le README pour le détail.

Usage : uv run python scripts/generate_draft_tagging_csv.py
"""

from pathlib import Path

import duckdb

from segment_predictor.calibrate.draft_tagging import generate_draft_tagging_csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"

# Toi + vélo — ajuste si ton poids ou ton vélo change.
MASS_KG = 91.0


def main() -> None:
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        generate_draft_tagging_csv(conn, CSV_PATH, mass_kg=MASS_KG)
    finally:
        conn.close()

    print(f"CSV généré : {CSV_PATH}")


if __name__ == "__main__":
    main()
