"""T-22 : récupère le wellness intervals.icu (HRV, sommeil, FC de repos,
poids) pour toute la période couverte par les activités déjà en base.

Un seul appel API (pas de pagination), qui écrase le fichier Parquet
précédent à chaque lancement — voir ingest/intervals_icu.py pour le
raisonnement (le wellness peut être corrigé rétroactivement).

Prérequis : `main.activities` doit déjà exister (scripts/build_database.py),
et `.env` doit contenir INTERVALS_ICU_ATHLETE_ID et INTERVALS_ICU_API_KEY
(générés dans intervals.icu > Réglages > Developer Settings).

Usage : uv run python scripts/fetch_wellness.py
"""

from datetime import date
from pathlib import Path

import duckdb
import httpx

from segment_predictor.ingest.intervals_icu import fetch_and_store_wellness, read_credentials

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WELLNESS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "intervals_icu"
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
ENV_PATH = PROJECT_ROOT / ".env"


def main() -> None:
    conn = duckdb.connect(str(DUCKDB_PATH))
    try:
        oldest_row = conn.execute("SELECT MIN(start_date) FROM activities").fetchone()
    finally:
        conn.close()

    if oldest_row is None or oldest_row[0] is None:
        raise SystemExit("Aucune activité en base — lance scripts/build_database.py d'abord.")
    oldest = oldest_row[0].date().isoformat()
    newest = date.today().isoformat()

    athlete_id, api_key = read_credentials(ENV_PATH)

    with httpx.Client(timeout=30.0) as client:
        written_path = fetch_and_store_wellness(
            client, athlete_id, api_key, oldest, newest, WELLNESS_RAW_DIR
        )

    print(f"Wellness {oldest} -> {newest} écrit : {written_path}")


if __name__ == "__main__":
    main()
