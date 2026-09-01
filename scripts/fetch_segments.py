"""T-06 : récupère des segments Strava (KOM + PR) et alimente la table DuckDB `segments`.

Vérifie d'abord ce qui est déjà sur disque : un segment déjà téléchargé
n'est jamais redemandé à Strava. La table `segments` est reconstruite à
partir de TOUS les segments bruts déjà présents, pas seulement ceux
demandés dans cet appel.

Usage : uv run python scripts/fetch_segments.py <segment_id> [<segment_id> ...]
"""

import sys
from pathlib import Path

import duckdb
import httpx

from segment_predictor.ingest.strava_auth import get_valid_access_token
from segment_predictor.ingest.strava_segments import fetch_and_store_segments
from segment_predictor.storage.segments import build_segments_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_segments"
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"


def main() -> None:
    segment_ids = [int(arg) for arg in sys.argv[1:]]
    if not segment_ids:
        raise SystemExit("Usage : uv run python scripts/fetch_segments.py <segment_id> [...]")

    env_path = PROJECT_ROOT / ".env"
    with httpx.Client(timeout=30.0) as client:
        access_token = get_valid_access_token(client, env_path)
        summary = fetch_and_store_segments(client, access_token, segment_ids, SEGMENTS_RAW_DIR)

    print(f"Segments récupérés : {len(summary.fetched_ids)}")
    print(f"Déjà présents (sautés) : {len(summary.already_downloaded_ids)}")
    if summary.stopped_due_to_daily_quota:
        print(
            f"Quota journalier Strava atteint — {len(summary.remaining_ids)} "
            "segment(s) restent à télécharger. Relance le script demain."
        )

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DUCKDB_PATH))
    try:
        build_segments_table(conn, SEGMENTS_RAW_DIR)
        row_count = conn.execute("SELECT count(*) FROM segments").fetchone()[0]
    finally:
        conn.close()

    print(f"Table `segments` reconstruite dans {DUCKDB_PATH} ({row_count} lignes).")


if __name__ == "__main__":
    main()
