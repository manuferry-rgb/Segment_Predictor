"""T-06 : récupère des segments Strava (KOM + PR) et alimente la table DuckDB `segments`.

Sans argument, récupère TOUS tes segments favoris (`GET /segments/starred`)
— c'est le mode normal. Donne des IDs explicites en argument seulement
pour cibler des segments précis (pas forcément favoris).

Vérifie d'abord ce qui est déjà sur disque : un segment déjà téléchargé
n'est jamais redemandé à Strava. La table `segments` est reconstruite à
partir de TOUS les segments bruts déjà présents, pas seulement ceux
demandés dans cet appel.

Usage :
  uv run python scripts/fetch_segments.py                    # tous les favoris
  uv run python scripts/fetch_segments.py <segment_id> [...]  # IDs précis
"""

import sys
from pathlib import Path

import duckdb
import httpx

from segment_predictor.ingest.strava_auth import get_valid_access_token
from segment_predictor.ingest.strava_segments import (
    fetch_and_store_segments,
    list_starred_segment_ids,
)
from segment_predictor.storage.segments import build_segments_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_segments"
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"


def main() -> None:
    explicit_ids = [int(arg) for arg in sys.argv[1:]]

    env_path = PROJECT_ROOT / ".env"
    with httpx.Client(timeout=30.0) as client:
        access_token = get_valid_access_token(client, env_path)

        if explicit_ids:
            segment_ids = explicit_ids
        else:
            segment_ids = list_starred_segment_ids(client, access_token)
            print(f"{len(segment_ids)} segments favoris trouvés sur Strava")

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
