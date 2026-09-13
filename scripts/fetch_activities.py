"""T-04 : récupère les nouvelles activités Strava et les stocke en Parquet brut.

Relançable sans risque : ne redemande à Strava que ce qui est postérieur
à la dernière activité déjà stockée (voir strava_activities.read_watermark).

`user_id` : requis (T-44d) — chaque utilisateur a son propre sous-dossier
`data/raw/strava_activities/<user_id>/`, sinon la synchro de deux
personnes mélangerait leurs fichiers bruts dans le même dossier.

Usage : uv run python scripts/fetch_activities.py <user_id>
"""

import sys
from pathlib import Path

import httpx

from segment_predictor.ingest.strava_activities import fetch_and_store_new_activities
from segment_predictor.ingest.strava_auth import get_valid_access_token

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR_ROOT = PROJECT_ROOT / "data" / "raw" / "strava_activities"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage : uv run python scripts/fetch_activities.py <user_id>")
        raise SystemExit(1)
    raw_dir = RAW_DIR_ROOT / sys.argv[1]

    env_path = PROJECT_ROOT / ".env"
    with httpx.Client(timeout=30.0) as client:
        access_token = get_valid_access_token(client, env_path)
        written_path = fetch_and_store_new_activities(client, access_token, raw_dir)

    if written_path is None:
        print("Aucune nouvelle activité.")
    else:
        print(f"Écrit : {written_path}")


if __name__ == "__main__":
    main()
