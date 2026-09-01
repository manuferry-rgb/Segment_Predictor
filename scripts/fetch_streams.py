"""T-05 : télécharge les streams (puissance, altitude, ...) des sorties vélo
avec capteur de puissance (Ride/VirtualRide, device_watts=true).

Reprenable : une activité déjà téléchargée (fichier déjà sur disque) est
sautée au lancement suivant. S'arrête proprement si le quota journalier
Strava est atteint — relance le lendemain pour continuer.

Usage : uv run python scripts/fetch_streams.py
"""

from pathlib import Path

import httpx

from segment_predictor.ingest.strava_auth import get_valid_access_token
from segment_predictor.ingest.strava_streams import (
    ensure_path_is_gitignored,
    fetch_and_store_streams,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVITIES_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_activities"
STREAMS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_streams"


def main() -> None:
    ensure_path_is_gitignored(STREAMS_RAW_DIR, PROJECT_ROOT)

    env_path = PROJECT_ROOT / ".env"
    with httpx.Client(timeout=30.0) as client:
        access_token = get_valid_access_token(client, env_path)
        summary = fetch_and_store_streams(client, access_token, ACTIVITIES_RAW_DIR, STREAMS_RAW_DIR)

    print(f"Streams téléchargés : {len(summary.fetched_activity_ids)}")
    print(f"Déjà présents (sautés) : {len(summary.already_downloaded_activity_ids)}")
    if summary.stopped_due_to_daily_quota:
        print(
            f"Quota journalier Strava atteint — {len(summary.remaining_activity_ids)} "
            "activité(s) restent à télécharger. Relance le script demain pour continuer."
        )
    elif summary.remaining_activity_ids:
        print(f"{len(summary.remaining_activity_ids)} activité(s) non traitées (inattendu).")
    else:
        print("Terminé : toutes les activités éligibles ont leurs streams.")


if __name__ == "__main__":
    main()
