"""T-07b : récupère la vue détaillée des activités éligibles (celles avec
streams, T-05) pour en extraire les efforts sur segment.

Reprenable : une activité déjà téléchargée n'est jamais redemandée.
S'arrête proprement si le quota journalier Strava est atteint.

Usage : uv run python scripts/fetch_activity_details.py
"""

from pathlib import Path

import httpx

from segment_predictor.ingest.strava_activity_details import fetch_and_store_activity_details
from segment_predictor.ingest.strava_auth import get_valid_access_token
from segment_predictor.ingest.strava_streams import list_eligible_activity_ids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVITIES_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_activities"
DETAILS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_activity_details"


def main() -> None:
    eligible_ids = list_eligible_activity_ids(ACTIVITIES_RAW_DIR)
    print(f"{len(eligible_ids)} activités éligibles (mêmes que T-05)")

    env_path = PROJECT_ROOT / ".env"
    with httpx.Client(timeout=30.0) as client:
        access_token = get_valid_access_token(client, env_path)
        summary = fetch_and_store_activity_details(
            client, access_token, eligible_ids, DETAILS_RAW_DIR
        )

    print(f"Détails récupérés : {len(summary.fetched_ids)}")
    print(f"Déjà présents (sautés) : {len(summary.already_downloaded_ids)}")
    if summary.stopped_due_to_daily_quota:
        print(
            f"Quota journalier Strava atteint — {len(summary.remaining_ids)} "
            "activité(s) restent à télécharger. Relance le script demain."
        )
    elif summary.remaining_ids:
        print(f"{len(summary.remaining_ids)} activité(s) non traitées (inattendu).")
    else:
        print("Terminé : toutes les activités éligibles ont leur vue détaillée.")


if __name__ == "__main__":
    main()
