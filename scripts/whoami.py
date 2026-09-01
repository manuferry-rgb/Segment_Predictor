"""Critère de fin de T-03 : affiche le prénom de l'athlète Strava authentifié.

Usage : uv run python scripts/whoami.py
"""

from pathlib import Path

import httpx

from segment_predictor.ingest.strava_auth import get_valid_access_token
from segment_predictor.ingest.strava_client import get_athlete


def main() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    with httpx.Client(timeout=10.0) as client:
        access_token = get_valid_access_token(client, env_path)
        athlete = get_athlete(client, access_token)
    print(athlete["firstname"])


if __name__ == "__main__":
    main()
