"""Appels bruts à l'API Strava (couche ingest : aucune transformation)."""

import httpx

STRAVA_API_BASE_URL = "https://www.strava.com/api/v3"


def get_athlete(http_client: httpx.Client, access_token: str) -> dict:
    """GET /athlete — profil de l'athlète authentifié, JSON brut tel que renvoyé par Strava."""
    response = http_client.get(
        f"{STRAVA_API_BASE_URL}/athlete",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    response.raise_for_status()
    return response.json()
