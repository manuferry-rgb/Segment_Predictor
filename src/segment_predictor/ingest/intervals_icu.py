"""Wellness intervals.icu (HRV, sommeil, FC de repos, poids) — couche
ingest, aucune transformation (T-22).

Authentification par Basic Auth : utilisateur "API_KEY", mot de passe la
clé API elle-même (générée dans intervals.icu > Réglages > Developer
Settings) — pas d'OAuth ni de refresh token, contrairement à Strava
(strava_auth.py).

Un seul fichier Parquet (wellness.parquet), écrasé à chaque fetch —
pas accumulé comme les activités Strava (strava_activities.py) : un seul
appel API couvre toute la plage de dates demandée (pas de pagination ni
de rate-limit agressif comme Strava), et les entrées de wellness peuvent
être corrigées rétroactivement (ex. saisie HRV en retard). Un refetch
complet reste donc correct là où un "depuis la dernière fois" manquerait
les corrections sur des jours déjà téléchargés.
"""

import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import dotenv_values

INTERVALS_ICU_BASE_URL = "https://intervals.icu/api/v1"
DEFAULT_RETRY_BACKOFF_S = 60.0
MAX_RETRIES = 5


def read_credentials(env_path: Path) -> tuple[str, str]:
    """(athlete_id, api_key) depuis le .env — jamais de valeur par défaut
    silencieuse, comme strava_auth._require."""
    values = dotenv_values(env_path)
    athlete_id = values.get("INTERVALS_ICU_ATHLETE_ID")
    if not athlete_id:
        raise KeyError("INTERVALS_ICU_ATHLETE_ID manquant dans le .env")
    api_key = values.get("INTERVALS_ICU_API_KEY")
    if not api_key:
        raise KeyError("INTERVALS_ICU_API_KEY manquant dans le .env")
    return athlete_id, api_key


def get_wellness(
    http_client: httpx.Client,
    athlete_id: str,
    api_key: str,
    oldest: str,
    newest: str,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """GET /athlete/{id}/wellness.json, JSON brut tel que renvoyé par
    intervals.icu (une entrée par jour civil ayant au moins un champ
    renseigné). `oldest`/`newest` : dates ISO-8601 "YYYY-MM-DD", bornes
    incluses.
    """
    url = f"{INTERVALS_ICU_BASE_URL}/athlete/{athlete_id}/wellness.json"
    params = {"oldest": oldest, "newest": newest}
    auth = httpx.BasicAuth("API_KEY", api_key)

    attempts = 0
    while True:
        response = http_client.get(url, params=params, auth=auth)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()

        attempts += 1
        if attempts > MAX_RETRIES:
            response.raise_for_status()  # toujours 429 -> on abandonne, l'erreur remonte

        wait_s = float(response.headers.get("Retry-After", DEFAULT_RETRY_BACKOFF_S))
        sleep(wait_s)


def save_wellness(raw_dir: Path, records: list[dict]) -> Path:
    """Écrit tout l'historique wellness dans un seul Parquet, en écrasant
    le précédent (voir docstring du module)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / "wellness.parquet"
    table = (
        pa.Table.from_pylist(records)
        if records
        else pa.Table.from_pylist([], schema=pa.schema([("id", pa.string())]))
    )
    pq.write_table(table, file_path)
    return file_path


def fetch_and_store_wellness(
    http_client: httpx.Client,
    athlete_id: str,
    api_key: str,
    oldest: str,
    newest: str,
    raw_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """get_wellness + save_wellness, la fonction que les scripts appellent."""
    records = get_wellness(http_client, athlete_id, api_key, oldest, newest, sleep=sleep)
    return save_wellness(raw_dir, records)
