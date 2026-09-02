"""Récupération d'activités Strava en vue détaillée (couche ingest, T-07b).

Différent de T-04 (`GET /athlete/activities`, résumé sans efforts) :
`GET /activities/{id}` embarque un tableau `segment_efforts` — tous les
segments publics croisés par l'activité, pas seulement mes favoris.
C'est le seul moyen d'obtenir mon historique de passages sur un segment,
`GET /segments/{id}` (T-06) ne donnant que mon PR agrégé.

Même pattern reprenable et gestion de quota que T-05 (strava_streams.py) :
un Parquet par activité, arrêt propre sur quota journalier.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from segment_predictor.ingest.strava_client import (
    DEFAULT_RATE_LIMIT_BACKOFF_S,
    authenticated_get,
    parse_rate_limit,
)


def _fetch_activity_detail_response(
    http_client: httpx.Client, access_token: str, activity_id: int, sleep: Callable[[float], None]
) -> httpx.Response:
    return authenticated_get(http_client, f"/activities/{activity_id}", access_token, sleep=sleep)


def get_activity_detail(
    http_client: httpx.Client,
    access_token: str,
    activity_id: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """GET /activities/{id} (vue détaillée), JSON brut tel que renvoyé par Strava."""
    return _fetch_activity_detail_response(http_client, access_token, activity_id, sleep).json()


def save_activity_detail(raw_dir: Path, activity_id: int, detail: dict) -> Path:
    """Écrit le détail d'UNE activité dans son propre Parquet, forme brute inchangée."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{activity_id}.parquet"
    pq.write_table(pa.Table.from_pylist([detail]), file_path)
    return file_path


def _already_downloaded_ids(raw_dir: Path) -> set[int]:
    if not raw_dir.exists():
        return set()
    return {int(path.stem) for path in raw_dir.glob("*.parquet")}


@dataclass(frozen=True)
class FetchActivityDetailsSummary:
    """Bilan d'un lancement : ce qui a été fait, ce qu'il reste, et pourquoi ça s'est arrêté."""

    fetched_ids: list[int]
    already_downloaded_ids: list[int]
    remaining_ids: list[int]
    stopped_due_to_daily_quota: bool


def fetch_and_store_activity_details(
    http_client: httpx.Client,
    access_token: str,
    activity_ids: list[int],
    raw_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchActivityDetailsSummary:
    """Télécharge la vue détaillée des activités demandées pas encore sur disque.

    Même logique de quota que T-05/T-06 : pause sur quota court terme,
    arrêt propre sur quota journalier ou 429 persistant.
    """
    already_downloaded_ids = _already_downloaded_ids(raw_dir)
    to_fetch = [
        activity_id for activity_id in activity_ids if activity_id not in already_downloaded_ids
    ]

    fetched_ids: list[int] = []
    for position, activity_id in enumerate(to_fetch):
        try:
            response = _fetch_activity_detail_response(
                http_client, access_token, activity_id, sleep
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                return FetchActivityDetailsSummary(
                    fetched_ids=fetched_ids,
                    already_downloaded_ids=sorted(already_downloaded_ids),
                    remaining_ids=to_fetch[position:],
                    stopped_due_to_daily_quota=True,
                )
            raise

        save_activity_detail(raw_dir, activity_id, response.json())
        fetched_ids.append(activity_id)

        rate_limit = parse_rate_limit(response)
        if rate_limit is not None:
            if rate_limit.usage_daily >= rate_limit.limit_daily:
                return FetchActivityDetailsSummary(
                    fetched_ids=fetched_ids,
                    already_downloaded_ids=sorted(already_downloaded_ids),
                    remaining_ids=to_fetch[position + 1 :],
                    stopped_due_to_daily_quota=True,
                )
            if rate_limit.short_term_exhausted:
                sleep(DEFAULT_RATE_LIMIT_BACKOFF_S)

    return FetchActivityDetailsSummary(
        fetched_ids=fetched_ids,
        already_downloaded_ids=sorted(already_downloaded_ids),
        remaining_ids=[],
        stopped_due_to_daily_quota=False,
    )
