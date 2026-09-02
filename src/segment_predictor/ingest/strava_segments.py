"""Récupération des segments Strava favoris (couche ingest, aucune transformation).

Un JSON brut par segment (`{id}.parquet`), stocké tel quel — y compris
le KOM sous sa forme "mm:ss" et le PR déjà en secondes. Le parsing du
KOM se fait dans storage/segments.py, pas ici.

Reprise : comme T-05, l'id est une clé stable, "déjà téléchargé" = le
fichier existe déjà. Un segment déjà présent n'est jamais redemandé à
Strava, même si son KOM a changé depuis côté serveur — il faut
supprimer le fichier localement pour forcer un refetch.
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

PER_PAGE = 200


def list_starred_segment_ids(
    http_client: httpx.Client,
    access_token: str,
    sleep: Callable[[float], None] = time.sleep,
) -> list[int]:
    """GET /segments/starred, pagination jusqu'à épuisement (même pattern que
    list_activities, T-04). Renvoie juste les IDs — get_segment récupère le
    détail de chacun (KOM, PR, pente...) séparément, comme pour une liste
    d'IDs donnée à la main.
    """
    segment_ids: list[int] = []
    page = 1
    while True:
        response = authenticated_get(
            http_client,
            "/segments/starred",
            access_token,
            params={"per_page": PER_PAGE, "page": page},
            sleep=sleep,
        )
        batch = response.json()
        segment_ids.extend(segment["id"] for segment in batch)
        if len(batch) < PER_PAGE:
            break
        page += 1
    return segment_ids


def _fetch_segment_response(
    http_client: httpx.Client, access_token: str, segment_id: int, sleep: Callable[[float], None]
) -> httpx.Response:
    return authenticated_get(http_client, f"/segments/{segment_id}", access_token, sleep=sleep)


def get_segment(
    http_client: httpx.Client,
    access_token: str,
    segment_id: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """GET /segments/{id}, JSON brut (xoms.kom, athlete_segment_stats, ...)."""
    return _fetch_segment_response(http_client, access_token, segment_id, sleep).json()


def save_segment(raw_dir: Path, segment_id: int, segment: dict) -> Path:
    """Écrit UN segment dans son propre Parquet, forme brute inchangée."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{segment_id}.parquet"
    pq.write_table(pa.Table.from_pylist([segment]), file_path)
    return file_path


def _already_downloaded_ids(raw_dir: Path) -> set[int]:
    if not raw_dir.exists():
        return set()
    return {int(path.stem) for path in raw_dir.glob("*.parquet")}


@dataclass(frozen=True)
class FetchSegmentsSummary:
    """Bilan d'un lancement : ce qui a été fait, ce qu'il reste, et pourquoi ça s'est arrêté."""

    fetched_ids: list[int]
    already_downloaded_ids: list[int]
    remaining_ids: list[int]
    stopped_due_to_daily_quota: bool


def fetch_and_store_segments(
    http_client: httpx.Client,
    access_token: str,
    segment_ids: list[int],
    raw_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchSegmentsSummary:
    """Récupère les segments demandés qui ne sont pas encore sur disque.

    Même logique de quota que T-05 (fetch_and_store_streams) : pause sur
    quota court terme, arrêt propre sur quota journalier ou 429 persistant.
    """
    already_downloaded_ids = _already_downloaded_ids(raw_dir)
    to_fetch = [
        segment_id for segment_id in segment_ids if segment_id not in already_downloaded_ids
    ]

    fetched_ids: list[int] = []
    for position, segment_id in enumerate(to_fetch):
        try:
            response = _fetch_segment_response(http_client, access_token, segment_id, sleep)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                return FetchSegmentsSummary(
                    fetched_ids=fetched_ids,
                    already_downloaded_ids=sorted(already_downloaded_ids),
                    remaining_ids=to_fetch[position:],
                    stopped_due_to_daily_quota=True,
                )
            raise

        save_segment(raw_dir, segment_id, response.json())
        fetched_ids.append(segment_id)

        rate_limit = parse_rate_limit(response)
        if rate_limit is not None:
            if rate_limit.usage_daily >= rate_limit.limit_daily:
                return FetchSegmentsSummary(
                    fetched_ids=fetched_ids,
                    already_downloaded_ids=sorted(already_downloaded_ids),
                    remaining_ids=to_fetch[position + 1 :],
                    stopped_due_to_daily_quota=True,
                )
            if rate_limit.short_term_exhausted:
                sleep(DEFAULT_RATE_LIMIT_BACKOFF_S)

    return FetchSegmentsSummary(
        fetched_ids=fetched_ids,
        already_downloaded_ids=sorted(already_downloaded_ids),
        remaining_ids=[],
        stopped_due_to_daily_quota=False,
    )
