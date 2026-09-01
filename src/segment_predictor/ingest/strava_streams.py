"""Récupération des streams Strava (couche ingest, aucune transformation).

Reprise : un Parquet par activité (`{id}.parquet`), écrit immédiatement
après chaque appel. Contrairement à T-04, pas besoin de watermark temporel
— l'`id` de l'activité est une clé stable et suffit : "déjà téléchargée"
= le fichier existe déjà.

Quota : contrôle proactif via les en-têtes X-RateLimit-* après chaque
réponse (comme T-04), avec en plus un arrêt propre — pas une attente —
quand le quota *journalier* est atteint, puisqu'attendre la remise à
zéro n'a pas de sens à l'échelle d'un script interactif.
"""

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from segment_predictor.ingest.strava_client import (
    DEFAULT_RATE_LIMIT_BACKOFF_S,
    authenticated_get,
    parse_rate_limit,
)

ELIGIBLE_ACTIVITY_TYPES = ("Ride", "VirtualRide")
STREAM_TYPES = ("time", "watts", "altitude", "latlng", "distance", "heartrate", "cadence")


def list_eligible_activity_ids(activities_raw_dir: Path) -> list[int]:
    """IDs des activités Ride/VirtualRide avec un vrai capteur de puissance (`device_watts`)."""
    dataset = ds.dataset(activities_raw_dir, format="parquet")
    rows = dataset.to_table(columns=["id", "type", "device_watts"]).to_pylist()
    return [
        row["id"]
        for row in rows
        if row["type"] in ELIGIBLE_ACTIVITY_TYPES and row["device_watts"] is True
    ]


def _fetch_streams_response(
    http_client: httpx.Client,
    access_token: str,
    activity_id: int,
    sleep: Callable[[float], None],
) -> httpx.Response:
    return authenticated_get(
        http_client,
        f"/activities/{activity_id}/streams",
        access_token,
        params={"keys": ",".join(STREAM_TYPES), "key_by_type": "true"},
        sleep=sleep,
    )


def get_activity_streams(
    http_client: httpx.Client,
    access_token: str,
    activity_id: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """GET /activities/{id}/streams, JSON brut tel que renvoyé par Strava (key_by_type=true)."""
    return _fetch_streams_response(http_client, access_token, activity_id, sleep).json()


def save_streams(streams_raw_dir: Path, activity_id: int, streams: dict) -> Path:
    """Écrit les streams d'UNE activité dans son propre Parquet, forme brute inchangée."""
    streams_raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = streams_raw_dir / f"{activity_id}.parquet"
    table = pa.Table.from_pylist([streams])
    pq.write_table(table, file_path)
    return file_path


def _already_downloaded_ids(streams_raw_dir: Path) -> set[int]:
    if not streams_raw_dir.exists():
        return set()
    return {int(path.stem) for path in streams_raw_dir.glob("*.parquet")}


@dataclass(frozen=True)
class FetchStreamsSummary:
    """Bilan d'un lancement : ce qui a été fait, ce qu'il reste, et pourquoi ça s'est arrêté."""

    fetched_activity_ids: list[int]
    already_downloaded_activity_ids: list[int]
    remaining_activity_ids: list[int]
    stopped_due_to_daily_quota: bool


def fetch_and_store_streams(
    http_client: httpx.Client,
    access_token: str,
    activities_raw_dir: Path,
    streams_raw_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchStreamsSummary:
    """Télécharge les streams des activités éligibles pas encore sur disque.

    Reprenable par construction (le fichier déjà présent = déjà fait) et
    s'arrête proprement dès que le quota journalier Strava est atteint,
    en annonçant ce qui reste via le résumé renvoyé.
    """
    eligible_ids = list_eligible_activity_ids(activities_raw_dir)
    already_downloaded_ids = _already_downloaded_ids(streams_raw_dir)
    to_fetch = [
        activity_id for activity_id in eligible_ids if activity_id not in already_downloaded_ids
    ]

    fetched_ids: list[int] = []
    for position, activity_id in enumerate(to_fetch):
        try:
            response = _fetch_streams_response(http_client, access_token, activity_id, sleep)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                # Filet de sécurité : le contrôle proactif ci-dessous n'a pas
                # suffi (Strava n'a pas renvoyé d'en-têtes de quota exploitables
                # sur cette réponse). Après MAX_RATE_LIMIT_RETRIES tentatives
                # espacées par le backoff par défaut (~75 min au total), un 429
                # qui persiste est presque certainement le quota journalier
                # (le quota court terme, lui, serait déjà retombé) : même
                # traitement, arrêt propre plutôt que de laisser planter.
                return FetchStreamsSummary(
                    fetched_activity_ids=fetched_ids,
                    already_downloaded_activity_ids=sorted(already_downloaded_ids),
                    remaining_activity_ids=to_fetch[position:],
                    stopped_due_to_daily_quota=True,
                )
            raise

        save_streams(streams_raw_dir, activity_id, response.json())
        fetched_ids.append(activity_id)

        rate_limit = parse_rate_limit(response)
        if rate_limit is not None:
            if rate_limit.usage_daily >= rate_limit.limit_daily:
                return FetchStreamsSummary(
                    fetched_activity_ids=fetched_ids,
                    already_downloaded_activity_ids=sorted(already_downloaded_ids),
                    remaining_activity_ids=to_fetch[position + 1 :],
                    stopped_due_to_daily_quota=True,
                )
            if rate_limit.short_term_exhausted:
                sleep(DEFAULT_RATE_LIMIT_BACKOFF_S)

    return FetchStreamsSummary(
        fetched_activity_ids=fetched_ids,
        already_downloaded_activity_ids=sorted(already_downloaded_ids),
        remaining_activity_ids=[],
        stopped_due_to_daily_quota=False,
    )


def ensure_path_is_gitignored(path: Path, project_root: Path) -> None:
    """Vérifie via `git check-ignore` que `path` est bien filtré avant d'y écrire.

    Ceinture et bretelles : ne fait pas confiance au .gitignore "de mémoire",
    le vérifie réellement avant d'écrire des données personnelles sur disque.
    """
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=project_root,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{path} n'est pas gitignoré : ajoute-le à .gitignore avant de continuer."
        )
