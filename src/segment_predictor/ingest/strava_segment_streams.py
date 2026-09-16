"""Récupération du profil réel d'un segment (couche ingest, T-51a).

`GET /segments/{id}/streams` renvoie distance/altitude/latlng le long du
TRACÉ OFFICIEL du segment — différent d'`ingest/strava_streams.py`
(`GET /activities/{id}/streams`), qui porte sur UNE sortie précise faite
PAR MOI. Ce fichier-ci n'est jamais spécifique à un athlète : le profil
d'un segment est le même pour tout le monde (contrairement aux dossiers
bruts `data/raw/<source>/<user_id>/`, T-44d, pour toutes les autres
sources personnelles) — stocké à plat, comme open_meteo.

But : remplacer l'unique `average_grade` (segments.average_grade) par un
vrai relief tronçon par tronçon (T-51c), pour des calculs de puissance
qui ne supposent plus une pente uniforme sur toute la longueur — trouvé
nécessaire en testant KomInfo.power_w en vrai (T-51, un segment à pente
moyenne 4.8% mais probablement très irrégulière donnait une puissance
requise irréaliste dans les deux sens).

Reprise, quota : même mécanique que T-05/T-06 (fetch_and_store_streams,
fetch_and_store_segments) — pas réinventée ici.
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

SEGMENT_STREAM_TYPES = ("distance", "altitude", "latlng")


def _fetch_segment_streams_response(
    http_client: httpx.Client,
    access_token: str,
    segment_id: int,
    sleep: Callable[[float], None],
) -> httpx.Response:
    return authenticated_get(
        http_client,
        f"/segments/{segment_id}/streams",
        access_token,
        params={"keys": ",".join(SEGMENT_STREAM_TYPES), "key_by_type": "true"},
        sleep=sleep,
    )


def get_segment_streams(
    http_client: httpx.Client,
    access_token: str,
    segment_id: int,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """GET /segments/{id}/streams, JSON brut tel que renvoyé par Strava (key_by_type=true)."""
    return _fetch_segment_streams_response(http_client, access_token, segment_id, sleep).json()


def save_segment_streams(raw_dir: Path, segment_id: int, streams: dict) -> Path:
    """Écrit le profil d'UN segment dans son propre Parquet, forme brute inchangée."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / f"{segment_id}.parquet"
    pq.write_table(pa.Table.from_pylist([streams]), file_path)
    return file_path


def _already_downloaded_ids(raw_dir: Path) -> set[int]:
    if not raw_dir.exists():
        return set()
    return {int(path.stem) for path in raw_dir.glob("*.parquet")}


@dataclass(frozen=True)
class FetchSegmentStreamsSummary:
    """Bilan d'un lancement : ce qui a été fait, ce qu'il reste, et pourquoi ça s'est arrêté."""

    fetched_ids: list[int]
    already_downloaded_ids: list[int]
    remaining_ids: list[int]
    stopped_due_to_daily_quota: bool


def fetch_and_store_segment_streams(
    http_client: httpx.Client,
    access_token: str,
    segment_ids: list[int],
    raw_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchSegmentStreamsSummary:
    """Récupère le profil des segments demandés qui ne sont pas encore sur disque.

    Même logique de quota que fetch_and_store_segments (T-06) : pause sur
    quota court terme, arrêt propre sur quota journalier ou 429 persistant
    — ce quota est PARTAGÉ entre tous les utilisateurs de l'app (T-46),
    pas question de le gaspiller en retéléchargeant un profil déjà connu.
    """
    already_downloaded_ids = _already_downloaded_ids(raw_dir)
    to_fetch = [
        segment_id for segment_id in segment_ids if segment_id not in already_downloaded_ids
    ]

    fetched_ids: list[int] = []
    for position, segment_id in enumerate(to_fetch):
        try:
            response = _fetch_segment_streams_response(http_client, access_token, segment_id, sleep)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                return FetchSegmentStreamsSummary(
                    fetched_ids=fetched_ids,
                    already_downloaded_ids=sorted(already_downloaded_ids),
                    remaining_ids=to_fetch[position:],
                    stopped_due_to_daily_quota=True,
                )
            raise

        save_segment_streams(raw_dir, segment_id, response.json())
        fetched_ids.append(segment_id)

        rate_limit = parse_rate_limit(response)
        if rate_limit is not None:
            if rate_limit.usage_daily >= rate_limit.limit_daily:
                return FetchSegmentStreamsSummary(
                    fetched_ids=fetched_ids,
                    already_downloaded_ids=sorted(already_downloaded_ids),
                    remaining_ids=to_fetch[position + 1 :],
                    stopped_due_to_daily_quota=True,
                )
            if rate_limit.short_term_exhausted:
                sleep(DEFAULT_RATE_LIMIT_BACKOFF_S)

    return FetchSegmentStreamsSummary(
        fetched_ids=fetched_ids,
        already_downloaded_ids=sorted(already_downloaded_ids),
        remaining_ids=[],
        stopped_due_to_daily_quota=False,
    )
