"""Récupération incrémentale des activités Strava (couche ingest, aucune transformation).

Chargement incrémental : le watermark n'est pas stocké à part, il est
recalculé à chaque lancement à partir des données déjà écrites sur
disque (max de `start_date` dans les Parquet existants). Avantage :
pas de fichier d'état séparé qui pourrait se désynchroniser du Parquet
réel. Le dédoublonnage par `id` en plus du watermark protège contre le
cas où le filtre `after` de Strava serait inclusif à la frontière.
"""

import time
from collections.abc import Callable
from datetime import UTC, datetime
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

PER_PAGE = 200


def list_activities(
    http_client: httpx.Client,
    access_token: str,
    after: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """GET /athlete/activities, pagination jusqu'à épuisement.

    `after` : timestamp epoch, ne renvoie que les activités postérieures
    (watermark incrémental, voir read_watermark).
    """
    activities: list[dict] = []
    page = 1
    while True:
        params: dict[str, int] = {"per_page": PER_PAGE, "page": page}
        if after is not None:
            params["after"] = after

        response = authenticated_get(
            http_client, "/athlete/activities", access_token, params=params, sleep=sleep
        )

        # Lu à chaque réponse : si le quota court terme est déjà au max,
        # on marque une pause avant la page suivante plutôt que d'attendre le 429.
        rate_limit = parse_rate_limit(response)
        if rate_limit is not None and rate_limit.short_term_exhausted:
            sleep(DEFAULT_RATE_LIMIT_BACKOFF_S)

        batch = response.json()
        activities.extend(batch)
        if len(batch) < PER_PAGE:
            break
        page += 1

    return activities


def _parse_strava_datetime(value: str) -> int:
    """ "2024-05-01T10:15:00Z" -> epoch secondes (format `start_date` de Strava)."""
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _existing_dataset(raw_dir: Path) -> ds.Dataset | None:
    if not raw_dir.exists() or not any(raw_dir.glob("*.parquet")):
        return None
    return ds.dataset(raw_dir, format="parquet")


def read_watermark(raw_dir: Path) -> int | None:
    """Epoch du `start_date` le plus récent déjà stocké, ou None si rien n'existe encore."""
    dataset = _existing_dataset(raw_dir)
    if dataset is None:
        return None
    start_dates = dataset.to_table(columns=["start_date"]).column("start_date").to_pylist()
    if not start_dates:
        return None
    return _parse_strava_datetime(max(start_dates))


def read_stored_activity_ids(raw_dir: Path) -> set[int]:
    """IDs déjà présents dans les Parquet existants, pour le dédoublonnage."""
    dataset = _existing_dataset(raw_dir)
    if dataset is None:
        return set()
    return set(dataset.to_table(columns=["id"]).column("id").to_pylist())


def save_activities(
    raw_dir: Path, activities: list[dict], run_time: datetime | None = None
) -> Path | None:
    """Écrit les activités dans un nouveau Parquet daté. None si la liste est vide."""
    if not activities:
        return None
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_time = run_time or datetime.now(UTC)
    # Précision à la microseconde : deux lancements rapprochés (ou, comme
    # ici, deux runs dans le même test) ne doivent pas se retrouver avec
    # le même nom de fichier et s'écraser l'un l'autre.
    file_path = raw_dir / f"activities_{run_time:%Y%m%dT%H%M%S%f}.parquet"
    table = pa.Table.from_pylist(activities)
    pq.write_table(table, file_path)
    return file_path


def fetch_and_store_new_activities(
    http_client: httpx.Client,
    access_token: str,
    raw_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
    run_time: datetime | None = None,
) -> Path | None:
    """Récupère les activités postérieures au watermark local et les stocke.

    Relançable sans doublon : le watermark limite déjà ce qui est
    redemandé à Strava, et le filtre par `id` protège contre tout
    chevauchement résiduel à la frontière.
    """
    after = read_watermark(raw_dir)
    fetched = list_activities(http_client, access_token, after=after, sleep=sleep)

    already_stored_ids = read_stored_activity_ids(raw_dir)
    new_activities = [activity for activity in fetched if activity["id"] not in already_stored_ids]

    return save_activities(raw_dir, new_activities, run_time=run_time)
