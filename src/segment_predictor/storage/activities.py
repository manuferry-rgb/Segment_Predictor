"""Construction de la table DuckDB `main.activities` à partir du JSON brut Strava.

Sélection curée : les ~59 champs bruts de Strava ne sont pas tous
recopiés, seulement ceux utiles à la prédiction (identité, durée,
distance, dénivelé, puissance/FC/cadence moyennes). Le reste (kudos,
athlete, polyline de la carte...) reste consultable via raw.activities.

Champs "structurels" (id, name, type, dates, distance, durées) requis
sans défaut — une activité Strava en a toujours. Les métriques capteur
(watts, FC, cadence) peuvent légitimement être NULL (ex. une randonnée
n'a pas de watts) : `.get()`, pas d'exception.
"""

from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def _parse_start_date(value: str) -> datetime:
    """ "2024-05-01T10:15:00Z" -> datetime naïf (UTC implicite).

    `start_date` est toujours en UTC côté Strava (contrairement à
    `start_date_local`). On enlève le tzinfo après parsing plutôt que de
    stocker un TIMESTAMPTZ : DuckDB a besoin de `pytz` pour relire une
    colonne timezone-aware, une dépendance en plus pour rien puisque le
    fuseau est déjà implicite et constant ici.
    """
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _extract_start_latlng(raw_activity: dict) -> tuple[float | None, float | None]:
    """`start_latlng` vaut `[]` (pas absent) pour une activité indoor sans GPS."""
    start_latlng = raw_activity.get("start_latlng")
    if not start_latlng:
        return None, None
    lat, lng = start_latlng
    return lat, lng


def _activity_to_row(raw_activity: dict, user_id: int) -> dict:
    start_lat, start_lng = _extract_start_latlng(raw_activity)
    return {
        "id": raw_activity["id"],
        "name": raw_activity["name"],
        "type": raw_activity["type"],
        "sport_type": raw_activity["sport_type"],
        "start_date": _parse_start_date(raw_activity["start_date"]),
        "distance_m": raw_activity["distance"],
        "moving_time_s": raw_activity["moving_time"],
        "elapsed_time_s": raw_activity["elapsed_time"],
        "total_elevation_gain_m": raw_activity["total_elevation_gain"],
        "average_watts": raw_activity.get("average_watts"),
        "device_watts": raw_activity.get("device_watts"),
        "average_heartrate": raw_activity.get("average_heartrate"),
        "max_heartrate": raw_activity.get("max_heartrate"),
        "average_cadence": raw_activity.get("average_cadence"),
        "start_lat": start_lat,
        "start_lng": start_lng,
        "user_id": user_id,
    }


def build_activities_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path, user_id: int) -> None:
    """Lit toutes les activités brutes de `raw_dir` (potentiellement plusieurs
    fichiers datés, un par lancement de T-04) et les rattache à `user_id`.

    DELETE puis INSERT (T-44b, multi-utilisateur), pas CREATE OR REPLACE
    comme avant : un CREATE OR REPLACE effacerait aussi les activités déjà
    en base pour les AUTRES utilisateurs — `raw_dir` ne contient que les
    activités de CET utilisateur (T-44d), la table, elle, en contient
    plusieurs. `CREATE TABLE IF NOT EXISTS ... WHERE FALSE` ne crée le
    schéma que la toute première fois (base neuve) ; sinon la table
    existe déjà avec sa colonne `user_id`.
    """
    rows = [
        _activity_to_row(raw_activity, user_id)
        for path in sorted(raw_dir.glob("*.parquet"))
        for raw_activity in pq.read_table(path).to_pylist()
    ]

    activities_table = pa.Table.from_pylist(rows)
    conn.register("activities_table", activities_table)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS activities AS SELECT * FROM activities_table WHERE FALSE"
        )
        conn.execute("DELETE FROM activities WHERE user_id = ?", [user_id])
        conn.execute("INSERT INTO activities SELECT * FROM activities_table")
    finally:
        conn.unregister("activities_table")
