"""Construction de la table DuckDB `segment_efforts` à partir des activités
détaillées Strava (T-07b).

Extrait TOUS les efforts trouvés dans `segment_efforts` (Strava matche
automatiquement tous les segments publics croisés), pas seulement ceux
suivis dans `main.segments` — cette table reste générale ; la jointure
vers les segments qu'on suit se fait en aval (T-16).
"""

from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def _parse_start_date(value: str) -> datetime:
    """ "2024-07-16T05:34:37Z" -> datetime naïf (UTC implicite), même convention que T-07."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _effort_to_row(effort: dict) -> dict:
    """Un effort brut (élément de `segment_efforts`) -> une ligne de la table.

    Champs structurels (id, segment.id, activity.id, temps, distance)
    requis sans défaut. `average_watts`/`average_heartrate`/`pr_rank`/
    `kom_rank` peuvent être NULL (pas de capteur ce jour-là, ou pas de
    record) : `.get()`, pas d'exception.
    """
    return {
        "id": effort["id"],
        "segment_id": effort["segment"]["id"],
        "activity_id": effort["activity"]["id"],
        "start_date": _parse_start_date(effort["start_date"]),
        "elapsed_time_s": effort["elapsed_time"],
        "moving_time_s": effort["moving_time"],
        "distance_m": effort["distance"],
        "average_watts": effort.get("average_watts"),
        "device_watts": effort.get("device_watts"),
        "average_heartrate": effort.get("average_heartrate"),
        "pr_rank": effort.get("pr_rank"),
        "kom_rank": effort.get("kom_rank"),
    }


def build_segment_efforts_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Lit toutes les activités détaillées de `raw_dir` et (re)crée `segment_efforts`."""
    rows = [
        _effort_to_row(effort)
        for path in sorted(raw_dir.glob("*.parquet"))
        for detail in pq.read_table(path).to_pylist()
        for effort in (detail.get("segment_efforts") or [])
    ]

    efforts_table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("id", pa.int64()),
                ("segment_id", pa.int64()),
                ("activity_id", pa.int64()),
                ("start_date", pa.timestamp("us")),
                ("elapsed_time_s", pa.int64()),
                ("moving_time_s", pa.int64()),
                ("distance_m", pa.float64()),
                ("average_watts", pa.float64()),
                ("device_watts", pa.bool_()),
                ("average_heartrate", pa.float64()),
                ("pr_rank", pa.int64()),
                ("kom_rank", pa.int64()),
            ]
        ),
    )
    conn.register("segment_efforts_table", efforts_table)
    try:
        conn.execute(
            "CREATE OR REPLACE TABLE segment_efforts AS SELECT * FROM segment_efforts_table"
        )
    finally:
        conn.unregister("segment_efforts_table")
