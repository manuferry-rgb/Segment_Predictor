"""Construction de la table DuckDB `main.streams` à partir des streams bruts.

Transformation : dépivotage. Le brut stocke 1 ligne par activité avec des
colonnes-listes (une par type de stream) ; `streams` stocke 1 ligne par
échantillon temporel (format long), plus pratique pour les modèles
physiques qui traitent le profil d'une activité point par point.

Hypothèse vérifiée sur les 497 activités réelles avant d'écrire ce
module : tous les streams présents pour une activité ont la même
longueur que `time` (alignement 1:1, comme documenté par l'API Strava).
Si un fichier futur ne respecte pas ça, on lève plutôt que de tronquer
ou décaler silencieusement les points.
"""

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# distance_m/heartrate/cadence/watts/altitude_m : nom de colonne -> clé brute Strava
_SIMPLE_STREAM_KEYS = {
    "watts": "watts",
    "altitude_m": "altitude",
    "distance_m": "distance",
    "heartrate": "heartrate",
    "cadence": "cadence",
}
_COLUMN_NAMES = (
    "activity_id",
    "sample_index",
    "t_s",
    "watts",
    "altitude_m",
    "lat",
    "lng",
    "distance_m",
    "heartrate",
    "cadence",
)


def _activity_streams_to_columns(activity_id: int, raw_streams: dict) -> dict[str, list]:
    """Un JSON brut de streams (1 activité) -> des colonnes alignées, format long."""
    time_stream = raw_streams.get("time")
    if time_stream is None:
        raise ValueError(f"activité {activity_id} : stream 'time' absent, impossible de dépivoter")
    time_data = time_stream["data"]
    n = len(time_data)

    def _aligned_column(column_name: str, raw_key: str) -> list:
        stream = raw_streams.get(raw_key)
        if stream is None:
            return [None] * n
        data = stream["data"]
        if len(data) != n:
            raise ValueError(
                f"activité {activity_id} : stream '{raw_key}' a {len(data)} points, "
                f"'time' en a {n} — alignement cassé, refus de dépivoter"
            )
        return data

    latlng_stream = raw_streams.get("latlng")
    if latlng_stream is None:
        lat_col: list = [None] * n
        lng_col: list = [None] * n
    else:
        points = latlng_stream["data"]
        if len(points) != n:
            raise ValueError(
                f"activité {activity_id} : stream 'latlng' a {len(points)} points, "
                f"'time' en a {n} — alignement cassé, refus de dépivoter"
            )
        lat_col = [point[0] for point in points]
        lng_col = [point[1] for point in points]

    columns: dict[str, list] = {
        "activity_id": [activity_id] * n,
        "sample_index": list(range(n)),
        "t_s": time_data,
        "lat": lat_col,
        "lng": lng_col,
    }
    for column_name, raw_key in _SIMPLE_STREAM_KEYS.items():
        columns[column_name] = _aligned_column(column_name, raw_key)
    return columns


def build_streams_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Lit tous les streams bruts de `raw_dir` et (re)crée la table `streams`.

    Construction colonnaire (plutôt qu'une liste de dicts par ligne) :
    à l'échelle réelle du projet (~2,15M échantillons), c'est nettement
    moins coûteux en mémoire et en temps que `Table.from_pylist` sur
    autant de lignes.
    """
    columns: dict[str, list] = {name: [] for name in _COLUMN_NAMES}

    for path in sorted(raw_dir.glob("*.parquet")):
        activity_id = int(path.stem)
        raw_streams = pq.read_table(path).to_pylist()[0]
        activity_columns = _activity_streams_to_columns(activity_id, raw_streams)
        for name in _COLUMN_NAMES:
            columns[name].extend(activity_columns[name])

    streams_table = pa.table(columns)
    conn.register("streams_table", streams_table)
    try:
        conn.execute("CREATE OR REPLACE TABLE streams AS SELECT * FROM streams_table")
    finally:
        conn.unregister("streams_table")
