"""Construction de la table DuckDB `main.segment_streams` (T-51b).

Même dépivotage que storage/streams.py (T-07), appliqué au profil OFFICIEL
d'un segment (ingest/strava_segment_streams.py, T-51a) plutôt qu'au flux
d'une activité précise :
- pas de `time` : un segment n'a pas d'horodatage propre, le profil est
  indexé par la distance parcourue depuis le départ (`distance`, toujours
  présent — sert d'ancre pour la longueur `n`, comme `time` pour
  build_streams_table).
- pas de `user_id` : table PARTAGÉE (le tracé d'un segment est identique
  pour tout le monde, même raisonnement que build_segments_table, T-44d)
  — CREATE OR REPLACE, pas DELETE+INSERT par utilisateur.
"""

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

_COLUMN_NAMES = ("segment_id", "sample_index", "distance_m", "altitude_m", "lat", "lng")


def _segment_streams_to_columns(segment_id: int, raw_streams: dict) -> dict[str, list]:
    """Un JSON brut de profil de segment -> des colonnes alignées, format long."""
    distance_stream = raw_streams.get("distance")
    if distance_stream is None:
        raise ValueError(
            f"segment {segment_id} : stream 'distance' absent, impossible de dépivoter"
        )
    distance_data = distance_stream["data"]
    n = len(distance_data)

    def _aligned_column(column_name: str, raw_key: str) -> list:
        stream = raw_streams.get(raw_key)
        if stream is None:
            return [None] * n
        data = stream["data"]
        if len(data) != n:
            raise ValueError(
                f"segment {segment_id} : stream '{raw_key}' a {len(data)} points, "
                f"'distance' en a {n} — alignement cassé, refus de dépivoter"
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
                f"segment {segment_id} : stream 'latlng' a {len(points)} points, "
                f"'distance' en a {n} — alignement cassé, refus de dépivoter"
            )
        lat_col = [point[0] for point in points]
        lng_col = [point[1] for point in points]

    return {
        "segment_id": [segment_id] * n,
        "sample_index": list(range(n)),
        "distance_m": distance_data,
        "altitude_m": _aligned_column("altitude_m", "altitude"),
        "lat": lat_col,
        "lng": lng_col,
    }


def build_segment_streams_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Lit tous les profils bruts de `raw_dir` (table PARTAGÉE, pas de sous-dossier
    par utilisateur — voir ingest/strava_segment_streams.py) et (re)crée
    `segment_streams`.

    CREATE OR REPLACE reste correct ici (comme build_segments_table,
    T-44d) : ces lignes sont des faits physiques identiques pour tout le
    monde, il n'y a pas de lignes "d'un autre utilisateur" à préserver.
    """
    columns: dict[str, list] = {name: [] for name in _COLUMN_NAMES}

    for path in sorted(raw_dir.glob("*.parquet")):
        segment_id = int(path.stem)
        raw_streams = pq.read_table(path).to_pylist()[0]
        segment_columns = _segment_streams_to_columns(segment_id, raw_streams)
        for name in _COLUMN_NAMES:
            columns[name].extend(segment_columns[name])

    segment_streams_table = pa.table(columns)
    conn.register("segment_streams_table", segment_streams_table)
    try:
        conn.execute(
            "CREATE OR REPLACE TABLE segment_streams AS SELECT * FROM segment_streams_table"
        )
    finally:
        conn.unregister("segment_streams_table")
