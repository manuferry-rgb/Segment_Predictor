"""T-07 : (re)construit tout le schéma DuckDB à partir du Parquet déjà ingéré.

Aucun appel réseau — uniquement de la lecture/transformation locale.
Relançable à volonté : chaque table/vue est recréée en entier
(CREATE OR REPLACE), le Parquet brut reste la seule source de vérité.

Usage : uv run python scripts/build_database.py
"""

from pathlib import Path

import duckdb

from segment_predictor.storage.activities import build_activities_table
from segment_predictor.storage.raw_views import create_raw_views
from segment_predictor.storage.segment_efforts import build_segment_efforts_table
from segment_predictor.storage.segments import build_segments_table
from segment_predictor.storage.streams import build_streams_table
from segment_predictor.storage.weather import build_activity_weather_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVITIES_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_activities"
STREAMS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_streams"
SEGMENTS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_segments"
WEATHER_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo"
ACTIVITY_DETAILS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_activity_details"
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"


def main() -> None:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DUCKDB_PATH))
    try:
        create_raw_views(
            conn,
            ACTIVITIES_RAW_DIR,
            STREAMS_RAW_DIR,
            SEGMENTS_RAW_DIR,
            WEATHER_RAW_DIR,
            ACTIVITY_DETAILS_RAW_DIR,
        )
        build_activities_table(conn, ACTIVITIES_RAW_DIR)
        build_streams_table(conn, STREAMS_RAW_DIR)
        build_segments_table(conn, SEGMENTS_RAW_DIR)
        # après activities : a besoin de main.activities pour savoir quelles
        # zones/dates interpoler
        build_activity_weather_table(conn, WEATHER_RAW_DIR)
        build_segment_efforts_table(conn, ACTIVITY_DETAILS_RAW_DIR)

        for table in ("activities", "streams", "segments", "activity_weather", "segment_efforts"):
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"main.{table}: {count} lignes")
    finally:
        conn.close()

    print(f"Base reconstruite dans {DUCKDB_PATH}")


if __name__ == "__main__":
    main()
