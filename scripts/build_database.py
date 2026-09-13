"""T-07 : (re)construit tout le schéma DuckDB à partir du Parquet déjà ingéré.

Aucun appel réseau — uniquement de la lecture/transformation locale.
Relançable à volonté pour un utilisateur donné : ses tables personnelles
(activities, streams, segment_efforts, wellness) sont remplacées SANS
toucher aux autres utilisateurs déjà en base (T-44b, DELETE+INSERT par
`user_id`, plus un CREATE OR REPLACE qui effacerait tout le monde) ;
`segments` reste partagée (pas encore de notion de favoris par
utilisateur, T-44c) et `activity_weather` se reconstruit pour tout le
monde à chaque appel (elle relit `activities` en entier, cf sa docstring).

`user_id` : requis, l'id athlète Strava de la personne dont on (re)lit
les fichiers de `data/raw/` — tant que T-44d (chemins bruts par
utilisateur) n'est pas fait, ces dossiers restent partagés, donc un seul
`user_id` a un sens à la fois.

Usage : uv run python scripts/build_database.py <user_id>
"""

import sys
from pathlib import Path

import duckdb

from segment_predictor.storage.activities import build_activities_table
from segment_predictor.storage.raw_views import create_raw_views
from segment_predictor.storage.segment_efforts import build_segment_efforts_table
from segment_predictor.storage.segments import build_segments_table
from segment_predictor.storage.streams import build_streams_table
from segment_predictor.storage.weather import build_activity_weather_table
from segment_predictor.storage.wellness import build_wellness_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVITIES_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_activities"
STREAMS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_streams"
SEGMENTS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_segments"
WEATHER_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo"
ACTIVITY_DETAILS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "strava_activity_details"
WELLNESS_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "intervals_icu"
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage : uv run python scripts/build_database.py <user_id>")
        raise SystemExit(1)
    user_id = int(sys.argv[1])

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
            WELLNESS_RAW_DIR,
        )
        build_activities_table(conn, ACTIVITIES_RAW_DIR, user_id=user_id)
        build_streams_table(conn, STREAMS_RAW_DIR, user_id=user_id)
        build_segments_table(conn, SEGMENTS_RAW_DIR)
        # après activities : a besoin de main.activities pour savoir quelles
        # zones/dates interpoler. Pas de user_id ici (T-44b) : reconstruit
        # pour tous les utilisateurs déjà en base, cf sa docstring.
        build_activity_weather_table(conn, WEATHER_RAW_DIR)
        build_segment_efforts_table(conn, ACTIVITY_DETAILS_RAW_DIR, user_id=user_id)
        build_wellness_table(conn, WELLNESS_RAW_DIR, user_id=user_id)

        tables = (
            "activities",
            "streams",
            "segments",
            "activity_weather",
            "segment_efforts",
            "wellness",
        )
        for table in tables:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"main.{table}: {count} lignes")
    finally:
        conn.close()

    print(f"Base reconstruite dans {DUCKDB_PATH}")


if __name__ == "__main__":
    main()
