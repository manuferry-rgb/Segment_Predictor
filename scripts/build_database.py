"""T-07 : (re)construit tout le schéma DuckDB à partir du Parquet déjà ingéré.

Aucun appel réseau — uniquement de la lecture/transformation locale.
Relançable à volonté pour un utilisateur donné : ses tables personnelles
(activities, streams, segment_efforts, wellness, user_segment_stats,
user_starred_segments) sont remplacées SANS toucher aux autres
utilisateurs déjà en base (T-44b/T-44c, DELETE+INSERT par `user_id`,
plus un CREATE OR REPLACE qui effacerait tout le monde) ; `segments`
reste partagée (lue depuis TOUS les sous-dossiers utilisateur, T-44d) et
`activity_weather` se reconstruit pour tout le monde à chaque appel
(elle relit `activities` en entier, cf sa docstring).

`user_id` : requis, l'id athlète Strava de la personne dont on (re)lit
les fichiers de `data/raw/<source>/<user_id>/` (T-44d — un sous-dossier
par utilisateur, sinon (re)lire pour l'un écraserait les fichiers de
l'autre). Note sur `raw.*` (create_raw_views, exploration seulement,
jamais utilisée par du code testé) : chaque vue ne reflète que LE
DERNIER utilisateur pour lequel ce script a tourné, pas tout le monde
combiné — limite assumée, pas cachée.

Usage : uv run python scripts/build_database.py <user_id>
"""

import sys
from pathlib import Path

import duckdb

from segment_predictor.storage.activities import build_activities_table
from segment_predictor.storage.raw_views import create_raw_views
from segment_predictor.storage.segment_efforts import build_segment_efforts_table
from segment_predictor.storage.segments import (
    build_segments_table,
    build_user_segment_stats_table,
    build_user_starred_segments_table,
)
from segment_predictor.storage.streams import build_streams_table
from segment_predictor.storage.weather import build_activity_weather_table
from segment_predictor.storage.wellness import build_wellness_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVITIES_RAW_DIR_ROOT = PROJECT_ROOT / "data" / "raw" / "strava_activities"
STREAMS_RAW_DIR_ROOT = PROJECT_ROOT / "data" / "raw" / "strava_streams"
SEGMENTS_RAW_DIR_ROOT = PROJECT_ROOT / "data" / "raw" / "strava_segments"
# Pas de sous-dossier par utilisateur ici (T-44d) : une zone météo (grille
# 0.1°) est identique pour tout le monde, contrairement aux 5 autres
# sources — voir build_activity_weather_table.
WEATHER_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo"
ACTIVITY_DETAILS_RAW_DIR_ROOT = PROJECT_ROOT / "data" / "raw" / "strava_activity_details"
WELLNESS_RAW_DIR_ROOT = PROJECT_ROOT / "data" / "raw" / "intervals_icu"
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage : uv run python scripts/build_database.py <user_id>")
        raise SystemExit(1)
    user_id_str = sys.argv[1]
    user_id = int(user_id_str)

    activities_raw_dir = ACTIVITIES_RAW_DIR_ROOT / user_id_str
    streams_raw_dir = STREAMS_RAW_DIR_ROOT / user_id_str
    segments_raw_dir = SEGMENTS_RAW_DIR_ROOT / user_id_str
    activity_details_raw_dir = ACTIVITY_DETAILS_RAW_DIR_ROOT / user_id_str
    wellness_raw_dir = WELLNESS_RAW_DIR_ROOT / user_id_str

    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DUCKDB_PATH))
    try:
        create_raw_views(
            conn,
            activities_raw_dir,
            streams_raw_dir,
            segments_raw_dir,
            WEATHER_RAW_DIR,
            activity_details_raw_dir,
            wellness_raw_dir,
        )
        build_activities_table(conn, activities_raw_dir, user_id=user_id)
        build_streams_table(conn, streams_raw_dir, user_id=user_id)
        # SEGMENTS_RAW_DIR_ROOT (parent, pas segments_raw_dir) : segments
        # est partagée, lit TOUS les sous-dossiers utilisateur (T-44d).
        build_segments_table(conn, SEGMENTS_RAW_DIR_ROOT)
        build_user_segment_stats_table(conn, segments_raw_dir, user_id=user_id)
        build_user_starred_segments_table(conn, segments_raw_dir, user_id=user_id)
        # après activities : a besoin de main.activities pour savoir quelles
        # zones/dates interpoler. Pas de user_id ici (T-44b) : reconstruit
        # pour tous les utilisateurs déjà en base, cf sa docstring.
        build_activity_weather_table(conn, WEATHER_RAW_DIR)
        build_segment_efforts_table(conn, activity_details_raw_dir, user_id=user_id)
        build_wellness_table(conn, wellness_raw_dir, user_id=user_id)

        tables = (
            "activities",
            "streams",
            "segments",
            "user_segment_stats",
            "user_starred_segments",
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
