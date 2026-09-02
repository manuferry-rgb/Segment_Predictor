"""Vues DuckDB `raw.*` : passthrough SQL sur le Parquet brut d'ingest.

Ce sont des VUES, pas des tables : aucune donnée n'est dupliquée dans le
fichier DuckDB, chaque requête relit directement le Parquet — cohérent
avec la taille du dataset (quelques dizaines de Mo) et avec la règle
"ingest ne transforme rien" : ces vues ne font qu'exposer le JSON brut
en SQL, sans caster ni renommer quoi que ce soit. `union_by_name=true`
gère le cas où plusieurs fichiers (ex. plusieurs lancements de T-04)
auraient des colonnes différentes.
"""

from pathlib import Path

import duckdb


def create_raw_views(
    conn: duckdb.DuckDBPyConnection,
    activities_raw_dir: Path,
    streams_raw_dir: Path,
    segments_raw_dir: Path,
    weather_raw_dir: Path | None = None,
) -> None:
    """(Re)crée le schéma `raw` et ses vues, chacune sur son dossier d'ingest.

    `weather_raw_dir` est optionnel (T-14 est postérieur à T-07) : la vue
    `raw.weather` n'est créée que si fourni.

    Un dossier pas encore peuplé (ex. `fetch_weather.py` pas encore lancé)
    ne fait pas planter la construction : `read_parquet` vérifie le glob
    dès la création de la vue (pas paresseusement à la requête, contre-
    intuitif pour une VIEW) — on saute juste cette vue plutôt que de faire
    échouer tout `build_database.py` pour une source pas encore ingérée.
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")

    sources = {
        "activities": activities_raw_dir,
        "streams": streams_raw_dir,
        "segments": segments_raw_dir,
    }
    if weather_raw_dir is not None:
        sources["weather"] = weather_raw_dir
    for view_name, raw_dir in sources.items():
        if not raw_dir.exists() or not any(raw_dir.glob("*.parquet")):
            continue
        glob_pattern = (raw_dir / "*.parquet").as_posix()
        conn.execute(
            f"CREATE OR REPLACE VIEW raw.{view_name} AS "
            f"SELECT * FROM read_parquet('{glob_pattern}', union_by_name = true)"
        )
