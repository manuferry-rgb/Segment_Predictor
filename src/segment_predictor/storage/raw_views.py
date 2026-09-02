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
) -> None:
    """(Re)crée le schéma `raw` et ses 3 vues, chacune sur son dossier d'ingest."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")

    sources = {
        "activities": activities_raw_dir,
        "streams": streams_raw_dir,
        "segments": segments_raw_dir,
    }
    for view_name, raw_dir in sources.items():
        glob_pattern = (raw_dir / "*.parquet").as_posix()
        conn.execute(
            f"CREATE OR REPLACE VIEW raw.{view_name} AS "
            f"SELECT * FROM read_parquet('{glob_pattern}', union_by_name = true)"
        )
