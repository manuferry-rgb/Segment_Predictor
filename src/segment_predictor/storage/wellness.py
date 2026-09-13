"""Construction de la table DuckDB `wellness` à partir du JSON brut intervals.icu.

Le brut contient bien plus de champs (ctl, atl, sportInfo, sommeil détaillé,
stress, humeur...) que les 4 demandés par le ticket (T-22 : HRV, sommeil,
FC de repos, poids) — extraits ici, le reste reste accessible via
`raw.wellness` si besoin un jour.

Vérifié en conditions réelles (premier fetch, T-22) : `weight` est bien en
kg (une valeur réelle de 83.0 pour un cycliste adulte — 83 lb serait
absurdement léger), pas de conversion nécessaire. `id` est la date
ISO-8601 du jour. `hrv` et `sleepSecs` n'ont en revanche jamais été
renseignés dans l'historique réel disponible pour vérifier — leur unité
exacte (rMSSD en ms pour hrv, à priori, mais non confirmée par la doc
publique ni par une valeur réelle) reste documentée comme non vérifiée
plutôt que supposée à tort.
"""

from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

_WELLNESS_SCHEMA = pa.schema(
    [
        ("date", pa.date32()),
        ("hrv", pa.float64()),
        ("sleep_s", pa.int64()),
        ("resting_heart_rate_bpm", pa.int64()),
        ("weight_kg", pa.float64()),
        ("user_id", pa.int64()),
    ]
)


def _wellness_to_row(raw: dict, user_id: int) -> dict:
    return {
        "date": date.fromisoformat(raw["id"]),
        "hrv": raw.get("hrv"),
        "sleep_s": raw.get("sleepSecs"),
        "resting_heart_rate_bpm": raw.get("restingHR"),
        "weight_kg": raw.get("weight"),
        "user_id": user_id,
    }


def build_wellness_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path, user_id: int) -> None:
    """(Re)rattache à `user_id` les jours de `wellness.parquet` — vide si
    `fetch_wellness.py` n'a encore jamais tourné pour cet utilisateur
    (même logique que `activity_weather` pour une source optionnelle pas
    encore fetchée) : jamais une erreur, `main.wellness` doit toujours
    exister pour que le reste du schéma puisse la référencer.

    DELETE puis INSERT (T-44b, multi-utilisateur), pas CREATE OR REPLACE
    comme avant — voir la docstring de build_activities_table pour le
    raisonnement complet, identique ici.
    """
    file_path = raw_dir / "wellness.parquet"
    raw_records = pq.read_table(file_path).to_pylist() if file_path.exists() else []
    rows = [_wellness_to_row(record, user_id) for record in raw_records]

    wellness_table = pa.Table.from_pylist(rows, schema=_WELLNESS_SCHEMA)
    conn.register("wellness_table", wellness_table)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS wellness AS SELECT * FROM wellness_table WHERE FALSE"
        )
        conn.execute("DELETE FROM wellness WHERE user_id = ?", [user_id])
        conn.execute("INSERT INTO wellness SELECT * FROM wellness_table")
    finally:
        conn.unregister("wellness_table")
