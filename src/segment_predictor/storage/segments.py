"""Construction de la table DuckDB `segments` à partir des JSON bruts Strava.

C'est ici, pas dans ingest, que le format "mm:ss" / "h:mm:ss" du KOM
(`xoms.kom`) est parsé en secondes. `athlete_segment_stats.pr_elapsed_time`
est en revanche déjà un entier côté Strava — pas de parsing nécessaire.

Aucune valeur par défaut silencieuse pour ce qui est structurel : un
segment dont le JSON brut ne contient pas `xoms.kom` ou
`athlete_segment_stats` fait lever une exception explicite. En revanche
`pr_elapsed_time`/`pr_date` peuvent légitimement être `null` — Strava
renvoie `athlete_segment_stats` avec `effort_count: 0` pour un segment
favori jamais roulé, ce n'est pas une anomalie mais un vrai NULL.
"""

import re
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from segment_predictor.models.segment import bearing_rad

# "4:12" (mm:ss) ou "1:02:35" (h:mm:ss). Le groupe des heures est optionnel.
_DURATION_PATTERN = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})$")
# "53s" : format à part pour les segments sous la minute (pas de ":"),
# trouvé en conditions réelles sur 7 des 75 segments favoris (T-17).
_SECONDS_ONLY_PATTERN = re.compile(r"^(\d+)s$")


def parse_strava_duration(value: str) -> int:
    """ "4:12" -> 252, "1:02:35" -> 3755, "53s" -> 53."""
    seconds_only_match = _SECONDS_ONLY_PATTERN.match(value)
    if seconds_only_match is not None:
        return int(seconds_only_match.group(1))

    match = _DURATION_PATTERN.match(value)
    if match is None:
        raise ValueError(
            f"Format de durée Strava inattendu : {value!r} (attendu 'mm:ss', 'h:mm:ss' ou 'Ns')"
        )
    hours_str, minutes_str, seconds_str = match.groups()
    hours = int(hours_str) if hours_str is not None else 0
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    return hours * 3600 + minutes * 60 + seconds


def _segment_to_row(raw_segment: dict) -> dict:
    """Un JSON brut `GET /segments/{id}` -> une ligne de la table `segments`.

    `athlete_segment_stats` doit exister (sinon la réponse est vraiment
    anormale), mais ses champs `pr_*` peuvent être `None` : Strava renvoie
    ça pour un segment jamais roulé (`effort_count: 0`), ce n'est pas une
    erreur à masquer, juste un PR qui n'existe pas encore.
    """
    stats = raw_segment["athlete_segment_stats"]
    start_lat, start_lng = raw_segment["start_latlng"]
    end_lat, end_lng = raw_segment["end_latlng"]
    return {
        "id": raw_segment["id"],
        "name": raw_segment["name"],
        "distance_m": raw_segment["distance"],
        # Strava renvoie average_grade en POURCENT (0.2 = 0.2%), mais toute
        # la couche models/ (grade de cyclist_power_required, T-10) attend
        # une fraction rise/run (0.002) — convention découverte tardivement
        # (T-16), en branchant enfin cette colonne sur la physique : un
        # 0.2% traité comme 20% donnait un temps prédit ~5x trop long.
        "average_grade": raw_segment["average_grade"] / 100.0,
        # Dénivelé positif (T-31), en mètres, tel que renvoyé par Strava —
        # pas de conversion d'unité ici, contrairement à average_grade.
        # Pas de champ symétrique pour le dénivelé négatif : Strava ne le
        # fournit pas au niveau segment (vérifié sur un payload réel), et
        # `elevation_high`/`elevation_low` ne suffisent pas à le
        # reconstruire (un profil qui monte et descend plusieurs fois n'est
        # pas déterminé par son seul min/max) — pas de valeur inventée.
        "total_elevation_gain_m": raw_segment["total_elevation_gain"],
        # Tracé encodé (format Google Polyline) du segment, tel quel — pas
        # décodé ici, `storage` n'extrait que des champs structurels, pas
        # de géométrie (T-32, préparatoire : servira à calculer un cap RÉEL
        # par tronçon plutôt que l'unique cap moyen start->end de
        # heading_rad, pour projeter le vent correctement sur tout le
        # segment, pas seulement à vol d'oiseau).
        "polyline": raw_segment["map"]["polyline"],
        # Cap GLOBAL du segment (T-27), pas tronçon par tronçon : on n'a
        # que start_latlng/end_latlng ici, pas de profil détaillé au niveau
        # segment (T-07b). Nécessaire pour que le vent (T-15) ait un sens
        # dans l'approximation "un seul tronçon" (T-16/T-17/T-20) — sans ça
        # heading_rad valait 0.0 en dur partout, un vent "de face" n'aurait
        # rien voulu dire pour un segment qui ne va pas plein nord.
        "heading_rad": bearing_rad(start_lat, start_lng, end_lat, end_lng),
        # Position de référence pour interroger la météo (T-27) — le
        # départ du segment, pas un milieu recalculé : suffisamment
        # précis pour une grille météo bien plus large qu'un segment.
        "start_lat": start_lat,
        "start_lng": start_lng,
        "kom_seconds": parse_strava_duration(raw_segment["xoms"]["kom"]),
        "pr_seconds": stats.get("pr_elapsed_time"),
        "pr_date": stats.get("pr_date"),
        "effort_count": stats.get("effort_count"),
    }


def build_segments_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Lit tous les segments bruts de `raw_dir` et (re)crée la table `segments`.

    Chaque fichier est lu individuellement (pas via un `pyarrow.dataset`
    sur tout le dossier) : les champs absents pour un segment donné
    (ex. `athlete_segment_stats.pr_elapsed_time` jamais renseigné) font
    inférer un type `null` à pyarrow pour CE fichier, incompatible avec
    le type réel (`int64`) inféré pour un autre fichier où le champ est
    renseigné — `pyarrow.dataset` refuse alors de les lire ensemble.
    Extraire d'abord chaque ligne en dict Python puis reconstruire une
    seule table à la fin (schéma volontairement restreint aux colonnes
    qu'on garde) évite complètement ce problème.

    `conn.register` expose ensuite cette table pyarrow à DuckDB sous un
    nom SQL, sans passer par un fichier intermédiaire.
    """
    segment_paths = sorted(raw_dir.glob("*.parquet"))
    raw_segments = [pq.read_table(path).to_pylist()[0] for path in segment_paths]
    rows = [_segment_to_row(segment) for segment in raw_segments]

    segments_table = pa.Table.from_pylist(rows)
    conn.register("segments_table", segments_table)
    try:
        conn.execute("CREATE OR REPLACE TABLE segments AS SELECT * FROM segments_table")
    finally:
        conn.unregister("segments_table")
