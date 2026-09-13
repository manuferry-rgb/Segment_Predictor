"""Construction des tables DuckDB dérivées des JSON bruts Strava de segments.

`segments` (T-06) reste PARTAGÉE entre tous les utilisateurs (T-44c) :
distance, tracé, cap, KOM — des faits physiques identiques pour
n'importe qui regarde ce segment. `user_segment_stats` et
`user_starred_segments`, elles, sont PAR ATHLÈTE : le PR
(`athlete_segment_stats`) reflète le token qui a fait la requête
`GET /segments/{id}`, pas une propriété du segment lui-même — avant
T-44c, il vivait à tort dans `segments`, comme s'il n'y avait qu'un
PR possible par segment.

C'est aussi ici, pas dans ingest, que le format "mm:ss" / "h:mm:ss" du
KOM (`xoms.kom`) est parsé en secondes.
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


def _read_raw_segments(raw_dir: Path) -> list[dict]:
    """Chaque fichier est lu individuellement (pas via un `pyarrow.dataset`
    sur tout le dossier) : les champs absents pour un segment donné
    (ex. `athlete_segment_stats.pr_elapsed_time` jamais renseigné) font
    inférer un type `null` à pyarrow pour CE fichier, incompatible avec
    le type réel (`int64`) inféré pour un autre fichier où le champ est
    renseigné — `pyarrow.dataset` refuse alors de les lire ensemble.
    Extraire d'abord chaque ligne en dict Python puis reconstruire une
    seule table à la fin évite complètement ce problème. Partagée par
    les trois builders de ce module : ils lisent tous la même source.
    """
    return [pq.read_table(path).to_pylist()[0] for path in sorted(raw_dir.glob("*.parquet"))]


def _read_all_shared_raw_segments(segments_raw_dir: Path) -> list[dict]:
    """`build_segments_table`, contrairement aux deux builders par
    utilisateur (T-44c), doit voir TOUS les segments jamais téléchargés
    par N'IMPORTE QUEL utilisateur — `segments_raw_dir` est ici le
    dossier PARENT (`data/raw/strava_segments/`, T-44d), un niveau
    au-dessus des sous-dossiers `<user_id>/` que lit `_read_raw_segments`.

    Dédoublonné par id de segment : le même segment partagé peut
    apparaître dans plusieurs sous-dossiers si plusieurs utilisateurs
    l'ont en favori — son contenu physique (distance, tracé, KOM) étant
    identique partout, prendre n'importe laquelle des copies suffit.

    `segments_raw_dir` peut ne pas encore exister (aucun utilisateur n'a
    jamais rien fetché) : liste vide plutôt qu'une erreur, même logique
    que `build_wellness_table` pour une source pas encore alimentée.
    """
    if not segments_raw_dir.exists():
        return []

    by_id: dict[int, dict] = {}
    for user_dir in sorted(p for p in segments_raw_dir.iterdir() if p.is_dir()):
        for segment in _read_raw_segments(user_dir):
            by_id[segment["id"]] = segment
    return list(by_id.values())


def _segment_to_row(raw_segment: dict) -> dict:
    """Un JSON brut `GET /segments/{id}` -> une ligne de `segments` — les
    faits PHYSIQUES seulement (T-44c) ; le PR de l'athlète qui a fait la
    requête vit dans `user_segment_stats`, pas ici (voir docstring du
    module).
    """
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
        # Record ABSOLU (tous athlètes confondus) : un vrai fait partagé,
        # contrairement au PR ci-dessous (par athlète, T-44c).
        "kom_seconds": parse_strava_duration(raw_segment["xoms"]["kom"]),
    }


def build_segments_table(conn: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    """Lit tous les segments bruts (tous utilisateurs confondus, T-44d) et
    (re)crée la table PARTAGÉE `segments`.

    `raw_dir` est le dossier PARENT `data/raw/strava_segments/`, PAS un
    sous-dossier `<user_id>/` (voir _read_all_shared_raw_segments) —
    contrairement à `build_user_segment_stats_table`/
    `build_user_starred_segments_table` ci-dessous, qui elles reçoivent
    le sous-dossier d'UN SEUL utilisateur.

    CREATE OR REPLACE reste correct ici (contrairement aux tables
    personnelles de T-44b) : cette table ne contient que des faits
    physiques identiques pour tout le monde, il n'y a pas de lignes
    "d'un autre utilisateur" à préserver en la reconstruisant.
    """
    rows = [_segment_to_row(segment) for segment in _read_all_shared_raw_segments(raw_dir)]

    segments_table = pa.Table.from_pylist(rows)
    conn.register("segments_table", segments_table)
    try:
        conn.execute("CREATE OR REPLACE TABLE segments AS SELECT * FROM segments_table")
    finally:
        conn.unregister("segments_table")


def _segment_to_user_stats_row(raw_segment: dict, user_id: int) -> dict:
    """`athlete_segment_stats` doit exister (sinon la réponse est vraiment
    anormale), mais ses champs `pr_*` peuvent être `None` : Strava renvoie
    ça pour un segment jamais roulé (`effort_count: 0`), ce n'est pas une
    erreur à masquer, juste un PR qui n'existe pas encore.
    """
    stats = raw_segment["athlete_segment_stats"]
    return {
        "user_id": user_id,
        "segment_id": raw_segment["id"],
        "pr_seconds": stats.get("pr_elapsed_time"),
        "pr_date": stats.get("pr_date"),
        "effort_count": stats.get("effort_count"),
    }


def build_user_segment_stats_table(
    conn: duckdb.DuckDBPyConnection, raw_dir: Path, user_id: int
) -> None:
    """(Re)rattache à `user_id` le PR/nombre de passages de chaque segment
    de `raw_dir` — DELETE puis INSERT (T-44b/T-44c), pas CREATE OR REPLACE :
    même raisonnement que les tables personnelles de T-44b, un autre
    utilisateur peut déjà avoir ses propres stats sur les MÊMES segments
    partagés.

    `raw_dir` : le sous-dossier `<user_id>/` de CET utilisateur (T-44d),
    pas le dossier parent que lit `build_segments_table`.
    """
    rows = [_segment_to_user_stats_row(segment, user_id) for segment in _read_raw_segments(raw_dir)]

    stats_table = pa.Table.from_pylist(rows)
    conn.register("user_segment_stats_table", stats_table)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_segment_stats AS "
            "SELECT * FROM user_segment_stats_table WHERE FALSE"
        )
        conn.execute("DELETE FROM user_segment_stats WHERE user_id = ?", [user_id])
        conn.execute("INSERT INTO user_segment_stats SELECT * FROM user_segment_stats_table")
    finally:
        conn.unregister("user_segment_stats_table")


def build_user_starred_segments_table(
    conn: duckdb.DuckDBPyConnection, raw_dir: Path, user_id: int
) -> None:
    """(Re)construit la liste des segments "connus" de `user_id` — dans
    l'usage normal (`fetch_segments.py` sans argument, T-06), ce sont
    exactement ses favoris Strava (`GET /segments/starred`). Si
    `fetch_segments.py` a été appelé avec des ids explicites (usage de
    test/ciblage documenté dans sa propre docstring), un segment non
    favori peut aussi s'y trouver — limite ASSUMÉE, pas cachée : `raw_dir`
    ne porte aucun signal permettant de distinguer les deux cas.

    DELETE puis INSERT par `user_id`, même raisonnement que
    `build_user_segment_stats_table` ci-dessus.
    """
    rows = [
        {"user_id": user_id, "segment_id": segment["id"]}
        for segment in _read_raw_segments(raw_dir)
    ]

    starred_table = pa.Table.from_pylist(
        rows, schema=pa.schema([("user_id", pa.int64()), ("segment_id", pa.int64())])
    )
    conn.register("user_starred_segments_table", starred_table)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_starred_segments AS "
            "SELECT * FROM user_starred_segments_table WHERE FALSE"
        )
        conn.execute("DELETE FROM user_starred_segments WHERE user_id = ?", [user_id])
        conn.execute(
            "INSERT INTO user_starred_segments SELECT * FROM user_starred_segments_table"
        )
    finally:
        conn.unregister("user_starred_segments_table")
