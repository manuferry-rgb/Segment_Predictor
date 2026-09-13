"""Tests de la construction de la table DuckDB `segments` (T-06, T-44c).

Faits PHYSIQUES partagés uniquement depuis T-44c (distance, tracé, cap,
KOM) — le PR par athlète a déménagé vers `user_segment_stats`, testée
séparément dans test_user_segments.py.

C'est ici que le format "mm:ss"/"h:mm:ss" du KOM Strava est parsé —
la couche ingest ne stocke que le JSON brut, aucun parsing là-bas.
"""

import math

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segment_predictor.storage.segments import build_segments_table, parse_strava_duration

# ---- parse_strava_duration -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_seconds"),
    [
        ("4:12", 252),  # mm:ss
        ("0:09", 9),  # mm:ss, minute à un chiffre
        ("39:56", 2396),  # mm:ss, proche de l'heure
        ("1:02:35", 3755),  # h:mm:ss
        ("2:00:00", 7200),  # h:mm:ss, heures rondes
        ("53s", 53),  # segment très court (< 1 min) : pas de ":", trouvé sur 7/75 favoris réels
        ("0s", 0),
    ],
)
def test_parse_strava_duration_valid_formats(raw: str, expected_seconds: int) -> None:
    assert parse_strava_duration(raw) == expected_seconds


def test_parse_strava_duration_rejects_invalid_format() -> None:
    with pytest.raises(ValueError, match="Format de durée"):
        parse_strava_duration("not-a-duration")


# ---- build_segments_table -------------------------------------------------------------


def _write_raw_segment(raw_dir, segment: dict, filename: str, user_dir: str = "1") -> None:
    """Écrit sous `raw_dir/<user_dir>/` (T-44d) : `build_segments_table`
    lit maintenant TOUS les sous-dossiers utilisateur du dossier PARENT
    `raw_dir` (voir _read_all_shared_raw_segments) — `user_dir` par
    défaut ne compte pas pour ces tests (peu importe qui a fetché quoi,
    seuls les faits physiques du segment sont vérifiés ici)."""
    target_dir = raw_dir / user_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([segment]), target_dir / filename)


def test_build_segments_table_parses_kom(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 229781,
            "name": "Alpe d'Huez",
            "distance": 13800.0,
            "average_grade": 8.1,
            "total_elevation_gain": 1120.0,
            "start_latlng": [45.09, 6.07],
            "end_latlng": [45.09, 6.10],
            "map": {"polyline": "fake_polyline_1"},
            "xoms": {"kom": "39:56"},
        },
        "229781.parquet",
    )
    _write_raw_segment(
        raw_dir,
        {
            "id": 654321,
            "name": "Col de la Colombiere",
            "distance": 7500.0,
            "average_grade": 6.7,
            "total_elevation_gain": 640.0,
            "start_latlng": [45.97, 6.47],
            "end_latlng": [45.99, 6.50],
            "map": {"polyline": "fake_polyline_2"},
            "xoms": {"kom": "1:02:35"},
        },
        "654321.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    rows = conn.execute("SELECT id, name, kom_seconds FROM segments ORDER BY id").fetchall()

    assert rows == [
        (229781, "Alpe d'Huez", 2396),
        (654321, "Col de la Colombiere", 3755),
    ]


def test_build_segments_table_reads_total_elevation_gain(tmp_path) -> None:
    """D+ (T-31) : renvoyé tel quel par Strava (mètres), pas de conversion
    d'unité contrairement à average_grade. Pas de D- ici : Strava ne le
    fournit pas au niveau segment (vérifié sur le payload réel), donc rien
    à stocker — inventer une valeur violerait la règle "pas de données
    inventées" du projet."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment",
            "distance": 1000.0,
            "average_grade": 5.0,
            "total_elevation_gain": 223.6,
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "map": {"polyline": "fake_polyline"},
            "xoms": {"kom": "1:00"},
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    row = conn.execute("SELECT total_elevation_gain_m FROM segments").fetchone()
    assert row == pytest.approx((223.6,))


def test_build_segments_table_reads_polyline(tmp_path) -> None:
    """Polyline encodé du tracé (T-32, préparatoire) : stocké TEL QUEL,
    pas décodé ici — `storage` extrait des champs structurels, il ne fait
    pas de géométrie. Le décodage (cap réel par tronçon, pour un vent
    correctement projeté sur tout le segment plutôt qu'un seul cap
    moyen) est un traitement séparé, pas encore branché."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment",
            "distance": 1000.0,
            "average_grade": 5.0,
            "total_elevation_gain": 50.0,
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "map": {"polyline": "okaaH{usl@|ChChB"},
            "xoms": {"kom": "1:00"},
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    row = conn.execute("SELECT polyline FROM segments").fetchone()
    assert row == ("okaaH{usl@|ChChB",)


def test_build_segments_table_converts_average_grade_from_percent_to_fraction(tmp_path) -> None:
    """Régression T-16 : Strava renvoie average_grade en pourcent (0.2 = 0.2%),
    mais models/physics.py (grade de cyclist_power_required, depuis T-10)
    attend une fraction rise/run (0.002). Non converti, un 0.2% était traité
    comme 20% par la physique — repéré en branchant enfin cette colonne sur
    un temps prédit réel (T-16), jamais avant."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment",
            "distance": 1000.0,
            "average_grade": 0.2,  # 0.2%, comme renvoyé par Strava
            "total_elevation_gain": 2.0,
            "map": {"polyline": "fake_polyline"},
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "xoms": {"kom": "1:00"},
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    average_grade = conn.execute("SELECT average_grade FROM segments").fetchone()[0]
    assert average_grade == pytest.approx(0.002)


def test_build_segments_table_replaces_existing_table(tmp_path) -> None:
    """Relancer le build ne doit pas empiler les anciennes lignes — table
    PARTAGÉE (T-44c), un CREATE OR REPLACE reste correct ici."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment",
            "distance": 1000.0,
            "average_grade": 5.0,
            "total_elevation_gain": 50.0,
            "map": {"polyline": "fake_polyline"},
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "xoms": {"kom": "1:00"},
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)
    build_segments_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM segments").fetchone()[0]
    assert count == 1


def test_build_segments_table_computes_heading_from_start_and_end_latlng(tmp_path) -> None:
    """Cap global du segment (T-27) : approximation à l'échelle du segment
    entier, faute d'un profil tronçon par tronçon stocké au niveau segment
    (T-07b) — nécessaire pour que le vent (T-15/T-27) ait un sens sur un
    vrai segment, pas juste un heading=0.0 arbitraire."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Plein est",
            "distance": 1000.0,
            "average_grade": 5.0,
            "total_elevation_gain": 50.0,
            "map": {"polyline": "fake_polyline"},
            "start_latlng": [45.0, 6.0],
            "end_latlng": [45.0, 6.1],  # même latitude, longitude croissante : plein est
            "xoms": {"kom": "1:00"},
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    heading_rad = conn.execute("SELECT heading_rad FROM segments").fetchone()[0]
    assert heading_rad == pytest.approx(math.pi / 2, abs=1e-3)


def test_build_segments_table_reads_start_lat_lng(tmp_path) -> None:
    """Position de référence pour interroger la météo (T-27) — le point de
    départ du segment, pas un milieu recalculé : assez précis pour une
    grille météo bien plus large qu'un segment."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment",
            "distance": 1000.0,
            "average_grade": 5.0,
            "total_elevation_gain": 50.0,
            "map": {"polyline": "fake_polyline"},
            "start_latlng": [45.123, 6.456],
            "end_latlng": [45.13, 6.46],
            "xoms": {"kom": "1:00"},
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    row = conn.execute("SELECT start_lat, start_lng FROM segments").fetchone()
    assert row == pytest.approx((45.123, 6.456))


def test_build_segments_table_raises_when_latlng_missing(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment sans coordonnées",
            "distance": 1000.0,
            "average_grade": 5.0,
            "xoms": {"kom": "1:00"},
            # pas de start_latlng / end_latlng
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    with pytest.raises(KeyError):
        build_segments_table(conn, raw_dir)


def test_build_segments_table_combines_and_dedupes_across_user_subdirectories(tmp_path) -> None:
    """T-44d : `raw_dir` est le dossier PARENT, un segment favori de
    PLUSIEURS utilisateurs (deux sous-dossiers) ne doit compter qu'une
    fois, et un segment propre à un seul utilisateur doit apparaître."""
    raw_dir = tmp_path / "segments"
    shared = {
        "id": 1,
        "name": "Segment partagé",
        "distance": 1000.0,
        "average_grade": 5.0,
        "total_elevation_gain": 50.0,
        "map": {"polyline": "fake_polyline"},
        "start_latlng": [47.5, 7.4],
        "end_latlng": [47.51, 7.41],
        "xoms": {"kom": "1:00"},
    }
    only_b = {**shared, "id": 2, "name": "Segment de B uniquement"}
    _write_raw_segment(raw_dir, shared, "1.parquet", user_dir="user_a")
    _write_raw_segment(raw_dir, shared, "1.parquet", user_dir="user_b")
    _write_raw_segment(raw_dir, only_b, "2.parquet", user_dir="user_b")

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    rows = conn.execute("SELECT id, name FROM segments ORDER BY id").fetchall()
    assert rows == [(1, "Segment partagé"), (2, "Segment de B uniquement")]
