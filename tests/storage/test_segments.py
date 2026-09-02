"""Tests de la construction de la table DuckDB `segments` (T-06).

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


def _write_raw_segment(raw_dir, segment: dict, filename: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([segment]), raw_dir / filename)


def test_build_segments_table_parses_kom_and_reads_pr(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 229781,
            "name": "Alpe d'Huez",
            "distance": 13800.0,
            "average_grade": 8.1,
            "start_latlng": [45.09, 6.07],
            "end_latlng": [45.09, 6.10],
            "xoms": {"kom": "39:56"},
            "athlete_segment_stats": {"pr_elapsed_time": 3120, "pr_date": "2023-07-14T09:15:00Z"},
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
            "start_latlng": [45.97, 6.47],
            "end_latlng": [45.99, 6.50],
            "xoms": {"kom": "1:02:35"},
            "athlete_segment_stats": {"pr_elapsed_time": 4500, "pr_date": "2022-05-01T08:00:00Z"},
        },
        "654321.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    rows = conn.execute(
        "SELECT id, name, kom_seconds, pr_seconds, pr_date FROM segments ORDER BY id"
    ).fetchall()

    assert rows == [
        (229781, "Alpe d'Huez", 2396, 3120, "2023-07-14T09:15:00Z"),
        (654321, "Col de la Colombiere", 3755, 4500, "2022-05-01T08:00:00Z"),
    ]


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
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "xoms": {"kom": "1:00"},
            "athlete_segment_stats": {"pr_elapsed_time": 60, "pr_date": "2023-01-01T00:00:00Z"},
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    average_grade = conn.execute("SELECT average_grade FROM segments").fetchone()[0]
    assert average_grade == pytest.approx(0.002)


def test_build_segments_table_handles_never_ridden_segment_across_files(tmp_path) -> None:
    """Régression : un segment jamais roulé a `athlete_segment_stats.pr_*` à
    None, ce qui fait inférer un type `null` par pyarrow pour CE fichier —
    incompatible avec le type réel (int64/string) d'un autre fichier où le
    champ est renseigné. build_segments_table doit lire les deux malgré tout.
    """
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Jamais roulé",
            "distance": 2000.0,
            "average_grade": 4.0,
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "xoms": {"kom": "1:26"},
            "athlete_segment_stats": {
                "pr_elapsed_time": None,
                "pr_date": None,
                "effort_count": 0,
            },
        },
        "1.parquet",
    )
    _write_raw_segment(
        raw_dir,
        {
            "id": 2,
            "name": "Déjà roulé",
            "distance": 3000.0,
            "average_grade": 3.0,
            "start_latlng": [45.97, 6.47],
            "end_latlng": [45.99, 6.50],
            "xoms": {"kom": "2:00"},
            "athlete_segment_stats": {
                "pr_elapsed_time": 180,
                "pr_date": "2024-01-01T00:00:00Z",
                "effort_count": 5,
            },
        },
        "2.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)  # ne doit pas lever

    rows = conn.execute("SELECT id, pr_seconds, pr_date FROM segments ORDER BY id").fetchall()

    assert rows == [(1, None, None), (2, 180, "2024-01-01T00:00:00Z")]


def test_build_segments_table_reads_effort_count(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment",
            "distance": 1000.0,
            "average_grade": 5.0,
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "xoms": {"kom": "1:00"},
            "athlete_segment_stats": {
                "pr_elapsed_time": 90,
                "pr_date": "2023-01-01T00:00:00Z",
                "effort_count": 16,
            },
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    build_segments_table(conn, raw_dir)

    row = conn.execute("SELECT effort_count FROM segments").fetchone()
    assert row == (16,)


def test_build_segments_table_raises_when_pr_stats_missing(tmp_path) -> None:
    """Pas de valeur par défaut silencieuse : un segment mal formé lève, il n'est pas ignoré."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment sans stats",
            "distance": 1000.0,
            "average_grade": 5.0,
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "xoms": {"kom": "1:00"},
            # pas de athlete_segment_stats
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    with pytest.raises(KeyError):
        build_segments_table(conn, raw_dir)


def test_build_segments_table_replaces_existing_table(tmp_path) -> None:
    """Relancer le build ne doit pas empiler les anciennes lignes."""
    raw_dir = tmp_path / "segments"
    _write_raw_segment(
        raw_dir,
        {
            "id": 1,
            "name": "Segment",
            "distance": 1000.0,
            "average_grade": 5.0,
            "start_latlng": [47.5, 7.4],
            "end_latlng": [47.51, 7.41],
            "xoms": {"kom": "1:00"},
            "athlete_segment_stats": {"pr_elapsed_time": 60, "pr_date": "2023-01-01T00:00:00Z"},
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
            "start_latlng": [45.0, 6.0],
            "end_latlng": [45.0, 6.1],  # même latitude, longitude croissante : plein est
            "xoms": {"kom": "1:00"},
            "athlete_segment_stats": {"pr_elapsed_time": 60, "pr_date": "2023-01-01T00:00:00Z"},
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
            "start_latlng": [45.123, 6.456],
            "end_latlng": [45.13, 6.46],
            "xoms": {"kom": "1:00"},
            "athlete_segment_stats": {"pr_elapsed_time": 60, "pr_date": "2023-01-01T00:00:00Z"},
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
            "athlete_segment_stats": {"pr_elapsed_time": 60, "pr_date": "2023-01-01T00:00:00Z"},
            # pas de start_latlng / end_latlng
        },
        "1.parquet",
    )

    conn = duckdb.connect(":memory:")
    with pytest.raises(KeyError):
        build_segments_table(conn, raw_dir)
