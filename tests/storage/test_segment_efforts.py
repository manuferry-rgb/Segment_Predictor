"""Tests de la construction de la table DuckDB `segment_efforts` (T-07b).

Extrait les efforts embarqués dans les activités détaillées — tous les
segments publics croisés, pas seulement ceux suivis dans `main.segments`
(la jointure vers les segments suivis, c'est le travail de T-16).
"""

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segment_predictor.storage.segment_efforts import build_segment_efforts_table


def _write_activity_detail(
    raw_dir, activity_id: int, detail: dict, filename: str | None = None
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([detail]), raw_dir / (filename or f"{activity_id}.parquet"))


def _real_shaped_effort(effort_id: int, segment_id: int, activity_id: int, **overrides) -> dict:
    """Forme vérifiée sur une vraie réponse Strava (GET /activities/{id})."""
    base = {
        "id": effort_id,
        "segment": {"id": segment_id, "name": "Sierentz - Kembs"},
        "activity": {"id": activity_id},
        "elapsed_time": 506,
        "moving_time": 506,
        "start_date": "2024-07-16T05:34:37Z",
        "distance": 4649.0,
        "average_watts": 187.1,
        "device_watts": True,
        "average_heartrate": 75.9,
        "pr_rank": 1,
        "kom_rank": None,
    }
    base.update(overrides)
    return base


def test_build_segment_efforts_table_extracts_efforts(tmp_path) -> None:
    raw_dir = tmp_path / "details"
    _write_activity_detail(
        raw_dir,
        11900338473,
        {"id": 11900338473, "segment_efforts": [_real_shaped_effort(1, 7722237, 11900338473)]},
    )

    conn = duckdb.connect(":memory:")
    build_segment_efforts_table(conn, raw_dir)

    row = conn.execute(
        "SELECT id, segment_id, activity_id, elapsed_time_s, moving_time_s, distance_m, "
        "average_watts, device_watts, average_heartrate, pr_rank, kom_rank "
        "FROM segment_efforts"
    ).fetchone()

    assert row == (1, 7722237, 11900338473, 506, 506, 4649.0, 187.1, True, 75.9, 1, None)


def test_build_segment_efforts_table_combines_multiple_activities(tmp_path) -> None:
    raw_dir = tmp_path / "details"
    _write_activity_detail(
        raw_dir, 1, {"id": 1, "segment_efforts": [_real_shaped_effort(10, 7722237, 1)]}
    )
    _write_activity_detail(
        raw_dir,
        2,
        {
            "id": 2,
            "segment_efforts": [
                _real_shaped_effort(20, 7722237, 2),
                _real_shaped_effort(21, 18955350, 2),
            ],
        },
    )

    conn = duckdb.connect(":memory:")
    build_segment_efforts_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM segment_efforts").fetchone()[0]
    assert count == 3


def test_build_segment_efforts_table_ignores_activities_without_efforts(tmp_path) -> None:
    raw_dir = tmp_path / "details"
    _write_activity_detail(raw_dir, 1, {"id": 1, "segment_efforts": []})
    _write_activity_detail(raw_dir, 2, {"id": 2})  # clé absente : jamais matché aucun segment

    conn = duckdb.connect(":memory:")
    build_segment_efforts_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM segment_efforts").fetchone()[0]
    assert count == 0


def test_build_segment_efforts_table_allows_null_optional_metrics(tmp_path) -> None:
    """Un effort sans capteur de puissance (vélo sans compteur ce jour-là) : NULL légitime."""
    raw_dir = tmp_path / "details"
    _write_activity_detail(
        raw_dir,
        1,
        {
            "id": 1,
            "segment_efforts": [
                _real_shaped_effort(
                    1,
                    7722237,
                    1,
                    average_watts=None,
                    device_watts=False,
                    average_heartrate=None,
                    pr_rank=None,
                )
            ],
        },
    )

    conn = duckdb.connect(":memory:")
    build_segment_efforts_table(conn, raw_dir)

    row = conn.execute(
        "SELECT average_watts, average_heartrate, pr_rank FROM segment_efforts"
    ).fetchone()
    assert row == (None, None, None)


def test_build_segment_efforts_table_raises_when_core_field_missing(tmp_path) -> None:
    raw_dir = tmp_path / "details"
    effort = _real_shaped_effort(1, 7722237, 1)
    del effort["elapsed_time"]
    _write_activity_detail(raw_dir, 1, {"id": 1, "segment_efforts": [effort]})

    conn = duckdb.connect(":memory:")
    with pytest.raises(KeyError):
        build_segment_efforts_table(conn, raw_dir)


def test_build_segment_efforts_table_replaces_existing_table(tmp_path) -> None:
    raw_dir = tmp_path / "details"
    _write_activity_detail(
        raw_dir, 1, {"id": 1, "segment_efforts": [_real_shaped_effort(1, 7722237, 1)]}
    )

    conn = duckdb.connect(":memory:")
    build_segment_efforts_table(conn, raw_dir)
    build_segment_efforts_table(conn, raw_dir)

    count = conn.execute("SELECT count(*) FROM segment_efforts").fetchone()[0]
    assert count == 1
