"""Tests du CSV de tri des efforts en groupe (T-16).

Le temps prédit vient d'un modèle non calibré (CdA/Crr génériques) et
d'une approximation supplémentaire (segment = un seul tronçon à pente
moyenne, pas son profil réel) — une aide au tri, pas une détection.
La seule règle de fusion qui compte vraiment ici : `draft_status` n'est
jamais écrasé pour un effort déjà annoté.
"""

import csv
import math

import duckdb
import pyarrow as pa
import pytest

from segment_predictor.calibrate.draft_tagging import (
    compute_aggregate_mmp_curve,
    fit_current_cp,
    generate_draft_tagging_csv,
    load_existing_annotations,
    predict_segment_time_s,
)
from segment_predictor.models.power import CriticalPowerFit

_EMPTY_SCHEMAS = {
    "activities": pa.schema(
        [
            ("id", pa.int64()),
            ("type", pa.string()),
            ("device_watts", pa.bool_()),
            ("name", pa.string()),
        ]
    ),
    "streams": pa.schema(
        [
            ("activity_id", pa.int64()),
            ("sample_index", pa.int64()),
            ("t_s", pa.int64()),
            ("watts", pa.float64()),
        ]
    ),
    "segments": pa.schema(
        [
            ("id", pa.int64()),
            ("name", pa.string()),
            ("distance_m", pa.float64()),
            ("average_grade", pa.float64()),
        ]
    ),
    "segment_efforts": pa.schema(
        [
            ("id", pa.int64()),
            ("segment_id", pa.int64()),
            ("activity_id", pa.int64()),
            ("start_date", pa.string()),
            ("elapsed_time_s", pa.int64()),
        ]
    ),
}


def _make_db(
    activities: list[dict], streams: list[dict], segments: list[dict], efforts: list[dict]
):
    conn = duckdb.connect(":memory:")
    for name, rows in (
        ("activities", activities),
        ("streams", streams),
        ("segments", segments),
        ("segment_efforts", efforts),
    ):
        table = (
            pa.Table.from_pylist(rows)
            if rows
            else pa.Table.from_pylist([], schema=_EMPTY_SCHEMAS[name])
        )
        conn.register("_rows", table)
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM _rows")
        conn.unregister("_rows")
    return conn


def _flat_stream_rows(activity_id: int, watts: float, n_seconds: int) -> list[dict]:
    return [
        {"activity_id": activity_id, "sample_index": i, "t_s": i, "watts": watts}
        for i in range(n_seconds)
    ]


# ---- compute_aggregate_mmp_curve -------------------------------------------------------


def test_compute_aggregate_mmp_curve_takes_the_max_across_activities() -> None:
    activities = [
        {"id": 1, "type": "Ride", "device_watts": True},
        {"id": 2, "type": "Ride", "device_watts": True},
    ]
    streams = _flat_stream_rows(1, 250.0, 300) + _flat_stream_rows(2, 300.0, 300)
    conn = _make_db(activities, streams, [], [])

    curve = compute_aggregate_mmp_curve(conn, [60, 300])

    assert curve[60] == pytest.approx(300.0)  # activité 2 gagne
    assert curve[300] == pytest.approx(300.0)


def test_compute_aggregate_mmp_curve_ignores_non_cycling_activities() -> None:
    activities = [
        {"id": 1, "type": "Ride", "device_watts": True},
        {"id": 2, "type": "Hike", "device_watts": True},  # hors périmètre
    ]
    streams = _flat_stream_rows(1, 200.0, 300) + _flat_stream_rows(2, 500.0, 300)
    conn = _make_db(activities, streams, [], [])

    curve = compute_aggregate_mmp_curve(conn, [60])

    assert curve[60] == pytest.approx(200.0)  # pas 500 (la randonnée est ignorée)


# ---- fit_current_cp ---------------------------------------------------------------------


def test_fit_current_cp_recovers_constant_power_exactly() -> None:
    """Puissance constante = cas dégénéré du modèle CP : CP = cette puissance,
    W' = 0 exactement (travail = CP·t, une droite parfaite). R² est indéfini
    (NaN) ici, pas 1.0 : toutes les valeurs de MMP sont identiques, donc la
    variance totale (ss_tot) est nulle — comportement voulu de
    fit_critical_power (T-09), pas une régression."""
    activities = [{"id": 1, "type": "Ride", "device_watts": True}]
    streams = _flat_stream_rows(1, 280.0, 1300)
    conn = _make_db(activities, streams, [], [])

    fit = fit_current_cp(conn, durations_s=(180, 300, 600, 900, 1200))

    assert fit.cp_watts == pytest.approx(280.0, abs=1e-6)
    assert fit.w_prime_joules == pytest.approx(0.0, abs=1e-3)
    assert math.isnan(fit.r_squared)


# ---- predict_segment_time_s --------------------------------------------------------------


def test_predict_segment_time_s_matches_direct_simulation() -> None:
    from segment_predictor.models.segment import SegmentChunk, simulate_segment_time

    cp_fit = CriticalPowerFit(
        cp_watts=250.0,
        w_prime_joules=15_000.0,
        r_squared=0.9,
        n_points=5,
        duration_range_s=(180, 1200),
    )
    distance_m, average_grade = 2000.0, 0.03

    predicted_time_s = predict_segment_time_s(distance_m, average_grade, cp_fit, 91.0, 0.32, 0.005)

    expected = simulate_segment_time(
        [SegmentChunk(0.0, distance_m, average_grade, 0.0)],
        cp_fit.cp_watts,
        cp_fit.w_prime_joules,
        91.0,
        0.32,
        0.005,
    )
    assert predicted_time_s == pytest.approx(expected)


# ---- load_existing_annotations -----------------------------------------------------------


def test_load_existing_annotations_returns_empty_dict_when_no_file(tmp_path) -> None:
    assert load_existing_annotations(tmp_path / "missing.csv") == {}


def test_load_existing_annotations_reads_draft_status_by_effort_id(tmp_path) -> None:
    csv_path = tmp_path / "draft_status.csv"
    csv_path.write_text("effort_id,draft_status\n1,solo\n2,drafted\n")

    assert load_existing_annotations(csv_path) == {1: "solo", 2: "drafted"}


# ---- generate_draft_tagging_csv (intégration) ---------------------------------------------


def _sample_segment_and_effort():
    segments = [{"id": 100, "name": "Cote Test", "distance_m": 2000.0, "average_grade": 0.03}]
    efforts = [
        {
            "id": 1,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-01T10:00:00",
            "elapsed_time_s": 300,
        }
    ]
    activities = [{"id": 1, "type": "Ride", "device_watts": True, "name": "Sortie"}]
    return activities, segments, efforts


def test_generate_draft_tagging_csv_creates_sorted_csv_with_unknown_status(tmp_path) -> None:
    activities, segments, efforts = _sample_segment_and_effort()
    conn = _make_db(activities, [], segments, efforts)
    csv_path = tmp_path / "draft_status.csv"
    cp_fit = CriticalPowerFit(250.0, 15_000.0, 0.9, 5, (180, 1200))

    generate_draft_tagging_csv(conn, csv_path, mass_kg=91.0, cp_fit=cp_fit)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["effort_id"] == "1"
    assert rows[0]["segment_name"] == "Cote Test"
    assert rows[0]["actual_time_s"] == "300"
    assert rows[0]["draft_status"] == "unknown"
    assert float(rows[0]["gap_s"]) == pytest.approx(
        float(rows[0]["predicted_time_s"]) - float(rows[0]["actual_time_s"])
    )


def test_generate_draft_tagging_csv_sorts_by_gap_descending(tmp_path) -> None:
    activities = [{"id": 1, "type": "Ride", "device_watts": True, "name": "Sortie"}]
    segments = [{"id": 100, "name": "Cote", "distance_m": 2000.0, "average_grade": 0.03}]
    efforts = [
        # même segment, même puissance soutenable -> même temps prédit ;
        # gap = prédit - réel, donc un temps réel plus PETIT donne un gap
        # plus GRAND (effort plus rapide que prévu = le plus "suspect")
        {
            "id": 1,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-01T10:00:00",
            "elapsed_time_s": 400,
        },  # lent -> gap le plus petit
        {
            "id": 2,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-02T10:00:00",
            "elapsed_time_s": 200,
        },  # rapide -> gap le plus grand, doit arriver en premier
    ]
    conn = _make_db(activities, [], segments, efforts)
    csv_path = tmp_path / "draft_status.csv"
    cp_fit = CriticalPowerFit(250.0, 15_000.0, 0.9, 5, (180, 1200))

    generate_draft_tagging_csv(conn, csv_path, mass_kg=91.0, cp_fit=cp_fit)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    gaps = [float(r["gap_s"]) for r in rows]
    assert gaps == sorted(gaps, reverse=True)
    assert rows[0]["effort_id"] == "2"  # actual=200 (le plus rapide) -> écart le plus grand


def test_generate_draft_tagging_csv_preserves_manual_annotations_across_runs(tmp_path) -> None:
    activities, segments, efforts = _sample_segment_and_effort()
    conn = _make_db(activities, [], segments, efforts)
    csv_path = tmp_path / "draft_status.csv"

    generate_draft_tagging_csv(
        conn, csv_path, mass_kg=91.0, cp_fit=CriticalPowerFit(250.0, 15_000.0, 0.9, 5, (180, 1200))
    )

    # annotation manuelle, comme si l'utilisateur avait édité le CSV
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["draft_status"] = "drafted"
    predicted_time_before = rows[0]["predicted_time_s"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # relance avec un modèle DIFFÉRENT (CP change) : le temps prédit doit
    # changer, mais pas l'annotation manuelle
    generate_draft_tagging_csv(
        conn, csv_path, mass_kg=91.0, cp_fit=CriticalPowerFit(300.0, 15_000.0, 0.9, 5, (180, 1200))
    )

    with csv_path.open(newline="") as f:
        rows_after = list(csv.DictReader(f))

    assert rows_after[0]["draft_status"] == "drafted"  # préservé
    assert rows_after[0]["predicted_time_s"] != predicted_time_before  # recalculé


def test_generate_draft_tagging_csv_appends_new_efforts_as_unknown(tmp_path) -> None:
    activities, segments, efforts = _sample_segment_and_effort()
    conn = _make_db(activities, [], segments, efforts)
    csv_path = tmp_path / "draft_status.csv"
    cp_fit = CriticalPowerFit(250.0, 15_000.0, 0.9, 5, (180, 1200))

    generate_draft_tagging_csv(conn, csv_path, mass_kg=91.0, cp_fit=cp_fit)
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["draft_status"] = "solo"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # un nouvel effort apparaît dans la base (nouvelle activité)
    efforts_with_new = efforts + [
        {
            "id": 2,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-02-01T10:00:00",
            "elapsed_time_s": 310,
        }
    ]
    conn2 = _make_db(activities, [], segments, efforts_with_new)

    generate_draft_tagging_csv(conn2, csv_path, mass_kg=91.0, cp_fit=cp_fit)

    with csv_path.open(newline="") as f:
        rows_after = {r["effort_id"]: r for r in csv.DictReader(f)}

    assert rows_after["1"]["draft_status"] == "solo"  # préservé
    assert rows_after["2"]["draft_status"] == "unknown"  # nouveau


def test_generate_draft_tagging_csv_only_includes_tracked_segments(tmp_path) -> None:
    """La jointure vers main.segments filtre aux segments suivis — un effort
    sur un segment non tracké n'apparaît pas."""
    activities = [{"id": 1, "type": "Ride", "device_watts": True, "name": "Sortie"}]
    segments = [{"id": 100, "name": "Suivi", "distance_m": 2000.0, "average_grade": 0.03}]
    efforts = [
        {
            "id": 1,
            "segment_id": 100,
            "activity_id": 1,
            "start_date": "2024-01-01T10:00:00",
            "elapsed_time_s": 300,
        },
        {
            "id": 2,
            "segment_id": 999,  # pas dans `segments`
            "activity_id": 1,
            "start_date": "2024-01-01T10:00:00",
            "elapsed_time_s": 300,
        },
    ]
    conn = _make_db(activities, [], segments, efforts)
    csv_path = tmp_path / "draft_status.csv"

    generate_draft_tagging_csv(
        conn, csv_path, mass_kg=91.0, cp_fit=CriticalPowerFit(250.0, 15_000.0, 0.9, 5, (180, 1200))
    )

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["effort_id"] for r in rows] == ["1"]


def test_generate_draft_tagging_csv_handles_prediction_failure_gracefully(tmp_path) -> None:
    """Si simulate_segment_time échoue pour un effort (CP/W' incohérents avec
    la pente), on le signale (colonnes vides) plutôt que de planter tout le
    CSV ou d'inventer un temps."""
    activities, segments, efforts = _sample_segment_and_effort()
    conn = _make_db(activities, [], segments, efforts)
    csv_path = tmp_path / "draft_status.csv"
    # CP très négatif : aucune vitesse ne peut satisfaire une puissance
    # positive avec cette pente -> cyclist_speed_from_power ne trouvera pas
    # de racine dans ses bornes par défaut.
    broken_cp_fit = CriticalPowerFit(-5000.0, 1.0, 0.9, 5, (180, 1200))

    generate_draft_tagging_csv(conn, csv_path, mass_kg=91.0, cp_fit=broken_cp_fit)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["predicted_time_s"] == ""
    assert rows[0]["gap_s"] == ""
    assert rows[0]["draft_status"] == "unknown"  # le reste de la ligne reste exploitable
