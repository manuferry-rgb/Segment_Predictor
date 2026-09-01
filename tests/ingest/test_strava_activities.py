"""Tests de la récupération incrémentale des activités Strava.

Aucun appel réseau réel (httpx.MockTransport) et aucun repos réel entre
tentatives (une fonction `sleep` factice est injectée partout).
"""

from datetime import UTC, datetime

import httpx
import pyarrow.parquet as pq

from segment_predictor.ingest.strava_activities import (
    fetch_and_store_new_activities,
    list_activities,
    read_stored_activity_ids,
    read_watermark,
    save_activities,
)

NO_SLEEP = lambda seconds: (_ for _ in ()).throw(  # noqa: E731
    AssertionError(f"no sleep expected, got {seconds}s")
)


def _activity(activity_id: int, start_date: str) -> dict:
    """Une activité Strava minimale mais réaliste (mêmes noms de champs que l'API)."""
    return {
        "id": activity_id,
        "name": f"Sortie {activity_id}",
        "start_date": start_date,
        "distance": 42000.0,
    }


# ---- list_activities : pagination ------------------------------------------------


def test_list_activities_paginates_until_a_short_page() -> None:
    page_1 = [_activity(i, "2024-01-01T10:00:00Z") for i in range(200)]
    page_2 = [_activity(i, "2024-01-02T10:00:00Z") for i in range(200, 250)]
    requested_pages = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        requested_pages.append(page)
        assert request.url.params["per_page"] == "200"
        body = page_1 if page == 1 else (page_2 if page == 2 else [])
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    activities = list_activities(client, "tok", sleep=NO_SLEEP)

    assert requested_pages == [1, 2]  # s'arrête dès qu'une page < per_page
    assert len(activities) == 250


def test_list_activities_passes_after_param_for_incremental_fetch() -> None:
    captured_params = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_params.update(request.url.params)
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))

    list_activities(client, "tok", after=1_700_000_000, sleep=NO_SLEEP)

    assert captured_params["after"] == "1700000000"


def test_list_activities_pauses_before_next_page_when_short_term_quota_exhausted() -> None:
    page_1 = [_activity(i, "2024-01-01T10:00:00Z") for i in range(200)]
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["value"] += 1
        if call_count["value"] == 1:
            return httpx.Response(
                200,
                json=page_1,
                headers={"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "200,200"},
            )
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    list_activities(client, "tok", sleep=lambda s: sleep_calls.append(s))

    assert sleep_calls == [900.0]


# ---- watermark & dédoublonnage, sur un dossier Parquet local ---------------------


def test_read_watermark_is_none_when_no_data_yet(tmp_path) -> None:
    assert read_watermark(tmp_path / "does_not_exist") is None


def test_read_watermark_returns_max_start_date_as_epoch(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    save_activities(
        raw_dir, [_activity(1, "2024-01-01T10:00:00Z"), _activity(2, "2024-03-01T08:00:00Z")]
    )

    watermark = read_watermark(raw_dir)

    assert watermark == int(datetime(2024, 3, 1, 8, 0, 0, tzinfo=UTC).timestamp())


def test_read_stored_activity_ids(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    save_activities(
        raw_dir, [_activity(1, "2024-01-01T10:00:00Z"), _activity(2, "2024-03-01T08:00:00Z")]
    )

    assert read_stored_activity_ids(raw_dir) == {1, 2}


def test_save_activities_writes_nothing_for_empty_list(tmp_path) -> None:
    raw_dir = tmp_path / "raw"

    written_path = save_activities(raw_dir, [])

    assert written_path is None
    assert not raw_dir.exists()


def test_save_activities_writes_raw_fields_unchanged(tmp_path) -> None:
    raw_dir = tmp_path / "raw"

    written_path = save_activities(raw_dir, [_activity(1, "2024-01-01T10:00:00Z")])

    table = pq.read_table(written_path)
    row = table.to_pylist()[0]
    assert row["id"] == 1
    assert row["distance"] == 42000.0
    assert row["start_date"] == "2024-01-01T10:00:00Z"


# ---- orchestration bout-en-bout : relançable sans doublons -----------------------


def test_fetch_and_store_is_rerunnable_without_duplicates(tmp_path) -> None:
    raw_dir = tmp_path / "raw"

    # 1er lancement : 2 activités.
    def handler_first_run(request: httpx.Request) -> httpx.Response:
        assert "after" not in request.url.params  # pas de watermark au premier lancement
        return httpx.Response(
            200, json=[_activity(1, "2024-01-01T10:00:00Z"), _activity(2, "2024-01-02T10:00:00Z")]
        )

    client = httpx.Client(transport=httpx.MockTransport(handler_first_run))
    written_path = fetch_and_store_new_activities(client, "tok", raw_dir, sleep=NO_SLEEP)

    assert written_path is not None
    assert read_stored_activity_ids(raw_dir) == {1, 2}

    # 2e lancement : Strava renvoie les 2 mêmes activités (limite `after` inclusive
    # côté serveur) + 1 nouvelle -> seule la nouvelle doit être écrite.
    def handler_second_run(request: httpx.Request) -> httpx.Response:
        assert request.url.params["after"] == str(read_watermark(raw_dir))
        return httpx.Response(
            200,
            json=[
                _activity(1, "2024-01-01T10:00:00Z"),
                _activity(2, "2024-01-02T10:00:00Z"),
                _activity(3, "2024-01-03T10:00:00Z"),
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler_second_run))
    written_path_2 = fetch_and_store_new_activities(client, "tok", raw_dir, sleep=NO_SLEEP)

    assert written_path_2 is not None
    assert read_stored_activity_ids(raw_dir) == {1, 2, 3}
    table = pq.read_table(written_path_2)
    assert table.to_pylist() == [_activity(3, "2024-01-03T10:00:00Z")]


def test_fetch_and_store_returns_none_when_nothing_new(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    save_activities(raw_dir, [_activity(1, "2024-01-01T10:00:00Z")])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_activity(1, "2024-01-01T10:00:00Z")])

    client = httpx.Client(transport=httpx.MockTransport(handler))

    written_path = fetch_and_store_new_activities(client, "tok", raw_dir, sleep=NO_SLEEP)

    assert written_path is None
