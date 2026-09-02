"""Tests de la récupération des activités détaillées Strava (T-07b).

Aucun appel réseau réel (httpx.MockTransport), aucune attente réelle
(un `sleep` factice est injecté partout). Même pattern reprenable et
gestion de quota que T-05 (strava_streams.py) — mêmes tests transposés,
pas ré-expliqués en détail ici.
"""

import httpx

from segment_predictor.ingest.strava_activity_details import (
    fetch_and_store_activity_details,
    get_activity_detail,
    save_activity_detail,
)


def NO_SLEEP(seconds: float) -> None:  # noqa: N802
    raise AssertionError(f"no sleep expected, got {seconds}s")


def _fake_detail(activity_id: int) -> dict:
    return {
        "id": activity_id,
        "name": f"Activité {activity_id}",
        "segment_efforts": [
            {
                "id": 1000 + activity_id,
                "segment": {"id": 7722237},
                "elapsed_time": 467,
                "moving_time": 467,
                "start_date": "2023-06-07T09:00:00Z",
                "distance": 4809.1,
                "average_watts": 280.0,
                "device_watts": True,
                "average_heartrate": 165.0,
                "pr_rank": 1,
            }
        ],
    }


def test_get_activity_detail_requests_the_right_path_and_returns_raw_json() -> None:
    fake = _fake_detail(123)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/activities/123"
        return httpx.Response(200, json=fake)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    detail = get_activity_detail(client, "tok", 123, sleep=NO_SLEEP)

    assert detail == fake


def test_save_activity_detail_writes_one_file_per_activity(tmp_path) -> None:
    raw_dir = tmp_path / "details"

    written_path = save_activity_detail(raw_dir, 123, _fake_detail(123))

    assert written_path == raw_dir / "123.parquet"


def test_fetch_and_store_activity_details_skips_already_downloaded(tmp_path) -> None:
    raw_dir = tmp_path / "details"
    save_activity_detail(raw_dir, 1, _fake_detail(1))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/activities/2"
        return httpx.Response(200, json=_fake_detail(2))

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_activity_details(client, "tok", [1, 2], raw_dir, sleep=NO_SLEEP)

    assert summary.fetched_ids == [2]
    assert summary.already_downloaded_ids == [1]
    assert summary.stopped_due_to_daily_quota is False


def test_fetch_and_store_activity_details_stops_cleanly_when_daily_quota_reached(tmp_path) -> None:
    raw_dir = tmp_path / "details"

    def handler(request: httpx.Request) -> httpx.Response:
        activity_id = int(request.url.path.rsplit("/", 1)[-1])
        if activity_id == 2:
            raise AssertionError(
                "activity 2 should never be requested: daily quota already reached"
            )
        return httpx.Response(
            200,
            json=_fake_detail(1),
            headers={"X-RateLimit-Limit": "200,1000", "X-RateLimit-Usage": "51,1000"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_activity_details(client, "tok", [1, 2], raw_dir, sleep=NO_SLEEP)

    assert summary.fetched_ids == [1]
    assert summary.remaining_ids == [2]
    assert summary.stopped_due_to_daily_quota is True
