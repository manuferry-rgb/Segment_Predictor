"""Tests de la récupération des segments Strava (T-06).

Aucun appel réseau réel (httpx.MockTransport), aucune attente réelle
(un `sleep` factice est injecté partout).
"""

import httpx
import pyarrow.parquet as pq

from segment_predictor.ingest.strava_segments import (
    fetch_and_store_segments,
    get_segment,
    save_segment,
)


def NO_SLEEP(seconds: float) -> None:  # noqa: N802
    raise AssertionError(f"no sleep expected, got {seconds}s")


def _fake_segment(segment_id: int) -> dict:
    return {
        "id": segment_id,
        "name": f"Segment {segment_id}",
        "distance": 1000.0,
        "average_grade": 5.0,
        "xoms": {"kom": "4:12"},
        "athlete_segment_stats": {"pr_elapsed_time": 300, "pr_date": "2023-01-01T00:00:00Z"},
    }


# ---- get_segment --------------------------------------------------------------------


def test_get_segment_requests_the_right_path_and_returns_raw_json() -> None:
    fake = _fake_segment(229781)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/segments/229781"
        return httpx.Response(200, json=fake)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    segment = get_segment(client, "tok", 229781, sleep=NO_SLEEP)

    assert segment == fake


# ---- save_segment ---------------------------------------------------------------------


def test_save_segment_writes_one_file_per_segment(tmp_path) -> None:
    raw_dir = tmp_path / "segments"

    written_path = save_segment(raw_dir, 229781, _fake_segment(229781))

    assert written_path == raw_dir / "229781.parquet"
    row = pq.read_table(written_path).to_pylist()[0]
    assert row["xoms"]["kom"] == "4:12"
    assert row["athlete_segment_stats"]["pr_elapsed_time"] == 300


# ---- fetch_and_store_segments : orchestration ------------------------------------------


def test_fetch_and_store_segments_skips_already_downloaded(tmp_path) -> None:
    raw_dir = tmp_path / "segments"
    save_segment(raw_dir, 1, _fake_segment(1))  # déjà téléchargé

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/segments/2"  # jamais appelé pour 1
        return httpx.Response(200, json=_fake_segment(2))

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_segments(client, "tok", [1, 2], raw_dir, sleep=NO_SLEEP)

    assert summary.fetched_ids == [2]
    assert summary.already_downloaded_ids == [1]
    assert summary.remaining_ids == []
    assert summary.stopped_due_to_daily_quota is False


def test_fetch_and_store_segments_writes_after_each_segment(tmp_path) -> None:
    raw_dir = tmp_path / "segments"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/segments/2":
            assert (raw_dir / "1.parquet").exists()
        return httpx.Response(200, json=_fake_segment(int(request.url.path.rsplit("/", 1)[-1])))

    client = httpx.Client(transport=httpx.MockTransport(handler))

    fetch_and_store_segments(client, "tok", [1, 2], raw_dir, sleep=NO_SLEEP)

    assert (raw_dir / "1.parquet").exists()
    assert (raw_dir / "2.parquet").exists()


def test_fetch_and_store_segments_pauses_then_continues_on_short_term_exhaustion(tmp_path) -> None:
    raw_dir = tmp_path / "segments"

    def handler(request: httpx.Request) -> httpx.Response:
        segment_id = int(request.url.path.rsplit("/", 1)[-1])
        if segment_id == 1:
            return httpx.Response(
                200,
                json=_fake_segment(1),
                headers={"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "200,50"},
            )
        return httpx.Response(200, json=_fake_segment(segment_id))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    summary = fetch_and_store_segments(
        client, "tok", [1, 2], raw_dir, sleep=lambda s: sleep_calls.append(s)
    )

    assert sleep_calls == [900.0]
    assert summary.fetched_ids == [1, 2]
    assert summary.stopped_due_to_daily_quota is False


def test_fetch_and_store_segments_stops_cleanly_when_daily_quota_reached(tmp_path) -> None:
    raw_dir = tmp_path / "segments"

    def handler(request: httpx.Request) -> httpx.Response:
        segment_id = int(request.url.path.rsplit("/", 1)[-1])
        if segment_id == 3:
            raise AssertionError("segment 3 should never be requested: daily quota already reached")
        if segment_id == 2:
            return httpx.Response(
                200,
                json=_fake_segment(2),
                headers={"X-RateLimit-Limit": "200,1000", "X-RateLimit-Usage": "51,1000"},
            )
        return httpx.Response(
            200,
            json=_fake_segment(1),
            headers={"X-RateLimit-Limit": "200,1000", "X-RateLimit-Usage": "50,999"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_segments(client, "tok", [1, 2, 3], raw_dir, sleep=NO_SLEEP)

    assert summary.fetched_ids == [1, 2]
    assert summary.remaining_ids == [3]
    assert summary.stopped_due_to_daily_quota is True
    assert not (raw_dir / "3.parquet").exists()


def test_fetch_and_store_segments_stops_cleanly_on_persistent_429_without_headers(tmp_path) -> None:
    raw_dir = tmp_path / "segments"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)  # jamais de succès, aucun en-tête de quota

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_segments(client, "tok", [1], raw_dir, sleep=lambda s: None)

    assert summary.fetched_ids == []
    assert summary.remaining_ids == [1]
    assert summary.stopped_due_to_daily_quota is True
