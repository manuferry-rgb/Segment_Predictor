"""Tests de la récupération du profil réel d'un segment (T-51a).

`GET /segments/{id}/streams` renvoie distance/altitude/latlng le long du
TRACÉ OFFICIEL du segment — différent de `GET /activities/{id}/streams`
(ingest/strava_streams.py) qui porte sur UNE sortie précise. Même forme
de réponse Strava (key_by_type=true), même mécanique de quota que T-05/
T-06 (réutilisée ici, pas réinventée) : aucun appel réseau réel
(httpx.MockTransport), aucune attente réelle (`sleep` factice partout).
"""

import httpx
import pyarrow.parquet as pq

from segment_predictor.ingest.strava_segment_streams import (
    fetch_and_store_segment_streams,
    get_segment_streams,
    save_segment_streams,
)


def NO_SLEEP(seconds: float) -> None:  # noqa: N802
    raise AssertionError(f"no sleep expected, got {seconds}s")


def _fake_segment_streams(n: int = 3) -> dict:
    return {
        "distance": {
            "data": [i * 10.0 for i in range(n)],
            "series_type": "distance",
            "original_size": n,
            "resolution": "high",
        },
        "altitude": {
            "data": [100.0 + i for i in range(n)],
            "series_type": "distance",
            "original_size": n,
            "resolution": "high",
        },
        "latlng": {
            "data": [[45.0, 5.0 + i * 0.001] for i in range(n)],
            "series_type": "distance",
            "original_size": n,
            "resolution": "high",
        },
    }


# ---- get_segment_streams ----------------------------------------------------------------


def test_get_segment_streams_requests_the_right_keys_and_returns_raw_json() -> None:
    fake = _fake_segment_streams()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/segments/229781/streams"
        assert request.url.params["keys"] == "distance,altitude,latlng"
        assert request.url.params["key_by_type"] == "true"
        return httpx.Response(200, json=fake)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    streams = get_segment_streams(client, "tok", 229781, sleep=NO_SLEEP)

    assert streams == fake


# ---- save_segment_streams ---------------------------------------------------------------


def test_save_segment_streams_writes_one_file_per_segment(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"

    written_path = save_segment_streams(raw_dir, 229781, _fake_segment_streams())

    assert written_path == raw_dir / "229781.parquet"
    row = pq.read_table(written_path).to_pylist()[0]
    assert row["altitude"]["data"] == [100.0, 101.0, 102.0]


# ---- fetch_and_store_segment_streams : orchestration -------------------------------------


def test_fetch_and_store_segment_streams_skips_already_downloaded(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"
    save_segment_streams(raw_dir, 1, _fake_segment_streams())  # déjà téléchargé

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/segments/2/streams"  # jamais appelé pour 1
        return httpx.Response(200, json=_fake_segment_streams())

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_segment_streams(client, "tok", [1, 2], raw_dir, sleep=NO_SLEEP)

    assert summary.fetched_ids == [2]
    assert summary.already_downloaded_ids == [1]
    assert summary.remaining_ids == []
    assert summary.stopped_due_to_daily_quota is False


def test_fetch_and_store_segment_streams_writes_after_each_segment(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/segments/2/streams":
            assert (raw_dir / "1.parquet").exists()
        return httpx.Response(200, json=_fake_segment_streams())

    client = httpx.Client(transport=httpx.MockTransport(handler))

    fetch_and_store_segment_streams(client, "tok", [1, 2], raw_dir, sleep=NO_SLEEP)

    assert (raw_dir / "1.parquet").exists()
    assert (raw_dir / "2.parquet").exists()


def test_fetch_and_store_segment_streams_pauses_then_continues_on_short_term_exhaustion(
    tmp_path,
) -> None:
    raw_dir = tmp_path / "segment_streams"

    def handler(request: httpx.Request) -> httpx.Response:
        segment_id = int(request.url.path.split("/")[-2])
        if segment_id == 1:
            return httpx.Response(
                200,
                json=_fake_segment_streams(),
                headers={"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "200,50"},
            )
        return httpx.Response(200, json=_fake_segment_streams())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    summary = fetch_and_store_segment_streams(
        client, "tok", [1, 2], raw_dir, sleep=lambda s: sleep_calls.append(s)
    )

    assert sleep_calls == [900.0]
    assert summary.fetched_ids == [1, 2]
    assert summary.stopped_due_to_daily_quota is False


def test_fetch_and_store_segment_streams_stops_cleanly_when_daily_quota_reached(tmp_path) -> None:
    raw_dir = tmp_path / "segment_streams"

    def handler(request: httpx.Request) -> httpx.Response:
        segment_id = int(request.url.path.split("/")[-2])
        if segment_id == 3:
            raise AssertionError("segment 3 should never be requested: daily quota already reached")
        if segment_id == 2:
            return httpx.Response(
                200,
                json=_fake_segment_streams(),
                headers={"X-RateLimit-Limit": "200,1000", "X-RateLimit-Usage": "51,1000"},
            )
        return httpx.Response(
            200,
            json=_fake_segment_streams(),
            headers={"X-RateLimit-Limit": "200,1000", "X-RateLimit-Usage": "50,999"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_segment_streams(client, "tok", [1, 2, 3], raw_dir, sleep=NO_SLEEP)

    assert summary.fetched_ids == [1, 2]
    assert summary.remaining_ids == [3]
    assert summary.stopped_due_to_daily_quota is True
    assert not (raw_dir / "3.parquet").exists()


def test_fetch_and_store_segment_streams_stops_cleanly_on_persistent_429_without_headers(
    tmp_path,
) -> None:
    raw_dir = tmp_path / "segment_streams"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)  # jamais de succès, aucun en-tête de quota

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_segment_streams(client, "tok", [1], raw_dir, sleep=lambda s: None)

    assert summary.fetched_ids == []
    assert summary.remaining_ids == [1]
    assert summary.stopped_due_to_daily_quota is True
