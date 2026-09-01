"""Tests de la récupération des streams Strava (T-05).

Aucun appel réseau réel (httpx.MockTransport), aucune attente réelle
(un `sleep` factice est injecté partout), aucun git réel en dehors du
dépôt temporaire créé pour tester ensure_path_is_gitignored.
"""

import subprocess

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from segment_predictor.ingest.strava_streams import (
    ensure_path_is_gitignored,
    fetch_and_store_streams,
    get_activity_streams,
    list_eligible_activity_ids,
    save_streams,
)


def NO_SLEEP(seconds: float) -> None:  # noqa: N802
    raise AssertionError(f"no sleep expected, got {seconds}s")


def _write_activities(raw_dir, rows: list[dict]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), raw_dir / "activities_test.parquet")


def _activity_row(activity_id: int, activity_type: str, device_watts: bool) -> dict:
    return {"id": activity_id, "type": activity_type, "device_watts": device_watts}


# ---- list_eligible_activity_ids ---------------------------------------------------


def test_list_eligible_activity_ids_filters_type_and_device_watts(tmp_path) -> None:
    raw_dir = tmp_path / "activities"
    _write_activities(
        raw_dir,
        [
            _activity_row(1, "Ride", True),  # éligible
            _activity_row(2, "Run", True),  # mauvais type
            _activity_row(3, "VirtualRide", False),  # pas de capteur de puissance
            _activity_row(4, "VirtualRide", True),  # éligible
        ],
    )

    assert list_eligible_activity_ids(raw_dir) == [1, 4]


# ---- get_activity_streams -----------------------------------------------------------


def test_get_activity_streams_requests_the_right_keys_and_returns_raw_json() -> None:
    fake_streams = {
        "time": {
            "data": [0, 1, 2],
            "series_type": "distance",
            "original_size": 3,
            "resolution": "high",
        },
        "watts": {
            "data": [100, 150, 200],
            "series_type": "distance",
            "original_size": 3,
            "resolution": "high",
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/activities/123/streams"
        assert request.url.params["keys"] == "time,watts,altitude,latlng,distance,heartrate,cadence"
        assert request.url.params["key_by_type"] == "true"
        return httpx.Response(200, json=fake_streams)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    streams = get_activity_streams(client, "tok", 123, sleep=NO_SLEEP)

    assert streams == fake_streams


# ---- save_streams : un fichier par activité, forme brute préservée ------------------


def test_save_streams_preserves_raw_nested_shape(tmp_path) -> None:
    streams_raw_dir = tmp_path / "streams"
    streams = {
        "time": {
            "data": [0, 1, 2],
            "series_type": "distance",
            "original_size": 3,
            "resolution": "high",
        },
        "latlng": {
            "data": [[45.1, 5.2], [45.2, 5.3]],
            "series_type": "distance",
            "original_size": 2,
            "resolution": "high",
        },
    }

    written_path = save_streams(streams_raw_dir, 123, streams)

    assert written_path == streams_raw_dir / "123.parquet"
    row = pq.read_table(written_path).to_pylist()[0]
    assert row["time"]["data"] == [0, 1, 2]
    assert row["latlng"]["data"] == [[45.1, 5.2], [45.2, 5.3]]
    assert row["time"]["resolution"] == "high"


# ---- fetch_and_store_streams : orchestration ----------------------------------------


def _setup_two_eligible_activities(tmp_path):
    activities_raw_dir = tmp_path / "activities"
    streams_raw_dir = tmp_path / "streams"
    _write_activities(
        activities_raw_dir, [_activity_row(1, "Ride", True), _activity_row(2, "Ride", True)]
    )
    return activities_raw_dir, streams_raw_dir


def test_fetch_and_store_streams_skips_already_downloaded(tmp_path) -> None:
    activities_raw_dir, streams_raw_dir = _setup_two_eligible_activities(tmp_path)
    save_streams(streams_raw_dir, 1, {"time": {"data": [0]}})  # déjà téléchargée

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/activities/2/streams"  # jamais appelée pour 1
        return httpx.Response(200, json={"time": {"data": [0]}})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_streams(
        client, "tok", activities_raw_dir, streams_raw_dir, sleep=NO_SLEEP
    )

    assert summary.fetched_activity_ids == [2]
    assert summary.already_downloaded_activity_ids == [1]
    assert summary.remaining_activity_ids == []
    assert summary.stopped_due_to_daily_quota is False


def test_fetch_and_store_streams_writes_after_each_activity(tmp_path) -> None:
    """Pas d'écriture groupée à la fin : le fichier de l'activité 1 doit déjà
    exister sur disque quand la requête pour l'activité 2 part."""
    activities_raw_dir, streams_raw_dir = _setup_two_eligible_activities(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/activities/2/streams":
            assert (streams_raw_dir / "1.parquet").exists()
        return httpx.Response(200, json={"time": {"data": [0]}})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    fetch_and_store_streams(client, "tok", activities_raw_dir, streams_raw_dir, sleep=NO_SLEEP)

    assert (streams_raw_dir / "1.parquet").exists()
    assert (streams_raw_dir / "2.parquet").exists()


def test_fetch_and_store_streams_pauses_then_continues_on_short_term_exhaustion(tmp_path) -> None:
    activities_raw_dir, streams_raw_dir = _setup_two_eligible_activities(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/activities/1/streams":
            return httpx.Response(
                200,
                json={"time": {"data": [0]}},
                headers={"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "200,50"},
            )
        return httpx.Response(200, json={"time": {"data": [0]}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    summary = fetch_and_store_streams(
        client, "tok", activities_raw_dir, streams_raw_dir, sleep=lambda s: sleep_calls.append(s)
    )

    assert sleep_calls == [900.0]
    assert summary.fetched_activity_ids == [1, 2]
    assert summary.stopped_due_to_daily_quota is False


def test_fetch_and_store_streams_stops_cleanly_when_daily_quota_reached(tmp_path) -> None:
    activities_raw_dir = tmp_path / "activities"
    streams_raw_dir = tmp_path / "streams"
    _write_activities(
        activities_raw_dir,
        [
            _activity_row(1, "Ride", True),
            _activity_row(2, "Ride", True),
            _activity_row(3, "Ride", True),
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/activities/3/streams":
            raise AssertionError(
                "activity 3 should never be requested: daily quota already reached"
            )
        if request.url.path == "/api/v3/activities/2/streams":
            return httpx.Response(
                200,
                json={"time": {"data": [0]}},
                headers={"X-RateLimit-Limit": "200,1000", "X-RateLimit-Usage": "51,1000"},
            )
        return httpx.Response(
            200,
            json={"time": {"data": [0]}},
            headers={"X-RateLimit-Limit": "200,1000", "X-RateLimit-Usage": "50,999"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_streams(
        client, "tok", activities_raw_dir, streams_raw_dir, sleep=NO_SLEEP
    )

    assert summary.fetched_activity_ids == [1, 2]
    assert summary.remaining_activity_ids == [3]
    assert summary.stopped_due_to_daily_quota is True
    # Les 2 premières doivent bien être sur disque malgré l'arrêt.
    assert (streams_raw_dir / "1.parquet").exists()
    assert (streams_raw_dir / "2.parquet").exists()
    assert not (streams_raw_dir / "3.parquet").exists()


def test_fetch_and_store_streams_stops_cleanly_on_persistent_429_without_headers(tmp_path) -> None:
    """Filet de sécurité : même sans en-têtes de quota exploitables, un 429
    persistant après tous les retries s'arrête proprement au lieu de planter."""
    activities_raw_dir = tmp_path / "activities"
    streams_raw_dir = tmp_path / "streams"
    _write_activities(activities_raw_dir, [_activity_row(1, "Ride", True)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)  # jamais de succès, aucun en-tête de quota

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_streams(
        client, "tok", activities_raw_dir, streams_raw_dir, sleep=lambda s: None
    )

    assert summary.fetched_activity_ids == []
    assert summary.remaining_activity_ids == [1]
    assert summary.stopped_due_to_daily_quota is True


# ---- ensure_path_is_gitignored -------------------------------------------------------


def test_ensure_path_is_gitignored_passes_for_an_ignored_path(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("ignored/\n")

    ensure_path_is_gitignored(tmp_path / "ignored" / "sub", tmp_path)  # ne doit pas lever


def test_ensure_path_is_gitignored_raises_for_an_untracked_path(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("ignored/\n")

    with pytest.raises(RuntimeError, match="gitignor"):
        ensure_path_is_gitignored(tmp_path / "not_ignored", tmp_path)
