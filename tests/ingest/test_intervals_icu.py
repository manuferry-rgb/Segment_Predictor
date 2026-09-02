"""Tests de la récupération wellness intervals.icu (T-22).

Aucun appel réseau réel (httpx.MockTransport), aucune attente réelle
(un `sleep` factice est injecté partout).
"""

import base64

import httpx
import pyarrow.parquet as pq
import pytest

from segment_predictor.ingest.intervals_icu import (
    fetch_and_store_wellness,
    get_wellness,
    read_credentials,
    save_wellness,
)


def NO_SLEEP(seconds: float) -> None:  # noqa: N802
    raise AssertionError(f"no sleep expected, got {seconds}s")


def _fake_wellness() -> list[dict]:
    return [
        {"id": "2025-06-01", "hrv": 65.0, "sleepSecs": 27000, "restingHR": 48, "weight": 78.2},
        {"id": "2025-06-02", "hrv": None, "sleepSecs": 25200, "restingHR": 50, "weight": None},
    ]


# ---- get_wellness -----------------------------------------------------------------------


def test_get_wellness_requests_the_right_path_params_and_auth() -> None:
    fake = _fake_wellness()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/athlete/12345/wellness.json"
        assert request.url.params["oldest"] == "2025-06-01"
        assert request.url.params["newest"] == "2025-06-02"
        assert request.headers["Authorization"].startswith("Basic ")
        return httpx.Response(200, json=fake)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    wellness = get_wellness(
        client, "12345", "fake_api_key", "2025-06-01", "2025-06-02", sleep=NO_SLEEP
    )

    assert wellness == fake


def test_get_wellness_uses_api_key_basic_auth_convention() -> None:
    """intervals.icu : Basic Auth avec utilisateur "API_KEY" et mot de
    passe la vraie clé — pas d'OAuth contrairement à Strava."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth_header"] = request.headers["Authorization"]
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    get_wellness(client, "12345", "secret123", "2025-06-01", "2025-06-02", sleep=NO_SLEEP)

    expected_token = base64.b64encode(b"API_KEY:secret123").decode("ascii")
    assert captured["auth_header"] == f"Basic {expected_token}"


def test_get_wellness_retries_on_429() -> None:
    responses = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(200, json=_fake_wellness()),
    ]
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses[call_count["value"]]
        call_count["value"] += 1
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    get_wellness(
        client, "12345", "key", "2025-06-01", "2025-06-02", sleep=lambda s: sleep_calls.append(s)
    )

    assert sleep_calls == [3.0]


# ---- save_wellness / fetch_and_store_wellness --------------------------------------------


def test_save_wellness_writes_one_overwritable_file(tmp_path) -> None:
    raw_dir = tmp_path / "wellness"

    written_path = save_wellness(raw_dir, _fake_wellness())

    assert written_path == raw_dir / "wellness.parquet"
    rows = pq.read_table(written_path).to_pylist()
    assert rows[0]["id"] == "2025-06-01"
    assert rows[0]["hrv"] == 65.0


def test_save_wellness_overwrites_previous_fetch(tmp_path) -> None:
    """Contrairement aux activités/streams Strava (accumulés fichier par
    fichier), le wellness peut être corrigé rétroactivement — chaque
    fetch écrase le précédent plutôt que de s'accumuler."""
    raw_dir = tmp_path / "wellness"
    save_wellness(raw_dir, _fake_wellness())

    updated = [
        {"id": "2025-06-01", "hrv": 70.0, "sleepSecs": 27000, "restingHR": 48, "weight": 78.0}
    ]
    save_wellness(raw_dir, updated)

    rows = pq.read_table(raw_dir / "wellness.parquet").to_pylist()
    assert len(rows) == 1
    assert rows[0]["hrv"] == 70.0


def test_fetch_and_store_wellness_fetches_and_saves(tmp_path) -> None:
    raw_dir = tmp_path / "wellness"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_fake_wellness())

    client = httpx.Client(transport=httpx.MockTransport(handler))

    written_path = fetch_and_store_wellness(
        client, "12345", "key", "2025-06-01", "2025-06-02", raw_dir, sleep=NO_SLEEP
    )

    assert written_path == raw_dir / "wellness.parquet"
    assert len(pq.read_table(written_path).to_pylist()) == 2


# ---- read_credentials ---------------------------------------------------------------------


def test_read_credentials_reads_both_keys_from_env_file(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("INTERVALS_ICU_ATHLETE_ID=12345\nINTERVALS_ICU_API_KEY=secretkey\n")

    athlete_id, api_key = read_credentials(env_path)

    assert athlete_id == "12345"
    assert api_key == "secretkey"


def test_read_credentials_raises_when_athlete_id_missing(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("INTERVALS_ICU_API_KEY=secretkey\n")

    with pytest.raises(KeyError, match="INTERVALS_ICU_ATHLETE_ID"):
        read_credentials(env_path)


def test_read_credentials_raises_when_api_key_missing(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("INTERVALS_ICU_ATHLETE_ID=12345\n")

    with pytest.raises(KeyError, match="INTERVALS_ICU_API_KEY"):
        read_credentials(env_path)
