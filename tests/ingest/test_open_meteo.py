"""Tests de la récupération météo historique Open-Meteo (T-14).

Aucun appel réseau réel (httpx.MockTransport), aucune attente réelle
(un `sleep` factice est injecté partout).
"""

import httpx
import pyarrow.parquet as pq
import pytest

from segment_predictor.ingest.open_meteo import (
    WeatherZoneRequest,
    compute_weather_zones,
    fetch_and_store_weather_zones,
    get_historical_weather,
    save_weather_zone,
    zone_filename,
    zone_key,
)


def NO_SLEEP(seconds: float) -> None:  # noqa: N802
    raise AssertionError(f"no sleep expected, got {seconds}s")


def _fake_weather() -> dict:
    return {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [5.0, 5.8],
            "relative_humidity_2m": [79, 72],
            "surface_pressure": [979.5, 980.1],
            "wind_speed_10m": [16.6, 21.9],
            "wind_direction_10m": [270, 280],
        }
    }


# ---- zone_key / compute_weather_zones (pures, pas d'I/O) --------------------------------


def test_zone_key_rounds_to_the_grid() -> None:
    assert zone_key(47.74, 7.42, grid_degrees=0.1) == pytest.approx((47.7, 7.4))
    assert zone_key(47.76, 7.46, grid_degrees=0.1) == pytest.approx((47.8, 7.5))


def test_compute_weather_zones_groups_by_rounded_location() -> None:
    locations = [
        (47.71, 7.41, "2024-01-01T10:00:00"),
        (47.74, 7.44, "2024-03-15T08:00:00"),  # même zone à 0.1°
        (-11.6, 167.0, "2023-11-09T06:00:00"),  # zone différente
    ]

    zones = compute_weather_zones(locations, grid_degrees=0.1)

    assert len(zones) == 2
    zone_alsace = next(z for z in zones if z.zone_lat == 47.7)
    assert zone_alsace.zone_lng == 7.4
    assert zone_alsace.start_date == "2024-01-01"
    assert zone_alsace.end_date == "2024-03-15"


def test_compute_weather_zones_raises_on_non_positive_grid() -> None:
    with pytest.raises(ValueError, match="grid_degrees"):
        compute_weather_zones([(47.7, 7.4, "2024-01-01T00:00:00")], grid_degrees=0.0)


# ---- get_historical_weather --------------------------------------------------------------


def test_get_historical_weather_requests_the_right_params_and_returns_raw_json() -> None:
    fake = _fake_weather()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/archive"
        assert request.url.params["latitude"] == "47.7"
        assert request.url.params["longitude"] == "7.4"
        assert request.url.params["start_date"] == "2024-01-01"
        assert request.url.params["end_date"] == "2024-01-02"
        assert request.url.params["timezone"] == "UTC"
        assert "wind_speed_10m" in request.url.params["hourly"]
        return httpx.Response(200, json=fake)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    weather = get_historical_weather(client, 47.7, 7.4, "2024-01-01", "2024-01-02", sleep=NO_SLEEP)

    assert weather == fake


def test_get_historical_weather_retries_on_429() -> None:
    responses = [
        httpx.Response(429, headers={"Retry-After": "2"}),
        httpx.Response(200, json=_fake_weather()),
    ]
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses[call_count["value"]]
        call_count["value"] += 1
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    get_historical_weather(
        client, 47.7, 7.4, "2024-01-01", "2024-01-02", sleep=lambda s: sleep_calls.append(s)
    )

    assert sleep_calls == [2.0]


# ---- save_weather_zone / fetch_and_store_weather_zones -----------------------------------


def test_save_weather_zone_writes_one_file_per_zone(tmp_path) -> None:
    raw_dir = tmp_path / "weather"

    written_path = save_weather_zone(raw_dir, 47.7, 7.4, _fake_weather())

    assert written_path == raw_dir / zone_filename(47.7, 7.4)
    row = pq.read_table(written_path).to_pylist()[0]
    assert row["hourly"]["temperature_2m"] == [5.0, 5.8]


def test_fetch_and_store_weather_zones_skips_already_downloaded(tmp_path) -> None:
    raw_dir = tmp_path / "weather"
    save_weather_zone(raw_dir, 47.7, 7.4, _fake_weather())  # déjà téléchargée

    requests = [
        WeatherZoneRequest(47.7, 7.4, "2024-01-01", "2024-01-02"),
        WeatherZoneRequest(-11.6, 167.0, "2023-11-09", "2023-11-10"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["latitude"] == "-11.6"  # jamais appelée pour 47.7
        return httpx.Response(200, json=_fake_weather())

    client = httpx.Client(transport=httpx.MockTransport(handler))

    summary = fetch_and_store_weather_zones(client, requests, raw_dir, sleep=NO_SLEEP)

    assert summary.fetched_zones == [(-11.6, 167.0)]
    assert summary.already_downloaded_zones == [(47.7, 7.4)]
