import httpx

from segment_predictor.ingest.strava_client import get_athlete


def test_get_athlete_sends_bearer_token_and_returns_json() -> None:
    captured_auth_header = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/athlete"
        captured_auth_header["value"] = request.headers["Authorization"]
        return httpx.Response(200, json={"firstname": "Manu", "lastname": "Ferry"})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    athlete = get_athlete(client, access_token="tok123")

    assert athlete["firstname"] == "Manu"
    assert captured_auth_header["value"] == "Bearer tok123"
