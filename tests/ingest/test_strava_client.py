import httpx
import pytest

from segment_predictor.ingest.strava_client import (
    RateLimitStatus,
    authenticated_get,
    get_athlete,
    parse_rate_limit,
)


def NO_SLEEP(seconds: float) -> None:  # noqa: N802
    raise AssertionError(f"no sleep expected, got {seconds}s")


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


def test_parse_rate_limit_reads_both_headers() -> None:
    response = httpx.Response(
        200, headers={"X-RateLimit-Limit": "200,2000", "X-RateLimit-Usage": "10,150"}
    )

    status = parse_rate_limit(response)

    assert status == RateLimitStatus(
        usage_15min=10, limit_15min=200, usage_daily=150, limit_daily=2000
    )


def test_parse_rate_limit_returns_none_when_headers_absent() -> None:
    response = httpx.Response(200)

    assert parse_rate_limit(response) is None


def test_rate_limit_status_short_term_exhausted() -> None:
    assert RateLimitStatus(200, 200, 10, 2000).short_term_exhausted is True
    assert RateLimitStatus(199, 200, 10, 2000).short_term_exhausted is False


def test_authenticated_get_retries_after_429_using_retry_after_header() -> None:
    """Pas de limite codée en dur : le délai d'attente vient du header Retry-After."""
    responses = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(200, json={"ok": True}),
    ]
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses[call_count["value"]]
        call_count["value"] += 1
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    result = authenticated_get(
        client, "/athlete", "tok123", sleep=lambda seconds: sleep_calls.append(seconds)
    )

    assert result.json() == {"ok": True}
    assert sleep_calls == [3.0]
    assert call_count["value"] == 2


def test_authenticated_get_falls_back_to_default_backoff_without_retry_after_header() -> None:
    responses = [httpx.Response(429), httpx.Response(200, json={"ok": True})]
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses[call_count["value"]]
        call_count["value"] += 1
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    authenticated_get(
        client, "/athlete", "tok123", sleep=lambda seconds: sleep_calls.append(seconds)
    )

    assert sleep_calls == [900.0]


def test_authenticated_get_caps_an_excessive_retry_after_value() -> None:
    """Trouvé en conditions réelles sur T-07b : un sleep() non plafonné sur une
    donnée externe (Retry-After) a bloqué le script ~10-30 min à deux reprises,
    sans rapport avec le quota réel (vérifié indépendamment, quota sain).
    Retry-After n'est pas garanti raisonnable ; on ne lui fait pas confiance
    au-delà de notre propre fenêtre de repli."""
    responses = [
        httpx.Response(429, headers={"Retry-After": "7200"}),  # 2h, largement excessif
        httpx.Response(200, json={"ok": True}),
    ]
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses[call_count["value"]]
        call_count["value"] += 1
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    authenticated_get(
        client, "/athlete", "tok123", sleep=lambda seconds: sleep_calls.append(seconds)
    )

    assert sleep_calls == [900.0]  # plafonné, pas 7200


def test_authenticated_get_gives_up_after_max_retries() -> None:
    """Toujours 429 : l'erreur HTTP doit remonter au bout d'un moment, pas de boucle infinie."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    with pytest.raises(httpx.HTTPStatusError):
        authenticated_get(
            client, "/athlete", "tok123", sleep=lambda seconds: sleep_calls.append(seconds)
        )


def test_authenticated_get_raises_on_other_http_errors_without_retrying() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        authenticated_get(client, "/athlete", "tok123", sleep=NO_SLEEP)


def test_authenticated_get_retries_on_transient_network_error() -> None:
    """Un incident réseau transitoire (timeout, connexion coupée) n'est pas un
    429 : trouvé en conditions réelles sur T-07b (~500 requêtes séquentielles,
    un httpx.ReadTimeout a fait planter tout le script au bout de 311/497)."""
    call_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep_calls = []

    result = authenticated_get(
        client, "/athlete", "tok123", sleep=lambda seconds: sleep_calls.append(seconds)
    )

    assert result.json() == {"ok": True}
    assert call_count["value"] == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] < 60  # backoff court, pas la temporisation de quota (900s)


def test_authenticated_get_gives_up_after_repeated_network_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.TransportError):
        authenticated_get(client, "/athlete", "tok123", sleep=lambda s: None)
