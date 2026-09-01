"""Tests du rafraîchissement de token Strava — aucun appel réseau réel.

Le httpx.Client reçoit un MockTransport : une fonction qui joue le rôle
du serveur Strava et renvoie une réponse HTTP fabriquée à la main.
"""

import os

import httpx
import pytest
from dotenv import dotenv_values

from segment_predictor.ingest.strava_auth import (
    TokenState,
    get_valid_access_token,
    persist_tokens,
    refresh_access_token,
)


def test_refresh_access_token_returns_new_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth/token"
        body = request.read()
        assert b"grant_type=refresh_token" in body
        assert b"refresh_token=old_refresh" in body
        return httpx.Response(
            200,
            json={
                "token_type": "Bearer",
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_at": 1_700_000_000,
                "expires_in": 21600,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = refresh_access_token(
        client, client_id="id", client_secret="secret", refresh_token="old_refresh"
    )

    assert result == TokenState(
        access_token="new_access", refresh_token="new_refresh", expires_at=1_700_000_000
    )


def test_get_valid_access_token_reuses_cached_token_when_not_expired(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "STRAVA_CLIENT_ID=id\n"
        "STRAVA_CLIENT_SECRET=secret\n"
        "STRAVA_REFRESH_TOKEN=refresh\n"
        "STRAVA_ACCESS_TOKEN=cached_access\n"
        "STRAVA_EXPIRES_AT=2000000000\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call expected: the cached token is still valid")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    token = get_valid_access_token(client, env_path, now=1_000_000_000)

    assert token == "cached_access"


def test_get_valid_access_token_refreshes_and_persists_when_expired(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "STRAVA_CLIENT_ID=id\n"
        "STRAVA_CLIENT_SECRET=secret\n"
        "STRAVA_REFRESH_TOKEN=old_refresh\n"
        "STRAVA_ACCESS_TOKEN=expired_access\n"
        "STRAVA_EXPIRES_AT=100\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "new_access",
                "refresh_token": "new_refresh",
                "expires_at": 999_999_999,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    token = get_valid_access_token(client, env_path, now=500)

    assert token == "new_access"
    # La rotation doit être persistée : nouveau refresh_token, pas l'ancien.
    persisted = dotenv_values(env_path)
    assert persisted["STRAVA_ACCESS_TOKEN"] == "new_access"
    assert persisted["STRAVA_REFRESH_TOKEN"] == "new_refresh"
    assert persisted["STRAVA_EXPIRES_AT"] == "999999999"


def test_get_valid_access_token_refreshes_when_no_cached_token_yet(tmp_path) -> None:
    """Premier lancement : .env ne contient que les 3 valeurs initiales."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "STRAVA_CLIENT_ID=id\nSTRAVA_CLIENT_SECRET=secret\nSTRAVA_REFRESH_TOKEN=refresh\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "first_access",
                "refresh_token": "rotated_refresh",
                "expires_at": 999_999_999,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))

    token = get_valid_access_token(client, env_path, now=500)

    assert token == "first_access"


def test_get_valid_access_token_raises_explicitly_when_client_id_missing(tmp_path) -> None:
    """Pas de valeur par défaut silencieuse : une clé manquante lève une erreur claire."""
    env_path = tmp_path / ".env"
    env_path.write_text("STRAVA_REFRESH_TOKEN=refresh\n")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call expected: should fail before any request")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(KeyError, match="STRAVA_CLIENT_ID"):
        get_valid_access_token(client, env_path, now=0)


def test_persist_tokens_writes_all_keys_in_one_atomic_replace(tmp_path, monkeypatch) -> None:
    """Les 3 clés doivent être écrites en un seul os.replace, pas 3 séparés —
    sinon un crash entre deux écritures laisserait un état mélangé
    (ex. nouvel access_token + ancien refresh_token, déjà mort côté Strava).
    """
    env_path = tmp_path / ".env"
    env_path.write_text(
        "STRAVA_CLIENT_ID=id\n"
        "STRAVA_CLIENT_SECRET=secret\n"
        "STRAVA_REFRESH_TOKEN=old_refresh\n"
        "STRAVA_ACCESS_TOKEN=old_access\n"
        "STRAVA_EXPIRES_AT=100\n"
    )

    real_replace = os.replace
    calls_made = {"count": 0}

    def counting_replace(*args, **kwargs):
        calls_made["count"] += 1
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(os, "replace", counting_replace)

    token_state = TokenState(access_token="new_access", refresh_token="new_refresh", expires_at=999)
    persist_tokens(env_path, token_state)

    assert calls_made["count"] == 1
    persisted = dotenv_values(env_path)
    assert persisted["STRAVA_ACCESS_TOKEN"] == "new_access"
    assert persisted["STRAVA_REFRESH_TOKEN"] == "new_refresh"
    assert persisted["STRAVA_EXPIRES_AT"] == "999"


def test_persist_tokens_leaves_file_untouched_if_write_fails(tmp_path, monkeypatch) -> None:
    """Si l'écriture atomique échoue (simulateur de crash), le .env doit
    rester exactement dans son ancien état — pas de mélange, pas de fichier
    temporaire résiduel.
    """
    env_path = tmp_path / ".env"
    original_content = (
        "STRAVA_CLIENT_ID=id\n"
        "STRAVA_CLIENT_SECRET=secret\n"
        "STRAVA_REFRESH_TOKEN=old_refresh\n"
        "STRAVA_ACCESS_TOKEN=old_access\n"
        "STRAVA_EXPIRES_AT=100\n"
    )
    env_path.write_text(original_content)

    def failing_replace(*args, **kwargs):
        raise OSError("simulated crash during os.replace")

    monkeypatch.setattr(os, "replace", failing_replace)

    token_state = TokenState(access_token="new_access", refresh_token="new_refresh", expires_at=999)

    with pytest.raises(OSError):
        persist_tokens(env_path, token_state)

    persisted = dotenv_values(env_path)
    assert persisted["STRAVA_ACCESS_TOKEN"] == "old_access"
    assert persisted["STRAVA_REFRESH_TOKEN"] == "old_refresh"
    leftover_tmp_files = [p for p in tmp_path.iterdir() if p.name != ".env"]
    assert leftover_tmp_files == []
