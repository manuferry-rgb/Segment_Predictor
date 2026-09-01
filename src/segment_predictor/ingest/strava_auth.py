"""Authentification OAuth Strava : obtention d'un access token valide.

Couche ingest : uniquement de l'I/O (appel réseau + lecture/écriture du
.env), aucune transformation de données métier.

Strava renvoie un NOUVEAU refresh_token à chaque rafraîchissement et
invalide l'ancien. On doit donc persister systématiquement les 3 valeurs
(access_token, refresh_token, expires_at) après un refresh, sinon l'auth
casse dès le refresh suivant. Le choix de stockage est le fichier .env
lui-même : c'est déjà le stockage des secrets du projet (gitignored),
et le projet est mono-utilisateur — une base ou un fichier séparé
serait une couche en trop pour 3 valeurs.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import dotenv_values, set_key

STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"


@dataclass(frozen=True)
class TokenState:
    """Un couple de tokens Strava valide à un instant donné."""

    access_token: str
    refresh_token: str
    expires_at: int  # timestamp unix (secondes), fourni par Strava


def refresh_access_token(
    http_client: httpx.Client, client_id: str, client_secret: str, refresh_token: str
) -> TokenState:
    """Échange un refresh_token contre un nouveau couple access/refresh.

    C'est un appel réseau pur : le client httpx est injecté plutôt que
    construit ici, pour pouvoir le remplacer par un MockTransport en test.
    """
    response = http_client.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
    )
    response.raise_for_status()
    data = response.json()
    return TokenState(
        access_token=data["access_token"],
        refresh_token=data["refresh_token"],
        expires_at=data["expires_at"],
    )


def _require(values: dict[str, str | None], key: str) -> str:
    """Lit une clé obligatoire dans le .env, sans valeur par défaut silencieuse."""
    value = values.get(key)
    if not value:
        raise KeyError(f"{key} manquant dans le .env — authentification Strava impossible")
    return value


def persist_tokens(env_path: Path, token_state: TokenState) -> None:
    """Réécrit les 3 clés Strava dans le .env, en place (rotation du refresh_token incluse)."""
    set_key(env_path, "STRAVA_ACCESS_TOKEN", token_state.access_token, quote_mode="never")
    set_key(env_path, "STRAVA_REFRESH_TOKEN", token_state.refresh_token, quote_mode="never")
    set_key(env_path, "STRAVA_EXPIRES_AT", str(token_state.expires_at), quote_mode="never")


def get_valid_access_token(
    http_client: httpx.Client, env_path: Path, now: int | None = None
) -> str:
    """Renvoie un access_token utilisable, en ne rafraîchissant que si nécessaire.

    L'access token Strava expire au bout de 6h (`expires_at`). On ne
    rafraîchit que si ce délai est dépassé, pour éviter de consommer le
    quota d'API Strava à chaque lancement.
    """
    values = dotenv_values(env_path)
    client_id = _require(values, "STRAVA_CLIENT_ID")
    client_secret = _require(values, "STRAVA_CLIENT_SECRET")
    refresh_token = _require(values, "STRAVA_REFRESH_TOKEN")

    cached_access_token = values.get("STRAVA_ACCESS_TOKEN")
    cached_expires_at = values.get("STRAVA_EXPIRES_AT")
    current_time = now if now is not None else int(time.time())

    if cached_access_token and cached_expires_at and int(cached_expires_at) > current_time:
        return cached_access_token

    token_state = refresh_access_token(http_client, client_id, client_secret, refresh_token)
    persist_tokens(env_path, token_state)
    return token_state.access_token
