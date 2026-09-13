"""Authentification OAuth Strava : obtention d'un access token valide.

Couche ingest : uniquement de l'I/O (appel réseau + lecture/écriture du
.env), aucune transformation de données métier.

Strava renvoie un NOUVEAU refresh_token à chaque rafraîchissement et
invalide l'ancien. On doit donc persister systématiquement les 3 valeurs
(access_token, refresh_token, expires_at) après un refresh, sinon l'auth
casse dès le refresh suivant. `refresh_access_token`/`get_valid_access_
token`/`persist_tokens` restent tels quels depuis l'époque mono-
utilisateur du projet : les scripts CLI (fetch_activities.py, etc.)
utilisent toujours le .env de l'auteur pour lire SES propres données.

`exchange_authorization_code` (T-45) est différente : c'est l'échange
utilisé par le flux web "Se connecter avec Strava", où N'IMPORTE QUI
peut s'autoriser — ses tokens sont alors stockés dans `users`
(storage/users.py), pas dans .env.
"""

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import httpx
from dotenv import dotenv_values

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


@dataclass(frozen=True)
class AuthorizedAthlete:
    """Résultat d'un échange `authorization_code` -> tokens (T-45).

    Contrairement à `refresh_access_token`, cette réponse contient AUSSI
    le profil de l'athlète qui vient de s'autoriser — la seule fois où
    Strava le fournit dans la réponse de /oauth/token (pas au
    rafraîchissement) : c'est ce qui permet de savoir QUI se connecte,
    sans lui poser la question ni faire un second appel.
    """

    athlete_id: int
    firstname: str | None
    lastname: str | None
    token_state: TokenState


def exchange_authorization_code(
    http_client: httpx.Client, client_id: str, client_secret: str, code: str
) -> AuthorizedAthlete:
    """Échange le `code` reçu sur `/auth/strava/callback` (T-45) contre les
    tokens de CETTE personne et son identité Strava.

    `firstname`/`lastname` peuvent être absents (Strava ne les garantit
    pas) — `None`, pas une KeyError : ce ne sont que des champs d'affichage
    ("connecté comme Prénom"), rien n'en dépend fonctionnellement.
    """
    response = http_client.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
        },
    )
    response.raise_for_status()
    data = response.json()
    athlete = data["athlete"]
    return AuthorizedAthlete(
        athlete_id=athlete["id"],
        firstname=athlete.get("firstname"),
        lastname=athlete.get("lastname"),
        token_state=TokenState(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
        ),
    )


def _require(values: dict[str, str | None], key: str) -> str:
    """Lit une clé obligatoire dans le .env, sans valeur par défaut silencieuse."""
    value = values.get(key)
    if not value:
        raise KeyError(f"{key} manquant dans le .env — authentification Strava impossible")
    return value


def persist_tokens(env_path: Path, token_state: TokenState) -> None:
    """Réécrit les 3 clés Strava dans le .env en une seule écriture atomique.

    Une seule écriture (fichier temporaire + os.replace), plutôt que 3
    appels séparés : si le process meurt en cours de route, le .env
    reste soit entièrement dans l'ancien état, soit entièrement dans le
    nouveau — jamais un mélange des deux. Un mélange serait pire qu'une
    perte totale : l'ancien refresh_token est déjà mort côté Strava une
    fois le nouveau émis, donc un access_token neuf combiné à un
    refresh_token périmé casse silencieusement le prochain rafraîchissement.
    """
    values = dict(dotenv_values(env_path))
    values["STRAVA_ACCESS_TOKEN"] = token_state.access_token
    values["STRAVA_REFRESH_TOKEN"] = token_state.refresh_token
    values["STRAVA_EXPIRES_AT"] = str(token_state.expires_at)
    content = "".join(f"{key}={value}\n" for key, value in values.items())

    # tempfile dans le même dossier que .env : os.replace n'est atomique
    # que si source et destination sont sur le même système de fichiers.
    fd, tmp_path = tempfile.mkstemp(dir=env_path.parent, prefix=".env.tmp-")
    try:
        with os.fdopen(fd, "w") as tmp_file:
            tmp_file.write(content)
        os.replace(tmp_path, env_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


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


def get_valid_access_token_for_user(
    http_client: httpx.Client,
    conn: duckdb.DuckDBPyConnection,
    client_id: str,
    client_secret: str,
    user_id: int,
    now: int | None = None,
) -> str:
    """Équivalent de `get_valid_access_token` (ci-dessus) pour un
    utilisateur du flux web "Se connecter avec Strava" (T-45/T-46) : le
    token vit dans `users` (storage/users.py), pas dans `.env` — chaque
    utilisateur a le sien, `.env` ne connaît que celui de l'auteur.

    Import de `storage.users` fait ICI (pas en tête de module) : cette
    fonction est la seule de tout `ingest/` à en dépendre — le reste du
    fichier reste "juste du .env", garder l'import local évite de faire
    croire que tout `strava_auth.py` dépend de la couche storage.

    `client_id`/`client_secret` restent ceux de l'application Strava
    elle-même (partagés par tout le monde, dans `.env`) — seul le
    `refresh_token` change selon l'utilisateur.
    """
    from segment_predictor.storage.users import get_user, upsert_user

    user = get_user(conn, user_id)
    if user is None:
        raise ValueError(f"utilisateur {user_id} introuvable dans `users`")

    current_time = now if now is not None else int(time.time())
    if user.expires_at > current_time:
        return user.access_token

    token_state = refresh_access_token(http_client, client_id, client_secret, user.refresh_token)
    upsert_user(
        conn,
        id=user_id,
        firstname=user.firstname,
        lastname=user.lastname,
        access_token=token_state.access_token,
        refresh_token=token_state.refresh_token,
        expires_at=token_state.expires_at,
    )
    return token_state.access_token
