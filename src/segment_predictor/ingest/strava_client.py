"""Appels bruts à l'API Strava (couche ingest : aucune transformation).

Gestion du rate limit : Strava expose les compteurs réels via les
en-têtes X-RateLimit-Limit / X-RateLimit-Usage ("15min,daily"). On lit
ces valeurs plutôt que de coder les quotas en dur, car ils sont propres
à chaque app et peuvent changer. Seule la durée de la fenêtre courte
(15 minutes) est une constante de repli : c'est une cadence protocolaire
fixe, pas une limite chiffrée, utilisée seulement quand Strava ne
renvoie pas de Retry-After exploitable.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

STRAVA_API_BASE_URL = "https://www.strava.com/api/v3"

DEFAULT_RATE_LIMIT_BACKOFF_S = 900.0  # durée de la fenêtre courte de Strava
MAX_RATE_LIMIT_RETRIES = 5

# Incident réseau transitoire (timeout, connexion coupée) : rien à voir avec
# le quota, une courte pause suffit. Trouvé en conditions réelles sur T-07b —
# ~500 requêtes séquentielles sans retry sur ce cas, un httpx.ReadTimeout a
# fait planter tout le script au bout de 311/497.
NETWORK_ERROR_BACKOFF_S = 5.0
MAX_NETWORK_ERROR_RETRIES = 3


@dataclass(frozen=True)
class RateLimitStatus:
    """Quota Strava au moment d'une réponse, lu depuis ses en-têtes."""

    usage_15min: int
    limit_15min: int
    usage_daily: int
    limit_daily: int

    @property
    def short_term_exhausted(self) -> bool:
        return self.usage_15min >= self.limit_15min


def parse_rate_limit(response: httpx.Response) -> RateLimitStatus | None:
    """Lit X-RateLimit-Limit / X-RateLimit-Usage. None si absents de la réponse."""
    limit_header = response.headers.get("X-RateLimit-Limit")
    usage_header = response.headers.get("X-RateLimit-Usage")
    if not limit_header or not usage_header:
        return None
    limit_15min, limit_daily = (int(x) for x in limit_header.split(","))
    usage_15min, usage_daily = (int(x) for x in usage_header.split(","))
    return RateLimitStatus(usage_15min, limit_15min, usage_daily, limit_daily)


def authenticated_get(
    http_client: httpx.Client,
    path: str,
    access_token: str,
    params: dict | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET authentifié vers l'API Strava, avec attente + retry sur 429.

    `sleep` est injecté (comme `http_client`) pour pouvoir simuler
    l'attente en test sans bloquer réellement le process.
    """
    url = f"{STRAVA_API_BASE_URL}{path}"
    headers = {"Authorization": f"Bearer {access_token}"}

    attempts = 0
    network_error_attempts = 0
    while True:
        try:
            response = http_client.get(url, headers=headers, params=params)
        except httpx.TransportError:
            network_error_attempts += 1
            if network_error_attempts > MAX_NETWORK_ERROR_RETRIES:
                raise  # incident réseau persistant -> pas juste un hoquet, on abandonne
            sleep(NETWORK_ERROR_BACKOFF_S)
            continue

        if response.status_code != 429:
            response.raise_for_status()
            return response

        attempts += 1
        if attempts > MAX_RATE_LIMIT_RETRIES:
            response.raise_for_status()  # toujours 429 -> on abandonne, l'erreur remonte

        # Retry-After vient de Strava, pas de nous : rien ne garantit qu'il
        # reste raisonnable (trouvé en conditions réelles sur T-07b — un
        # sleep() non plafonné sur cette valeur a bloqué le script ~10-30 min
        # à deux reprises, sans rapport avec le quota réel une fois vérifié
        # indépendamment). Plafonné à notre propre fenêtre de repli.
        wait_s = min(
            float(response.headers.get("Retry-After", DEFAULT_RATE_LIMIT_BACKOFF_S)),
            DEFAULT_RATE_LIMIT_BACKOFF_S,
        )
        sleep(wait_s)


def get_athlete(
    http_client: httpx.Client, access_token: str, sleep: Callable[[float], None] = time.sleep
) -> dict:
    """GET /athlete — profil de l'athlète authentifié, JSON brut tel que renvoyé par Strava."""
    return authenticated_get(http_client, "/athlete", access_token, sleep=sleep).json()
