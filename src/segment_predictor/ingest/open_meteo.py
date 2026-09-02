"""Météo historique Open-Meteo (couche ingest, aucune transformation).

Gratuit, sans clé. Un fichier Parquet par zone géographique (pas par
activité) : `compute_weather_zones` regroupe les activités par zone
arrondie et calcule la plage de dates à couvrir, pour un seul appel par
zone plutôt qu'un par activité.

Contrairement à Strava, Open-Meteo ne renvoie aucun en-tête de quota
(vérifié en conditions réelles) — pas de X-RateLimit-* à lire, donc pas
d'équivalent de `strava_client.parse_rate_limit` ici, juste un retry
basique sur 429.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = (
    "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
)

DEFAULT_RETRY_BACKOFF_S = 60.0
MAX_RETRIES = 5

DEFAULT_WEATHER_GRID_DEGREES = 0.1


def zone_key(
    lat: float, lng: float, grid_degrees: float = DEFAULT_WEATHER_GRID_DEGREES
) -> tuple[float, float]:
    """Arrondit (lat, lng) à la grille de zone.

    Utilisé à la fois pour regrouper les requêtes (ici) et pour retrouver
    le bon fichier de zone en storage (build_activity_weather_table) —
    une seule formule d'arrondi, partagée, pour ne jamais désynchroniser
    les deux usages.
    """
    return (round(lat / grid_degrees) * grid_degrees, round(lng / grid_degrees) * grid_degrees)


def zone_filename(zone_lat: float, zone_lng: float) -> str:
    return f"{zone_lat:.1f}_{zone_lng:.1f}.parquet"


@dataclass(frozen=True)
class WeatherZoneRequest:
    zone_lat: float
    zone_lng: float
    start_date: str  # "YYYY-MM-DD"
    end_date: str


def compute_weather_zones(
    activity_locations: list[tuple[float, float, str]],
    grid_degrees: float = DEFAULT_WEATHER_GRID_DEGREES,
) -> list[WeatherZoneRequest]:
    """(lat, lng, start_date_iso) par activité -> une requête par zone géographique.

    Fonction pure : ne fait aucun appel réseau, se contente de regrouper
    des données déjà en mémoire (typiquement lues depuis main.activities).
    """
    if grid_degrees <= 0:
        raise ValueError(f"grid_degrees doit être positif, reçu {grid_degrees}")

    zones: dict[tuple[float, float], list[str]] = {}
    for lat, lng, date_iso in activity_locations:
        zones.setdefault(zone_key(lat, lng, grid_degrees), []).append(date_iso)

    return [
        WeatherZoneRequest(
            zone_lat=lat,
            zone_lng=lng,
            start_date=min(dates)[:10],
            end_date=max(dates)[:10],
        )
        for (lat, lng), dates in zones.items()
    ]


def get_historical_weather(
    http_client: httpx.Client,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """GET /v1/archive, JSON brut tel que renvoyé par Open-Meteo.

    Pas de transformation d'unité ici : l'API renvoie wind_speed_10m en
    km/h, temperature_2m en °C, surface_pressure en hPa — stockés tels
    quels. La conversion en SI se fait dans storage/weather.py.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARIABLES,
        "timezone": "UTC",
    }

    attempts = 0
    while True:
        response = http_client.get(OPEN_METEO_ARCHIVE_URL, params=params)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()

        attempts += 1
        if attempts > MAX_RETRIES:
            response.raise_for_status()  # toujours 429 -> on abandonne, l'erreur remonte

        wait_s = float(response.headers.get("Retry-After", DEFAULT_RETRY_BACKOFF_S))
        sleep(wait_s)


def save_weather_zone(raw_dir: Path, zone_lat: float, zone_lng: float, weather: dict) -> Path:
    """Écrit la météo d'UNE zone dans son propre Parquet, forme brute inchangée."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / zone_filename(zone_lat, zone_lng)
    pq.write_table(pa.Table.from_pylist([weather]), file_path)
    return file_path


def _already_downloaded_zones(raw_dir: Path) -> set[tuple[float, float]]:
    if not raw_dir.exists():
        return set()
    zones = set()
    for path in raw_dir.glob("*.parquet"):
        lat_str, lng_str = path.stem.split("_")
        zones.add((float(lat_str), float(lng_str)))
    return zones


@dataclass(frozen=True)
class FetchWeatherSummary:
    fetched_zones: list[tuple[float, float]]
    already_downloaded_zones: list[tuple[float, float]]


def fetch_and_store_weather_zones(
    http_client: httpx.Client,
    zone_requests: list[WeatherZoneRequest],
    raw_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchWeatherSummary:
    """Récupère la météo des zones pas encore sur disque.

    Une zone déjà téléchargée n'est jamais redemandée — y compris si de
    nouvelles activités dans cette zone tombent hors de la plage de
    dates déjà couverte. Il faut supprimer le fichier localement pour
    forcer un refetch (même limite que pour les segments en T-06).
    """
    already_downloaded = _already_downloaded_zones(raw_dir)
    fetched: list[tuple[float, float]] = []

    for zone_request in zone_requests:
        key = (zone_request.zone_lat, zone_request.zone_lng)
        if key in already_downloaded:
            continue

        weather = get_historical_weather(
            http_client,
            zone_request.zone_lat,
            zone_request.zone_lng,
            zone_request.start_date,
            zone_request.end_date,
            sleep=sleep,
        )
        save_weather_zone(raw_dir, zone_request.zone_lat, zone_request.zone_lng, weather)
        fetched.append(key)

    return FetchWeatherSummary(
        fetched_zones=fetched,
        already_downloaded_zones=sorted(already_downloaded),
    )
