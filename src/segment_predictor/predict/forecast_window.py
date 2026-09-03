"""Meilleure fenêtre horaire sur les 10 prochains jours (T-27).

Prévision Open-Meteo créneau par créneau : temps prédit pour chaque
heure d'une plage raisonnable (pas la nuit, par défaut 6h-21h),
classement du plus rapide au plus lent.

Contrairement à toutes les autres sources de données du projet, la
prévision n'est PAS persistée (Parquet/`raw.*`/`main.*`) — voir
`ingest/open_meteo.get_forecast_weather` pour le raisonnement. Fetch et
évaluation restent confinés à ce module et au script qui l'appelle.

Air density laissée à STANDARD_AIR_DENSITY_KG_M3 (pas recalculée depuis
la température prévue) : la prévision apporte le vent, dominant d'une
heure à l'autre ; l'effet de la température sur la densité de l'air
existe mais n'est pas branché ici — limite assumée, pas cachée,
pas dans le périmètre de ce ticket.
"""

import math
from dataclasses import dataclass
from datetime import datetime

import duckdb
import httpx

from segment_predictor.ingest.open_meteo import get_forecast_weather
from segment_predictor.models.polyline import decode_polyline
from segment_predictor.models.power import CriticalPowerFit, sustainable_power_w
from segment_predictor.models.segment import (
    SegmentChunk,
    segment_chunks_from_polyline,
    simulate_segment_time,
)

DEFAULT_FORECAST_DAYS = 10
# Plage horaire raisonnable pour une sortie vélo — évite de classer un
# créneau à 3h du matin juste parce que le vent y est nul (choix explicite,
# pas une hypothèse cachée).
DEFAULT_MIN_HOUR = 6
DEFAULT_MAX_HOUR = 21


@dataclass(frozen=True)
class ForecastWindow:
    time: datetime
    predicted_time_s: float
    # Puissance CP+W'/t requise pour TENIR predicted_time_s (T-31) — donc
    # vent de ce créneau déjà inclus, contrairement à "Puissance
    # recommandée" dans app.py (optimize_pacing, appelé sans vent) : les
    # deux ne sont pas censées coïncider, ne pas les confondre en lisant
    # l'UI.
    required_power_w: float
    wind_speed_ms: float
    wind_direction_rad: float
    temperature_k: float


def extract_hourly_slot(hourly: dict, index: int) -> tuple[datetime, float, float, float]:
    """(heure locale, température K, vitesse vent m/s, direction vent rad)
    pour l'indice `index` — mêmes conversions SI que storage/weather.py
    (km/h -> m/s, °C -> K), un point par heure, sans interpolation :
    chaque créneau EST déjà une heure précise, contrairement à l'heure
    exacte d'une activité qui tombe généralement entre deux relevés.

    Publique (T-33) : même format de réponse Open-Meteo réutilisé par
    predict/wind_scan.py, pas de raison de reparser `hourly.*` deux fois.
    """
    time = datetime.fromisoformat(hourly["time"][index])
    temperature_k = hourly["temperature_2m"][index] + 273.15
    wind_speed_ms = hourly["wind_speed_10m"][index] / 3.6
    wind_direction_rad = math.radians(hourly["wind_direction_10m"][index])
    return time, temperature_k, wind_speed_ms, wind_direction_rad


def rank_forecast_windows(
    forecast: dict,
    chunks: list[SegmentChunk],
    cp_fit: CriticalPowerFit,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    min_hour: int = DEFAULT_MIN_HOUR,
    max_hour: int = DEFAULT_MAX_HOUR,
) -> list[ForecastWindow]:
    """`forecast` : JSON brut d'Open-Meteo (`get_forecast_weather`, T-27),
    `timezone=auto` donc `hourly.time` est déjà en heure locale. Classé
    par temps prédit croissant (le meilleur créneau en premier).

    `chunks` : un cap par tronçon (T-32, `segment_chunks_from_polyline`)
    plutôt qu'un unique `SegmentChunk` — c'est ce qui permet au vent
    d'être face sur une partie du segment et de dos sur une autre, au
    lieu d'un seul vent appliqué à toute la distance.

    Un créneau dont la vitesse n'a pas de solution (`simulate_segment_
    time`, T-13 — ex. vent de face extrême) est écarté plutôt que de
    faire planter tout le classement pour un seul créneau physiquement
    dégénéré.
    """
    hourly = forecast["hourly"]

    windows = []
    for i in range(len(hourly["time"])):
        time, temperature_k, wind_speed_ms, wind_direction_rad = extract_hourly_slot(hourly, i)
        if not (min_hour <= time.hour <= max_hour):
            continue

        try:
            predicted_time_s = simulate_segment_time(
                chunks,
                cp_fit.cp_watts,
                cp_fit.w_prime_joules,
                mass_kg,
                cda_m2,
                crr,
                wind_speed_ms=wind_speed_ms,
                wind_direction_rad=wind_direction_rad,
            )
        except ValueError:
            continue  # pas de vitesse solution pour ce créneau (ex. vent de face extrême)

        windows.append(
            ForecastWindow(
                time=time,
                predicted_time_s=predicted_time_s,
                required_power_w=sustainable_power_w(
                    cp_fit.cp_watts, cp_fit.w_prime_joules, predicted_time_s
                ),
                wind_speed_ms=wind_speed_ms,
                wind_direction_rad=wind_direction_rad,
                temperature_k=temperature_k,
            )
        )

    windows.sort(key=lambda w: w.predicted_time_s)
    return windows


def rank_forecast_windows_for_segment(
    http_client: httpx.Client,
    conn: duckdb.DuckDBPyConnection,
    segment_id: int,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    cp_fit: CriticalPowerFit,
    forecast_days: int = DEFAULT_FORECAST_DAYS,
    min_hour: int = DEFAULT_MIN_HOUR,
    max_hour: int = DEFAULT_MAX_HOUR,
) -> list[ForecastWindow]:
    """Récupère la prévision pour la position du segment (`start_lat`/
    `start_lng`, T-27) et classe les créneaux horaires.

    Cap RÉEL par tronçon depuis le polyline du segment (T-32,
    `segment_chunks_from_polyline`) — remplace l'ancienne approximation
    "un seul tronçon à cap moyen start->end" (T-16/T-17/T-20), qui
    n'avait pas de sens pour un segment qui tourne ou boucle. La pente
    reste en revanche `average_grade` appliquée à chaque tronçon : aucune
    source d'altitude par tronçon n'est disponible depuis le polyline
    (lat/lng seulement) — limite assumée, pas cachée (voir ROADMAP.md
    T-32 et segment_chunks_from_polyline).

    `cda_m2`/`crr`/`cp_fit` : déjà calibrés par l'appelant (T-16/T-17),
    pas recalculés ici — ce module ne connaît pas le chemin du CSV
    d'annotations (T-16), qui reste une responsabilité du script.
    """
    row = conn.execute(
        "SELECT average_grade, polyline, start_lat, start_lng FROM segments WHERE id = ?",
        [segment_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"segment {segment_id} introuvable dans main.segments")
    average_grade, polyline, start_lat, start_lng = row

    points = decode_polyline(polyline)
    chunks = segment_chunks_from_polyline(points, average_grade)
    forecast = get_forecast_weather(http_client, start_lat, start_lng, forecast_days)

    return rank_forecast_windows(
        forecast, chunks, cp_fit, mass_kg, cda_m2, crr, min_hour=min_hour, max_hour=max_hour
    )
