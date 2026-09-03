"""Balayage de tous les segments favoris contre le vent du jour (T-33/T-34).

Question différente de forecast_window.py (T-27) : pas "quelle est la
meilleure fenêtre POUR CE segment sur 10 jours", mais "PARMI TOUS mes
segments, lesquels ont le vent dans le bon sens AUJOURD'HUI" — un filtre
d'opportunité géométrique (`average_tailwind_speed_ms`, T-32/T-34), pas
une prédiction de temps. Ne dépend d'AUCUNE calibration (CP, CdA, Crr) :
seulement le tracé du segment (T-32) contre la prévision du jour, donc
utilisable même sans forme du jour connue.

Comme forecast_window.py, la prévision n'est pas persistée — fetch et
usage restent confinés à ce module et à l'appelant (T-27).
"""

from dataclasses import dataclass
from datetime import date, datetime

import duckdb
import httpx

from segment_predictor.ingest.open_meteo import get_forecast_weather
from segment_predictor.models.polyline import decode_polyline
from segment_predictor.models.segment import (
    SegmentChunk,
    average_tailwind_speed_ms,
    segment_chunks_from_polyline,
)
from segment_predictor.predict.forecast_window import extract_hourly_slot

# Même plage par défaut que forecast_window.py (T-27) : une sortie vélo
# raisonnable, pas 3h du matin juste parce qu'un chiffre y est meilleur.
DEFAULT_MIN_HOUR = 6
DEFAULT_MAX_HOUR = 21


@dataclass(frozen=True)
class SegmentWindOpportunity:
    segment_id: int
    segment_name: str
    distance_m: float
    best_hour: datetime
    # m/s, positif = vent de dos en moyenne sur le segment, négatif =
    # vent de face en moyenne (voir average_tailwind_speed_ms, T-34).
    average_tailwind_speed_ms: float
    wind_speed_ms: float
    wind_direction_rad: float


def best_wind_opportunity_today(
    forecast: dict,
    chunks: list[SegmentChunk],
    segment_id: int,
    segment_name: str,
    distance_m: float,
    today: date,
    now: datetime,
    min_hour: int = DEFAULT_MIN_HOUR,
    max_hour: int = DEFAULT_MAX_HOUR,
) -> SegmentWindOpportunity | None:
    """Meilleure heure d'AUJOURD'HUI (pas les 10 jours de
    rank_forecast_windows, T-27) pour CE segment, au sens du vent le
    plus favorable EN MOYENNE (`average_tailwind_speed_ms`) — `None` si
    aucune heure restante aujourd'hui ne tombe dans `[min_hour, max_hour]`.

    `today`/`now` injectés (pas `datetime.now()` en dur) : mêmes heures
    en entrée -> même résultat en sortie, testable sans dépendre de
    l'horloge système. `now` sert aussi à écarter les heures déjà
    passées : un créneau plus favorable mais révolu n'est pas
    actionnable.
    """
    hourly = forecast["hourly"]
    best: SegmentWindOpportunity | None = None

    for i in range(len(hourly["time"])):
        time, _temperature_k, wind_speed_ms, wind_direction_rad = extract_hourly_slot(hourly, i)
        if time.date() != today or time < now or not (min_hour <= time.hour <= max_hour):
            continue

        tailwind_ms = average_tailwind_speed_ms(chunks, wind_speed_ms, wind_direction_rad)
        if best is None or tailwind_ms > best.average_tailwind_speed_ms:
            best = SegmentWindOpportunity(
                segment_id=segment_id,
                segment_name=segment_name,
                distance_m=distance_m,
                best_hour=time,
                average_tailwind_speed_ms=tailwind_ms,
                wind_speed_ms=wind_speed_ms,
                wind_direction_rad=wind_direction_rad,
            )

    return best


def scan_segments_for_today(
    http_client: httpx.Client,
    conn: duckdb.DuckDBPyConnection,
    min_hour: int = DEFAULT_MIN_HOUR,
    max_hour: int = DEFAULT_MAX_HOUR,
    now: datetime | None = None,
) -> list[SegmentWindOpportunity]:
    """Un appel météo (`forecast_days=1`) par segment favori, classé par
    `average_tailwind_speed_ms` décroissante (le plus favorable en
    premier — voir sa docstring pour pourquoi une vitesse réelle plutôt
    qu'une fraction de distance).

    Un segment dont le polyline dégénère en 0 tronçon exploitable
    (`segment_chunks_from_polyline`, ex. tous les points confondus) est
    écarté plutôt que de faire planter tout le balayage pour un seul
    segment mal formé — même principe que `rank_forecast_windows` (T-27)
    pour un créneau physiquement dégénéré.

    `now` injectable pour les tests ; laissé à `None` en usage normal
    pour l'heure réelle à chaque appel.
    """
    if now is None:
        now = datetime.now()
    today = now.date()

    rows = conn.execute(
        "SELECT id, name, distance_m, average_grade, polyline, start_lat, start_lng FROM segments"
    ).fetchall()

    opportunities = []
    for segment_id, name, distance_m, average_grade, polyline, start_lat, start_lng in rows:
        try:
            points = decode_polyline(polyline)
            chunks = segment_chunks_from_polyline(points, average_grade)
        except ValueError:
            continue  # polyline dégénéré : segment écarté, pas de crash du balayage entier

        forecast = get_forecast_weather(http_client, start_lat, start_lng, forecast_days=1)
        opportunity = best_wind_opportunity_today(
            forecast, chunks, segment_id, name, distance_m, today, now, min_hour, max_hour
        )
        if opportunity is not None:
            opportunities.append(opportunity)

    opportunities.sort(key=lambda o: o.average_tailwind_speed_ms, reverse=True)
    return opportunities
