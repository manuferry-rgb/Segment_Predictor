"""Construction de la table DuckDB `activity_weather` (T-14).

C'est ici, pas dans ingest, que les unités passent en SI (km/h -> m/s,
°C -> K, hPa -> Pa) et que la météo horaire de chaque zone est
interpolée à l'heure exacte du `start_date` de chaque activité.

Une activité sans position connue, ou dont la zone n'a jamais été
téléchargée, ou dont la date tombe hors de la plage couverte par sa
zone, est sautée plutôt que complétée avec une valeur devinée.
"""

import math
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from segment_predictor.ingest.open_meteo import zone_filename, zone_key

_EPOCH = datetime(1970, 1, 1)


def _seconds_since_epoch(dt: datetime) -> float:
    """Secondes depuis epoch, en traitant `dt` (naïf) comme déjà en UTC.

    Évite `datetime.timestamp()`, qui réinterpréterait un datetime naïf
    dans le fuseau LOCAL du système plutôt qu'en UTC — nos `start_date`
    (T-07) et les horodatages Open-Meteo (timezone=UTC explicite) sont
    tous les deux naïfs mais UTC par convention, donc cohérents entre eux
    tant qu'on ne passe jamais par `.timestamp()`.
    """
    return (dt - _EPOCH).total_seconds()


def _interpolate_circular_degrees(
    times_s: np.ndarray, degrees: np.ndarray, target_s: float
) -> float:
    """Interpolation linéaire d'un angle en degrés (0-360), correcte au passage 360->0.

    Interpoler l'angle directement entre 350° et 10° donnerait 180°
    (faux : le vent n'a pas fait demi-tour, il a juste traversé le nord).
    On interpole les composantes (cos, sin) du vecteur unité associé,
    pas l'angle lui-même, puis on reconvertit via atan2.
    """
    radians = np.radians(degrees)
    cos_interp = np.interp(target_s, times_s, np.cos(radians))
    sin_interp = np.interp(target_s, times_s, np.sin(radians))
    return math.atan2(sin_interp, cos_interp) % (2 * math.pi)


def _interpolate_activity_weather(
    activity_id: int, start_date: datetime, raw_weather: dict
) -> dict | None:
    hourly = raw_weather["hourly"]
    times_s = np.array([_seconds_since_epoch(datetime.fromisoformat(t)) for t in hourly["time"]])
    target_s = _seconds_since_epoch(start_date)

    if target_s < times_s[0] or target_s > times_s[-1]:
        return None  # hors de la plage couverte par cette zone

    temperature_c = np.interp(target_s, times_s, hourly["temperature_2m"])
    humidity_pct = np.interp(target_s, times_s, hourly["relative_humidity_2m"])
    pressure_hpa = np.interp(target_s, times_s, hourly["surface_pressure"])
    wind_speed_kmh = np.interp(target_s, times_s, hourly["wind_speed_10m"])
    wind_direction_rad = _interpolate_circular_degrees(
        times_s, np.array(hourly["wind_direction_10m"], dtype=float), target_s
    )

    return {
        "activity_id": activity_id,
        "temperature_k": float(temperature_c) + 273.15,
        "relative_humidity_pct": float(humidity_pct),
        "pressure_pa": float(pressure_hpa) * 100.0,
        "wind_speed_ms": float(wind_speed_kmh) / 3.6,
        "wind_direction_rad": wind_direction_rad,
    }


def build_activity_weather_table(conn: duckdb.DuckDBPyConnection, weather_raw_dir: Path) -> None:
    """Interpole la météo de chaque zone à l'heure de chaque activité et
    (re)crée la table `activity_weather`.

    Restreint à Ride/VirtualRide, comme fetch_weather.py : la marche ou la
    course à pied sont hors du périmètre du projet (prédiction de temps
    cycliste), pas la peine de les enrichir même si leur zone est déjà là.
    """
    activities = conn.execute(
        "SELECT id, start_date, start_lat, start_lng FROM activities "
        "WHERE start_lat IS NOT NULL AND start_lng IS NOT NULL "
        "AND type IN ('Ride', 'VirtualRide')"
    ).fetchall()

    zone_weather_cache: dict[tuple[float, float], dict | None] = {}
    rows = []
    for activity_id, start_date, lat, lng in activities:
        key = zone_key(lat, lng)
        if key not in zone_weather_cache:
            file_path = weather_raw_dir / zone_filename(*key)
            zone_weather_cache[key] = (
                pq.read_table(file_path).to_pylist()[0] if file_path.exists() else None
            )

        raw_weather = zone_weather_cache[key]
        if raw_weather is None:
            continue  # zone jamais téléchargée

        row = _interpolate_activity_weather(activity_id, start_date, raw_weather)
        if row is not None:
            rows.append(row)

    weather_table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("activity_id", pa.int64()),
                ("temperature_k", pa.float64()),
                ("relative_humidity_pct", pa.float64()),
                ("pressure_pa", pa.float64()),
                ("wind_speed_ms", pa.float64()),
                ("wind_direction_rad", pa.float64()),
            ]
        ),
    )
    conn.register("activity_weather_table", weather_table)
    try:
        conn.execute(
            "CREATE OR REPLACE TABLE activity_weather AS SELECT * FROM activity_weather_table"
        )
    finally:
        conn.unregister("activity_weather_table")
