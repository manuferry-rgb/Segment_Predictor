"""T-14 : récupère la météo historique Open-Meteo pour les sorties vélo
géolocalisées déjà en base (Ride/VirtualRide — c'est le périmètre du
projet, pas la marche ou la course à pied).

Regroupe les activités par zone géographique (grille 0.1°) et plage de
dates — un appel Open-Meteo par zone, pas par activité. Reprenable :
une zone déjà téléchargée n'est jamais redemandée.

Prérequis : `main.activities` doit déjà exister (scripts/build_database.py).

Usage : uv run python scripts/fetch_weather.py
"""

from pathlib import Path

import duckdb
import httpx

from segment_predictor.ingest.open_meteo import compute_weather_zones, fetch_and_store_weather_zones

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEATHER_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "open_meteo"
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"


def main() -> None:
    conn = duckdb.connect(str(DUCKDB_PATH))
    try:
        activity_locations = conn.execute(
            "SELECT start_lat, start_lng, start_date FROM activities "
            "WHERE start_lat IS NOT NULL AND start_lng IS NOT NULL "
            "AND type IN ('Ride', 'VirtualRide')"
        ).fetchall()
    finally:
        conn.close()

    if not activity_locations:
        raise SystemExit(
            "Aucune activité géolocalisée en base — lance scripts/build_database.py d'abord."
        )

    locations = [(lat, lng, start_date.isoformat()) for lat, lng, start_date in activity_locations]
    zone_requests = compute_weather_zones(locations)
    print(
        f"{len(activity_locations)} activités géolocalisées -> {len(zone_requests)} zones à couvrir"
    )

    with httpx.Client(timeout=60.0) as client:
        summary = fetch_and_store_weather_zones(client, zone_requests, WEATHER_RAW_DIR)

    print(f"Zones récupérées : {len(summary.fetched_zones)}")
    print(f"Déjà présentes (sautées) : {len(summary.already_downloaded_zones)}")


if __name__ == "__main__":
    main()
