"""T-27 : meilleure fenêtre horaire sur les 10 prochains jours.

Combine calibration CdA/Crr (T-17), CP actuel (T-16), position et cap
réels du segment (T-27), et prévision Open-Meteo créneau par créneau
(T-27, 6h-21h par défaut) — classement du plus rapide au plus lent.

Usage : uv run python scripts/best_window.py <segment_id> [draft_preset]
        draft_preset : solo (défaut) | roue_collee | un_metre | groupe
"""

import sys
from datetime import datetime
from pathlib import Path

import duckdb
import httpx

from segment_predictor.calibrate.cda_crr import calibrate_cda_crr_from_db
from segment_predictor.calibrate.draft_tagging import fit_current_cp
from segment_predictor.models.draft import draft_ratio_for_preset
from segment_predictor.predict.forecast_window import rank_forecast_windows_for_segment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"

# Toi + vélo — ajuste si ton poids ou ton vélo change.
MASS_KG = 91.0

_FRENCH_WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def _format_day_hour(dt: datetime) -> str:
    """Jour/heure en français, sans dépendre de la locale système (souvent
    absente/incomplète pour le français) — jamais utilisé en calcul,
    uniquement à l'affichage (cf CLAUDE.md, conversions à l'affichage).

    Inclut la date (jj/mm) en plus du nom du jour : sur 10 jours, le même
    jour de semaine peut apparaître deux fois (trouvé en conditions
    réelles — un vendredi au début ET à la toute fin de la fenêtre) et
    "jeudi 17h" seul serait ambigu.
    """
    day = _FRENCH_WEEKDAYS[dt.weekday()]
    hour = f"{dt.hour}h" if dt.minute == 0 else f"{dt.hour}h{dt.minute:02d}"
    return f"{day} {dt.day:02d}/{dt.month:02d} {hour}"


def _format_mmss(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}"


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print(f"Usage : uv run python {sys.argv[0]} <segment_id> [draft_preset]", file=sys.stderr)
        sys.exit(1)
    segment_id = int(sys.argv[1])
    draft_preset = sys.argv[2] if len(sys.argv) == 3 else "solo"

    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        segment_name = conn.execute(
            "SELECT name FROM segments WHERE id = ?", [segment_id]
        ).fetchone()
        if segment_name is None:
            print(f"segment {segment_id} introuvable dans main.segments", file=sys.stderr)
            sys.exit(1)

        cp_fit = fit_current_cp(conn)
        cda_crr_fit = calibrate_cda_crr_from_db(conn, CSV_PATH, mass_kg=MASS_KG, cp_fit=cp_fit)
        effective_cda_m2 = cda_crr_fit.cda_m2 * draft_ratio_for_preset(draft_preset)

        with httpx.Client(timeout=30.0) as client:
            windows = rank_forecast_windows_for_segment(
                client,
                conn,
                segment_id=segment_id,
                mass_kg=MASS_KG,
                cda_m2=effective_cda_m2,
                crr=cda_crr_fit.crr,
                cp_fit=cp_fit,
            )
    finally:
        conn.close()

    if not windows:
        print("Aucun créneau exploitable sur les 10 prochains jours (6h-21h).")
        return

    best = windows[0]
    print(f"{segment_name[0]} (segment {segment_id}), scénario : {draft_preset}\n")
    print(f"Meilleure fenêtre : {_format_day_hour(best.time)}")
    print(
        f"  temps prédit {_format_mmss(best.predicted_time_s)}, "
        f"vent {best.wind_speed_ms * 3.6:.0f} km/h, "
        f"{best.temperature_k - 273.15:.0f}°C\n"
    )

    print("Classement complet :")
    print(f"{'Créneau':<18}{'Temps':>8}{'Vent':>10}{'Temp.':>8}")
    for window in windows:
        print(
            f"{_format_day_hour(window.time):<18}"
            f"{_format_mmss(window.predicted_time_s):>8}"
            f"{window.wind_speed_ms * 3.6:>9.0f} km/h"
            f"{window.temperature_k - 273.15:>7.0f}°C"
        )


if __name__ == "__main__":
    main()
