"""T-28 : incertitude Monte-Carlo autour du temps prédit pour la meilleure
fenêtre trouvée (T-27).

Trois sources d'incertitude propagées (voir models/uncertainty.py) : CP/W'
(écart-type réel de la régression T-09), forme (distribution empirique
récente de l'indice de performance, T-23, 90 derniers jours), vent
(écart-type relatif ASSUMÉ — pas d'historique prévision-vs-réalisé
disponible pour le calibrer, documenté comme hypothèse, pas comme mesure).

Usage : uv run python scripts/best_window_uncertainty.py <segment_id> [draft_preset]
"""

import sys
from pathlib import Path

import duckdb
import httpx

from segment_predictor.calibrate.cda_crr import calibrate_cda_crr_from_db
from segment_predictor.calibrate.draft_tagging import fit_current_cp
from segment_predictor.calibrate.form import recent_performance_index_values
from segment_predictor.models.draft import draft_ratio_for_preset
from segment_predictor.models.polyline import decode_polyline
from segment_predictor.models.segment import segment_chunks_from_polyline
from segment_predictor.models.uncertainty import propagate_uncertainty
from segment_predictor.predict.forecast_window import rank_forecast_windows_for_segment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"

# Toi + vélo — ajuste si ton poids ou ton vélo change.
MASS_KG = 91.0
# 300, pas 2000 (T-32) : chaque tirage simule TOUS les tronçons du
# polyline (segment_chunks_from_polyline), pas un seul comme avant —
# jusqu'à ~340 sur un long segment (HBFH). 2000 tirages à cette taille
# prend ~20s (mesuré), contre ~3s à 300 ; moyenne et écart-type mesurés
# quasi identiques entre les deux (< 1s d'écart sur HBFH).
N_MONTE_CARLO_SAMPLES = 300
# Hypothèse ASSUMÉE, pas mesurée : le projet n'a pas d'historique
# prévision-vs-réalisé pour calibrer une vraie erreur de prévision de
# vent (voir la question posée et tranchée avant de coder, T-28).
WIND_RELATIVE_STD = 0.20

_FRENCH_WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def _format_day_hour(dt) -> str:
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
        row = conn.execute(
            "SELECT name, average_grade, polyline FROM segments WHERE id = ?",
            [segment_id],
        ).fetchone()
        if row is None:
            print(f"segment {segment_id} introuvable dans main.segments", file=sys.stderr)
            sys.exit(1)
        segment_name, average_grade, polyline = row

        cp_fit = fit_current_cp(conn)
        cda_crr_fit = calibrate_cda_crr_from_db(conn, CSV_PATH, mass_kg=MASS_KG, cp_fit=cp_fit)
        effective_cda_m2 = cda_crr_fit.cda_m2 * draft_ratio_for_preset(draft_preset)

        with httpx.Client(timeout=30.0) as client:
            windows = rank_forecast_windows_for_segment(
                client, conn, segment_id, MASS_KG, effective_cda_m2, cda_crr_fit.crr, cp_fit
            )

        performance_index_samples = recent_performance_index_values(conn, cp_fit=cp_fit)
    finally:
        conn.close()

    if not windows:
        print("Aucun créneau exploitable sur les 10 prochains jours (6h-21h).")
        return
    if not performance_index_samples:
        print(
            "Aucun effort proche du maximum dans les 90 derniers jours (T-23) : "
            "impossible d'échantillonner l'incertitude de forme.",
            file=sys.stderr,
        )
        sys.exit(1)

    best = windows[0]
    # Cap réel par tronçon (T-32), comme rank_forecast_windows_for_segment
    # ci-dessus — sinon le vent perturbé serait projeté sur un cap moyen
    # unique, faux pour un segment qui tourne ou boucle (voir HBFH).
    chunks = segment_chunks_from_polyline(decode_polyline(polyline), average_grade)

    result = propagate_uncertainty(
        chunks,
        cp_watts=cp_fit.cp_watts,
        cp_watts_std=cp_fit.cp_watts_std,
        w_prime_joules=cp_fit.w_prime_joules,
        w_prime_joules_std=cp_fit.w_prime_joules_std,
        mass_kg=MASS_KG,
        cda_m2=effective_cda_m2,
        crr=cda_crr_fit.crr,
        performance_index_samples=performance_index_samples,
        wind_speed_ms=best.wind_speed_ms,
        wind_direction_rad=best.wind_direction_rad,
        wind_relative_std=WIND_RELATIVE_STD,
        n_samples=N_MONTE_CARLO_SAMPLES,
    )

    print(f"{segment_name} (segment {segment_id}), scénario : {draft_preset}\n")
    print(f"Meilleure fenêtre : {_format_day_hour(best.time)}")
    print(f"CP={cp_fit.cp_watts:.0f}±{cp_fit.cp_watts_std:.0f} W")
    print(f"{len(performance_index_samples)} valeurs de forme récente (90 derniers jours)")
    print(f"{result.n_excluded}/{result.n_samples} tirages écartés (vitesse insoluble)\n")
    print(f"Temps prédit : {_format_mmss(result.mean_time_s)} ± {result.std_time_s:.0f}s")


if __name__ == "__main__":
    main()
