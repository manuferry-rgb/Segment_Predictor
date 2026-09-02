"""T-20 : tableau comparatif des temps prédits sous plusieurs scénarios de draft.

Réutilise la calibration CdA/Crr officielle (T-17, sur les efforts
tagués solo) et le CP actuel (T-16), pour un segment donné.

Usage : uv run python scripts/compare_draft_scenarios.py <segment_id>
"""

import sys
from pathlib import Path

import duckdb

from segment_predictor.calibrate.cda_crr import calibrate_cda_crr_from_db
from segment_predictor.calibrate.draft_tagging import fit_current_cp
from segment_predictor.predict.scenarios import compare_draft_scenarios_for_segment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"

# Toi + vélo — ajuste si ton poids ou ton vélo change.
MASS_KG = 91.0

_PRESET_LABELS = {
    "solo": "Solo",
    "roue_collee": "Roue collée",
    "un_metre": "Un mètre",
    "groupe": "Groupe",
}


def _format_mmss(seconds: float) -> str:
    """Conversion en mm:ss uniquement à l'affichage (unités SI en interne,
    cf CLAUDE.md) — jamais stockée ni utilisée en calcul sous cette forme."""
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}"


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage : uv run python {sys.argv[0]} <segment_id>", file=sys.stderr)
        sys.exit(1)
    segment_id = int(sys.argv[1])

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
        results = compare_draft_scenarios_for_segment(
            conn,
            segment_id=segment_id,
            mass_kg=MASS_KG,
            cda_m2=cda_crr_fit.cda_m2,
            crr=cda_crr_fit.crr,
            cp_fit=cp_fit,
        )
    finally:
        conn.close()

    print(f"{segment_name[0]} (segment {segment_id})")
    print(
        f"CdA={cda_crr_fit.cda_m2:.4f} m², Crr={cda_crr_fit.crr:.5f}, CP={cp_fit.cp_watts:.0f} W\n"
    )
    print(f"{'Scénario':<14}{'Temps':>8}{'Gain (s)':>12}{'Gain (%)':>12}")
    for result in results:
        print(
            f"{_PRESET_LABELS[result.preset]:<14}"
            f"{_format_mmss(result.predicted_time_s):>8}"
            f"{result.gain_vs_solo_s:>12.1f}"
            f"{result.gain_vs_solo_fraction * 100:>11.1f}%"
        )


if __name__ == "__main__":
    main()
