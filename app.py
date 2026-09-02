"""T-29 : interface Streamlit — choix du segment, scénario de draft,
fenêtre optimale (T-27), incertitude (T-28), stratégie de pacing (T-26).

Assemble des fonctions déjà testées ailleurs — pas de nouvelle logique
métier ici, juste des widgets autour de ce qui existe (calibrate/,
predict/, models/). Pas de tests dédiés à ce fichier, même logique que
les scripts/ (T-16 à T-28) : la logique testée vit dans les modules
qu'il appelle, ce fichier ne fait qu'assembler et afficher.

Usage : uv run streamlit run app.py
"""

import math
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import streamlit as st

from segment_predictor.calibrate.cda_crr import calibrate_cda_crr_from_db
from segment_predictor.calibrate.draft_tagging import fit_current_cp
from segment_predictor.calibrate.form import recent_performance_index_values
from segment_predictor.models.draft import draft_ratio_for_preset
from segment_predictor.models.pacing import optimize_pacing
from segment_predictor.models.segment import SegmentChunk
from segment_predictor.models.uncertainty import propagate_uncertainty
from segment_predictor.predict.forecast_window import rank_forecast_windows_for_segment
from segment_predictor.predict.scenarios import ALL_DRAFT_PRESETS

PROJECT_ROOT = Path(__file__).resolve().parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"

DEFAULT_MASS_KG = 91.0
N_MONTE_CARLO_SAMPLES = 2000
# Hypothèse ASSUMÉE, pas mesurée (T-28) : pas d'historique
# prévision-vs-réalisé disponible pour la calibrer.
WIND_RELATIVE_STD = 0.20

_PRESET_LABELS = {
    "solo": "Solo",
    "roue_collee": "Roue collée",
    "un_metre": "Un mètre",
    "groupe": "Groupe",
}
_FRENCH_WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")


def _format_day_hour(dt) -> str:
    day = _FRENCH_WEEKDAYS[dt.weekday()]
    hour = f"{dt.hour}h" if dt.minute == 0 else f"{dt.hour}h{dt.minute:02d}"
    return f"{day} {dt.day:02d}/{dt.month:02d} {hour}"


def _format_mmss(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}:{secs:02d}"


# Ordre Nord -> Nord-Est -> ... -> Nord-Ouest, pas de traduction anglaise :
# Open-Meteo donne `wind_direction_10m` en degrés météo (0° = nord, sens
# horaire), direction D'OÙ VIENT le vent (même convention que
# `wind_direction_rad` dans physics.py).
_COMPASS_LABELS = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")


def _compass_label(direction_rad: float) -> str:
    degrees = math.degrees(direction_rad) % 360
    index = round(degrees / 45) % len(_COMPASS_LABELS)
    return _COMPASS_LABELS[index]


def _format_gap_s(predicted_s: float, target_s: float) -> str:
    """Écart signé prédiction - référence, pour st.metric(delta=...).

    Positif = prédit plus lent que la référence (KOM ou PR), négatif =
    plus rapide. `delta_color="inverse"` côté appelant fait que "plus
    rapide" s'affiche en vert, "plus lent" en rouge.
    """
    return f"{predicted_s - target_s:+.0f}s"


# st.cache_resource : la connexion DuckDB n'est ouverte qu'une fois par
# session, pas rouverte à chaque interaction (case cochée, bouton
# cliqué...) — Streamlit ré-exécute tout le script à chaque interaction,
# sans ce cache on rouvrirait la connexion en boucle pour rien.
@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


# st.cache_data : même logique que cache_resource, mais pour une VALEUR
# (pas une ressource à garder ouverte) — la liste de segments ne change
# pas d'une interaction à l'autre, pas la peine de refaire la requête.
@st.cache_data
def load_segments() -> list[tuple[int, str]]:
    conn = get_connection()
    return conn.execute("SELECT id, name FROM segments ORDER BY name").fetchall()


st.set_page_config(page_title="Kompass", page_icon="🚴")
st.title("Kompass")
st.caption("Meilleure fenêtre horaire et stratégie de pacing pour un segment Strava")

conn = get_connection()
segments = load_segments()
if not segments:
    st.error("Aucun segment en base — lance scripts/build_database.py d'abord.")
    st.stop()

segment_options = {f"{name} ({segment_id})": segment_id for segment_id, name in segments}
segment_label = st.selectbox("Segment", options=list(segment_options.keys()))
segment_id = segment_options[segment_label]

draft_preset = st.selectbox(
    "Scénario de draft", options=ALL_DRAFT_PRESETS, format_func=lambda p: _PRESET_LABELS[p]
)

mass_kg = st.number_input(
    "Poids (toi + vélo, kg)", min_value=30.0, max_value=200.0, value=DEFAULT_MASS_KG, step=0.5
)

if st.button("Chercher la meilleure fenêtre", type="primary"):
    with st.spinner("Calibration et prévision météo..."):
        cp_fit = fit_current_cp(conn)
        cda_crr_fit = calibrate_cda_crr_from_db(conn, CSV_PATH, mass_kg=mass_kg, cp_fit=cp_fit)
        effective_cda_m2 = cda_crr_fit.cda_m2 * draft_ratio_for_preset(draft_preset)

        with httpx.Client(timeout=30.0) as client:
            windows = rank_forecast_windows_for_segment(
                client, conn, segment_id, mass_kg, effective_cda_m2, cda_crr_fit.crr, cp_fit
            )

    st.caption(
        f"CP={cp_fit.cp_watts:.0f}±{cp_fit.cp_watts_std:.0f} W · "
        f"CdA={cda_crr_fit.cda_m2:.3f} m² · Crr={cda_crr_fit.crr:.4f}"
    )

    if not windows:
        st.warning("Aucun créneau exploitable sur les 10 prochains jours (6h-21h).")
        st.stop()

    distance_m, average_grade, heading_rad, kom_seconds, pr_seconds = conn.execute(
        "SELECT distance_m, average_grade, heading_rad, kom_seconds, pr_seconds "
        "FROM segments WHERE id = ?",
        [segment_id],
    ).fetchone()
    chunk = SegmentChunk(0.0, distance_m, average_grade, heading_rad)

    best = windows[0]
    st.subheader(f"Meilleure fenêtre : {_format_day_hour(best.time)}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Temps prédit", _format_mmss(best.predicted_time_s))
    col2.metric(
        "Vent",
        f"{best.wind_speed_ms * 3.6:.0f} km/h",
        delta=f"du {_compass_label(best.wind_direction_rad)}",
        delta_color="off",
    )
    col3.metric("Température", f"{best.temperature_k - 273.15:.0f}°C")

    # Écart à combler (T-31) : référence = temps prédit sur la meilleure
    # fenêtre, comparé au KOM du segment et à mon propre PR.
    kom_col, pr_col = st.columns(2)
    kom_col.metric(
        "KOM du segment",
        _format_mmss(kom_seconds),
        delta=_format_gap_s(best.predicted_time_s, kom_seconds),
        delta_color="inverse",
    )
    if pr_seconds is not None:
        pr_col.metric(
            "Mon PR",
            _format_mmss(pr_seconds),
            delta=_format_gap_s(best.predicted_time_s, pr_seconds),
            delta_color="inverse",
        )
        # Puissance moyenne sur L'EFFORT DU PR (T-31), pour estimer soi-même
        # la puissance à tenir pour viser le KOM. Retrouvé par
        # (segment_id, elapsed_time_s) : `segments.pr_seconds` ne porte pas
        # l'id de l'effort correspondant, donc pas de jointure directe.
        # `average_watts` peut être NULL (pas de capteur ce jour-là, T-07b) —
        # pas de valeur inventée, on l'indique explicitement.
        pr_power_row = conn.execute(
            "SELECT average_watts, device_watts FROM segment_efforts "
            "WHERE segment_id = ? AND elapsed_time_s = ? "
            "ORDER BY start_date DESC LIMIT 1",
            [segment_id, pr_seconds],
        ).fetchone()
        if pr_power_row is not None and pr_power_row[0] is not None:
            pr_average_watts, device_watts = pr_power_row
            sensor_note = "" if device_watts else " (non confirmé par un capteur)"
            pr_col.caption(f"Puissance moyenne : {pr_average_watts:.0f} W{sensor_note}")
        else:
            pr_col.caption("Puissance moyenne : non disponible")
    else:
        pr_col.metric("Mon PR", "Jamais roulé")

    # Incertitude (T-28)
    performance_index_samples = recent_performance_index_values(conn, cp_fit=cp_fit)
    if performance_index_samples:
        result = propagate_uncertainty(
            [chunk],
            cp_watts=cp_fit.cp_watts,
            cp_watts_std=cp_fit.cp_watts_std,
            w_prime_joules=cp_fit.w_prime_joules,
            w_prime_joules_std=cp_fit.w_prime_joules_std,
            mass_kg=mass_kg,
            cda_m2=effective_cda_m2,
            crr=cda_crr_fit.crr,
            performance_index_samples=performance_index_samples,
            wind_speed_ms=best.wind_speed_ms,
            wind_direction_rad=best.wind_direction_rad,
            wind_relative_std=WIND_RELATIVE_STD,
            n_samples=N_MONTE_CARLO_SAMPLES,
        )
        st.write(
            f"Avec incertitude (CP, forme sur {len(performance_index_samples)} jours "
            f"récents, vent) : **{_format_mmss(result.mean_time_s)} ± {result.std_time_s:.0f}s**"
        )
    else:
        st.info(
            "Pas assez d'efforts proches du maximum dans les 90 derniers jours (T-23) "
            "pour estimer l'incertitude de forme."
        )

    # Classement complet
    st.subheader("Classement des créneaux")
    table = pd.DataFrame(
        [
            {
                "Créneau": _format_day_hour(w.time),
                "Temps": _format_mmss(w.predicted_time_s),
                "Vent (km/h)": round(w.wind_speed_ms * 3.6),
                "Direction": _compass_label(w.wind_direction_rad),
                "Température (°C)": round(w.temperature_k - 273.15),
            }
            for w in windows
        ]
    )
    st.dataframe(table, hide_index=True, height=300)

    # Stratégie de pacing (T-26)
    st.subheader("Stratégie de pacing")
    st.caption(
        "⚠️ Un seul tronçon par segment — aucun profil pente/distance détaillé n'est "
        "stocké au niveau segment (T-07b jamais fait). La puissance recommandée est "
        "donc constante sur tout le segment : pas une vraie stratégie variable, juste "
        "la puissance soutenable optimale pour ce profil simplifié."
    )
    pacing_result = optimize_pacing(
        [chunk], cp_fit.cp_watts, cp_fit.w_prime_joules, mass_kg, effective_cda_m2, cda_crr_fit.crr
    )
    st.metric("Puissance recommandée", f"{pacing_result.power_profile_w[0]:.0f} W")
