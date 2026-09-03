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
from segment_predictor.calibrate.draft_tagging import fit_current_cp, load_existing_annotations
from segment_predictor.calibrate.form import recent_performance_index_values
from segment_predictor.models.draft import draft_ratio_for_preset
from segment_predictor.models.pacing import optimize_pacing
from segment_predictor.models.polyline import decode_polyline
from segment_predictor.models.power import sustainable_power_w
from segment_predictor.models.segment import SegmentChunk, segment_chunks_from_polyline
from segment_predictor.models.uncertainty import propagate_uncertainty
from segment_predictor.predict.forecast_window import rank_forecast_windows_for_segment
from segment_predictor.predict.scenarios import ALL_DRAFT_PRESETS

PROJECT_ROOT = Path(__file__).resolve().parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"

DEFAULT_MASS_KG = 91.0
# 300, pas 2000 (T-32) : chaque tirage simule TOUS les tronçons du
# polyline (segment_chunks_from_polyline), pas un seul comme avant —
# jusqu'à ~340 sur un long segment (HBFH). 2000 tirages à cette taille
# prend ~20s (mesuré), contre ~3s à 300 ; moyenne et écart-type mesurés
# quasi identiques entre les deux (< 1s d'écart sur HBFH) — le vent
# perturbé reste un tirage Normal simple, pas besoin de 2000 points pour
# le résumer par une moyenne et un écart-type stables.
N_MONTE_CARLO_SAMPLES = 300
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


def _render_metric_card(
    container,
    label: str,
    value: str,
    delta_s: float | None = None,
    extra_lines: list[str] | None = None,
) -> None:
    """Une "case" façon st.metric, mais où des lignes supplémentaires
    (date, lien Strava, puissance...) vivent DANS la même boîte colorée,
    pas en dessous dans une case visuellement séparée — ce que st.metric
    ne permet pas (rien n'est injectable après son delta). Demandé
    explicitement (T-31) pour les cases KOM/PR.

    `delta_s` : positif = prédiction plus LENTE que la référence (KOM ou
    PR), négatif = plus rapide — reproduit à la main la coloration
    "inverse" de st.metric (négatif = vert = bonne nouvelle, positif =
    rouge), qu'on ne peut plus utiliser une fois sorti de st.metric.
    `extra_lines` : déjà du HTML sûr à injecter tel quel (ex. un lien
    `<a href=...>`) — c'est l'appelant qui les construit, cette fonction
    ne fait qu'assembler la case.
    """
    delta_html = ""
    if delta_s is not None:
        color_class = "kompass-delta-green" if delta_s < 0 else "kompass-delta-red"
        arrow = "↓" if delta_s < 0 else "↑"
        delta_html = f'<div class="kompass-card-delta {color_class}">{arrow} {delta_s:+.0f}s</div>'

    extra_html = "".join(
        f'<div class="kompass-card-extra">{line}</div>' for line in (extra_lines or [])
    )

    container.markdown(
        f"""
        <div class="kompass-card">
            <div class="kompass-card-label">{label}</div>
            <div class="kompass-card-value">{value}</div>
            {delta_html}
            {extra_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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

# CSS injecté (T-31) : Streamlit n'expose pas via config.toml le style des
# cartes st.metric individuellement, seulement la palette globale — on
# cible leur conteneur (data-testid, stable dans l'API publique Streamlit,
# contrairement aux classes générées) pour un liseré couleur effort et une
# valeur plus imposante, cohérent avec le thème "sportif" de
# .streamlit/config.toml.
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #FFF1EC;
        border-left: 4px solid #FF4B2B;
        border-radius: 8px;
        padding: 0.9rem 1rem 0.7rem 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
        font-weight: 700;
    }
    h1 {
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    /* Cases KOM/PR (T-31 suite) : même habillage que div[data-testid="stMetric"]
       ci-dessus, mais en HTML direct pour pouvoir empiler des lignes de
       texte (date, lien Strava, puissance...) DANS la même boîte. */
    .kompass-card {
        background-color: #FFF1EC;
        border-left: 4px solid #FF4B2B;
        border-radius: 8px;
        padding: 0.9rem 1rem 0.9rem 1rem;
        margin-bottom: 1rem;
    }
    .kompass-card-label {
        font-size: 0.875rem;
        color: rgba(49, 51, 63, 0.6);
    }
    .kompass-card-value {
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kompass-card-delta {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.05rem 0.45rem;
        border-radius: 6px;
        margin-top: 0.3rem;
    }
    .kompass-delta-green {
        color: #1a7f37;
        background-color: rgba(26, 127, 55, 0.12);
    }
    .kompass-delta-red {
        color: #cf222e;
        background-color: rgba(207, 34, 46, 0.12);
    }
    .kompass-card-extra {
        font-size: 0.8rem;
        color: rgba(49, 51, 63, 0.75);
        margin-top: 0.4rem;
    }
    .kompass-card-extra a {
        color: #FF4B2B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚴 Kompass")
st.caption("Meilleure fenêtre horaire et stratégie de pacing pour un segment Strava")

conn = get_connection()
segments = load_segments()
if not segments:
    st.error("Aucun segment en base — lance scripts/build_database.py d'abord.")
    st.stop()

segment_options = {f"{name} ({segment_id})": segment_id for segment_id, name in segments}

# Pré-sélection venant de "Segments du jour" (clic sur une ligne du
# tableau, T-35) : posée dans st.session_state AVANT la création du
# widget, pas via `index=` sur st.selectbox — un `index` n'est pris en
# compte qu'à la création du widget, un changement manuel de segment
# ensuite serait écrasé à chaque re-exécution du script si on le
# recalculait à chaque fois. `.pop` : valeur consommée une fois, pas un
# filtre permanent qui rejouerait la même sélection à chaque interaction
# sur cette page (ex. changement de scénario de draft plus bas).
preselected_segment_id = st.session_state.pop("selected_segment_id", None)
if preselected_segment_id is not None:
    for label, sid in segment_options.items():
        if sid == preselected_segment_id:
            st.session_state["segment_select"] = label
            break

segment_label = st.selectbox("Segment", options=list(segment_options.keys()), key="segment_select")
segment_id = segment_options[segment_label]

# Longueur et dénivelé du segment choisi (T-31) : affiché dès la
# sélection, indépendamment du bouton de recherche plus bas — pour juger
# le profil avant même de lancer un calcul. Pas de D- : Strava ne le
# fournit pas au niveau segment (seulement D+, `total_elevation_gain`) ;
# `elevation_high`/`elevation_low` ne suffiraient pas à le reconstruire
# sans supposer un profil qui monte et descend une seule fois, une
# donnée inventée pour une boucle ou un profil accidenté.
segment_distance_m, segment_elevation_gain_m = conn.execute(
    "SELECT distance_m, total_elevation_gain_m FROM segments WHERE id = ?",
    [segment_id],
).fetchone()
st.caption(
    f"{segment_distance_m / 1000:.1f} km · D+ {segment_elevation_gain_m:.0f} m · "
    "D- non disponible (pas fourni par Strava au niveau segment)"
)

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

    distance_m, average_grade, heading_rad, polyline, kom_seconds, pr_seconds = conn.execute(
        "SELECT distance_m, average_grade, heading_rad, polyline, kom_seconds, pr_seconds "
        "FROM segments WHERE id = ?",
        [segment_id],
    ).fetchone()
    # Un seul tronçon (pacing, T-26) : le vent n'y entre pas du tout
    # (optimize_pacing simule à vent nul), donc un cap unique ne change
    # rien à son résultat — pas la peine d'y payer le coût des ~340
    # tronçons du polyline pour zéro différence.
    chunk = SegmentChunk(0.0, distance_m, average_grade, heading_rad)
    # Tronçons multiples, cap réel chacun (incertitude, T-28/T-32) : là le
    # vent compte, un cap moyen unique le fausse (voir HBFH, T-32).
    chunks = segment_chunks_from_polyline(decode_polyline(polyline), average_grade)

    best = windows[0]
    st.subheader(f"Meilleure fenêtre : {_format_day_hour(best.time)}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Temps prédit", _format_mmss(best.predicted_time_s))
    # Puissance requise (T-31) : CP + W'/t pour CE créneau, vent inclus —
    # à comparer à ta puissance de PR réelle affichée plus bas. Le
    # libellé précise "ce créneau, vent inclus" pour ne pas la confondre
    # avec "Puissance de pacing (sans vent)" juste en dessous : deux
    # réponses à deux questions différentes, pas deux estimations du
    # même nombre (voir son propre avertissement).
    col1.caption(f"Puissance requise pour ce créneau (vent inclus) : {best.required_power_w:.0f} W")
    col2.metric(
        "Vent",
        f"{best.wind_speed_ms * 3.6:.0f} km/h",
        delta=f"du {_compass_label(best.wind_direction_rad)}",
        delta_color="off",
    )
    col3.metric("Température", f"{best.temperature_k - 273.15:.0f}°C")

    # Stratégie de pacing (T-26) — placée ici, avant le classement détaillé
    # des créneaux, pour rester juste sous la meilleure fenêtre plutôt
    # qu'en bas de page.
    st.subheader("Stratégie de pacing")
    st.caption(
        "Question différente de la puissance requise ci-dessus : pas 'combien de watts "
        "pour ce créneau précis', mais 'quelle puissance constante optimale tenir sur ce "
        "profil, vent mis à part'. ⚠️ Un seul tronçon par segment — aucun profil "
        "pente/distance détaillé n'est stocké au niveau segment (T-07b jamais fait), donc "
        "'constante' n'est pas une vraie stratégie variable, juste la puissance soutenable "
        "optimale pour ce profil simplifié."
    )
    pacing_result = optimize_pacing(
        [chunk], cp_fit.cp_watts, cp_fit.w_prime_joules, mass_kg, effective_cda_m2, cda_crr_fit.crr
    )
    st.metric("Puissance de pacing (sans vent)", f"{pacing_result.power_profile_w[0]:.0f} W")

    # Écart à combler (T-31) : référence = temps prédit sur la meilleure
    # fenêtre, comparé au KOM du segment et à mon propre PR. Rendu via
    # _render_metric_card (T-31 suite) plutôt que st.metric + st.caption :
    # les infos annexes (puissance, date, lien Strava, statut draft)
    # doivent apparaître DANS la même case que le chiffre.
    kom_col, pr_col = st.columns(2)

    # Puissance estimée pour le KOM (T-31) : CP + W'/t appliqué à
    # kom_seconds — TA puissance nécessaire si tu tenais ce temps, pas
    # celle réellement développée par le recordman (Strava ne la fournit
    # pas, aucune donnée à afficher pour un effort qui n'est pas le tien).
    kom_power_w = sustainable_power_w(cp_fit.cp_watts, cp_fit.w_prime_joules, kom_seconds)
    kom_extra_lines = [
        f"Puissance estimée pour toi : {kom_power_w:.0f} W",
        "Basée sur TON modèle CP à cette durée, pas la puissance réelle du recordman.",
    ]
    # CP+W'/t extrapolé hors de la plage de durées sur laquelle CP/W' ont
    # été ajustés (fit_current_cp) est moins fiable — signalé plutôt que
    # présenté comme une puissance aussi sûre que dans la plage calibrée.
    duration_min_s, duration_max_s = cp_fit.duration_range_s
    if not duration_min_s <= kom_seconds <= duration_max_s:
        kom_extra_lines.append(
            f"⚠️ {kom_seconds}s hors de la plage calibrée du modèle CP "
            f"({duration_min_s}-{duration_max_s}s) : estimation moins fiable."
        )
    _render_metric_card(
        kom_col,
        "KOM du segment",
        _format_mmss(kom_seconds),
        delta_s=best.predicted_time_s - kom_seconds,
        extra_lines=kom_extra_lines,
    )

    if pr_seconds is not None:
        # Détails de L'EFFORT DU PR (T-31), pour estimer soi-même la
        # puissance à tenir pour viser le KOM et retrouver la sortie
        # d'origine. Retrouvé par (segment_id, elapsed_time_s) :
        # `segments.pr_seconds` ne porte pas l'id de l'effort correspondant,
        # donc pas de jointure directe.
        pr_effort_row = conn.execute(
            "SELECT id, average_watts, device_watts, start_date, activity_id "
            "FROM segment_efforts WHERE segment_id = ? AND elapsed_time_s = ? "
            "ORDER BY start_date DESC LIMIT 1",
            [segment_id, pr_seconds],
        ).fetchone()
        pr_extra_lines = []
        if pr_effort_row is not None:
            pr_effort_id, pr_average_watts, device_watts, pr_start_date, pr_activity_id = (
                pr_effort_row
            )
            strava_url = f"https://www.strava.com/activities/{pr_activity_id}"
            pr_extra_lines.append(
                f"Réalisé le {pr_start_date:%d/%m/%Y} · "
                f'<a href="{strava_url}" target="_blank">Voir sur Strava</a>'
            )
            # `average_watts` peut être NULL (pas de capteur ce jour-là,
            # T-07b) — pas de valeur inventée, on l'indique explicitement.
            if pr_average_watts is not None:
                sensor_note = "" if device_watts else " (non confirmé par un capteur)"
                pr_extra_lines.append(f"Puissance moyenne : {pr_average_watts:.0f} W{sensor_note}")
            else:
                pr_extra_lines.append("Puissance moyenne : non disponible")
            # Statut de draft de CET effort (T-31) : la comparaison plus
            # haut (temps prédit vs PR) suppose implicitement un PR solo,
            # comme la prédiction elle-même — mais rien ne le garantit tant
            # que l'effort n'a pas été trié dans le CSV (T-16). Un PR
            # obtenu dans une roue serait plus rapide qu'un effort solo à
            # puissance égale : le signaler plutôt que le supposer.
            draft_status = load_existing_annotations(CSV_PATH).get(pr_effort_id, "unknown")
            if draft_status != "solo":
                pr_extra_lines.append(
                    f"⚠️ Statut draft de cet effort : {draft_status} — pas confirmé solo, "
                    "la comparaison peut être biaisée."
                )
        else:
            pr_extra_lines.append("Puissance moyenne : non disponible")

        _render_metric_card(
            pr_col,
            "Mon PR",
            _format_mmss(pr_seconds),
            delta_s=best.predicted_time_s - pr_seconds,
            extra_lines=pr_extra_lines,
        )
    else:
        _render_metric_card(pr_col, "Mon PR", "Jamais roulé")

    # Incertitude (T-28)
    performance_index_samples = recent_performance_index_values(conn, cp_fit=cp_fit)
    if performance_index_samples:
        # Peut prendre quelques secondes sur un long segment (T-32 : un
        # tirage = une simulation de TOUS ses tronçons, pas un seul) —
        # un spinner plutôt qu'un gel silencieux de l'UI.
        with st.spinner("Propagation de l'incertitude..."):
            result = propagate_uncertainty(
                chunks,
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
                "Puissance (W)": round(w.required_power_w),
                "Vent (km/h)": round(w.wind_speed_ms * 3.6),
                "Direction": _compass_label(w.wind_direction_rad),
                "Température (°C)": round(w.temperature_k - 273.15),
            }
            for w in windows
        ]
    )
    st.dataframe(table, hide_index=True, height=300)
