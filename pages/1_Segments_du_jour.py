"""T-33 : page Streamlit indépendante — "quels segments favoris ont le
vent dans le bon sens aujourd'hui ?"

Contrairement à app.py (un segment choisi, prédiction de temps complète
avec CP/CdA/Crr calibrés), cette page ne fait AUCUNE calibration : juste
la géométrie du tracé de chaque segment (T-32, polyline décodé) contre
la météo du jour — un filtre d'opportunité rapide, pas une prédiction.
Utilisable même sans historique d'efforts pour calibrer quoi que ce
soit. Page distincte (dossier `pages/`, convention multipage Streamlit)
plutôt qu'une section de app.py : deux usages différents, un segment à
la fois vs vue d'ensemble.

Pas de tests dédiés à ce fichier — même logique que app.py (cf sa
docstring) : la logique testée vit dans predict/wind_scan.py.
"""

import math
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import streamlit as st

from segment_predictor.predict.wind_scan import scan_segments_for_today

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"

# Même convention que app.py/physics.py : direction D'OÙ VIENT le vent.
_COMPASS_LABELS = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")


def _compass_label(direction_rad: float) -> str:
    degrees = math.degrees(direction_rad) % 360
    index = round(degrees / 45) % len(_COMPASS_LABELS)
    return _COMPASS_LABELS[index]


# st.cache_resource : même raisonnement que app.py — une connexion par
# session, pas rouverte à chaque interaction.
@st.cache_resource
def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


st.set_page_config(page_title="Segments du jour", page_icon="🎯")
st.title("🎯 Segments du jour")
st.caption(
    "Parmi tous tes segments favoris, lesquels ont le vent dans le bon sens "
    "aujourd'hui ? Question géométrique pure (orientation du tracé contre le "
    "vent prévu) — indépendante de ta forme, ton CP ou ton poids. Pas une "
    "prédiction de temps, juste un filtre d'opportunité."
)

conn = get_connection()

if st.button("Scanner mes segments", type="primary"):
    with st.spinner("Récupération de la météo du jour pour chaque segment favori..."):
        with httpx.Client(timeout=30.0) as client:
            opportunities = scan_segments_for_today(conn=conn, http_client=client)
    # Persisté (T-35) : le clic sur une ligne du tableau plus bas déclenche
    # lui aussi un rerun de tout le script (comportement Streamlit), pendant
    # lequel `st.button` redevient False — sans ce stockage, le tableau
    # disparaîtrait avant même d'avoir pu lire la ligne cliquée.
    st.session_state["wind_opportunities"] = opportunities

opportunities = st.session_state.get("wind_opportunities")

if opportunities is not None:
    if not opportunities:
        st.warning(
            "Aucun créneau exploitable aujourd'hui (6h-21h, heures déjà passées "
            "exclues) sur aucun segment."
        )
        st.stop()

    # "Favorable" = vent de dos EN MOYENNE sur le segment (positif) —
    # seuil naturel, pas un pourcentage arbitraire de distance (T-34) :
    # un segment tout juste positif aide un peu, un négatif freine.
    n_good = sum(1 for o in opportunities if o.average_tailwind_speed_ms > 0)
    st.write(
        f"**{n_good}** segment(s) sur **{len(opportunities)}** avec un vent favorable "
        "en moyenne aujourd'hui."
    )
    st.caption("Clique sur une ligne pour ouvrir ce segment dans Kompass.")

    # Déjà trié par average_tailwind_speed_ms décroissante
    # (scan_segments_for_today) : les segments "à tenter" sont donc
    # naturellement en haut du tableau, sans logique de tri ici.
    table = pd.DataFrame(
        [
            {
                "Segment": o.segment_name,
                # Signe explicite (+/-) : positif = aide, négatif = freine —
                # une vraie vitesse (km/h), pas une fraction abstraite.
                "Vent favorable": f"{o.average_tailwind_speed_ms * 3.6:+.0f} km/h",
                "Meilleure heure": o.best_hour.strftime("%Hh%M"),
                "Vent": round(o.wind_speed_ms * 3.6),
                "Direction": _compass_label(o.wind_direction_rad),
                "Distance (km)": round(o.distance_m / 1000, 1),
            }
            for o in opportunities
        ]
    )
    # on_select="rerun" + selection_mode="single-row" (T-35) : Streamlit
    # renvoie l'index de ligne cliquée dans `event.selection.rows`, pas
    # besoin de composant tiers pour un tableau cliquable.
    event = st.dataframe(
        table,
        hide_index=True,
        height=600,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = event.selection.rows
    if selected_rows:
        # Consommé côté app.py (st.session_state.pop) : ce n'est pas un
        # filtre permanent, juste la valeur de départ du prochain
        # affichage du selectbox "Segment".
        st.session_state["selected_segment_id"] = opportunities[selected_rows[0]].segment_id
        st.switch_page("app.py")
