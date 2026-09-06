"""T-36 : squelette FastAPI — remplace progressivement l'app Streamlit
(T-29). Même rôle que app.py avant lui : assemble des fonctions déjà
testées ailleurs (storage/, predict/, calibrate/, models/), aucune
logique métier nouvelle ici — juste du câblage HTTP et de la
sérialisation JSON. Pas de tests dédiés à ce fichier pour la même
raison que app.py (cf sa docstring, T-29).

Usage : uv run uvicorn segment_predictor.api.main:app --reload
"""

from datetime import datetime
from pathlib import Path

import duckdb
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from segment_predictor.calibrate.cda_crr import calibrate_cda_crr_from_db
from segment_predictor.calibrate.draft_tagging import fit_current_cp
from segment_predictor.models.draft import draft_ratio_for_preset
from segment_predictor.predict.forecast_window import rank_forecast_windows_for_segment

# 4 parents : main.py -> api/ -> segment_predictor/ -> src/ -> racine du
# projet (app.py, lui, est à la racine et n'a besoin que d'un seul .parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"

app = FastAPI(title="Kompass API")

# Une connexion DuckDB par PROCESSUS uvicorn, ouverte une seule fois au
# chargement du module — pas par requête. Équivalent du st.cache_resource
# de app.py, mais ici la raison d'être est différente : Streamlit
# ré-exécute tout le script à chaque interaction (d'où le besoin d'un
# cache explicite), alors qu'un module Python n'est importé qu'une fois
# par processus de toute façon. read_only=True : l'API ne modifie jamais
# la base, seuls les scripts d'ingestion le font.
_connection = duckdb.connect(str(DUCKDB_PATH), read_only=True)


# pydantic.BaseModel (nouveau concept) : décrit la forme exacte d'une
# réponse JSON. FastAPI l'utilise pour valider automatiquement les
# champs et générer la documentation interactive (/docs) — pas besoin de
# construire le JSON à la main comme on le ferait avec un dict brut.
class SegmentSummary(BaseModel):
    id: int
    name: str
    distance_m: float
    elevation_gain_m: float


@app.get("/segments", response_model=list[SegmentSummary])
def list_segments() -> list[SegmentSummary]:
    """Équivalent de load_segments() dans app.py, mais avec distance et
    D+ inclus directement : app.py les récupérait après coup (au moment
    du choix du segment), ici le frontend a besoin de tout d'un coup
    pour peindre la liste déroulante sans un second aller-retour.
    """
    rows = _connection.execute(
        "SELECT id, name, distance_m, total_elevation_gain_m FROM segments ORDER BY name"
    ).fetchall()
    return [
        SegmentSummary(id=row[0], name=row[1], distance_m=row[2], elevation_gain_m=row[3])
        for row in rows
    ]


class PredictRequest(BaseModel):
    segment_id: int
    draft_preset: str
    mass_kg: float


class CalibrationInfo(BaseModel):
    """Valeurs affichées en petit sous "Meilleure fenêtre" dans app.py —
    utiles pour comprendre D'OÙ vient une prédiction, pas juste le
    résultat brut."""

    cp_watts: float
    cp_watts_std: float
    cda_m2: float
    crr: float


class Window(BaseModel):
    time: datetime
    predicted_time_s: float
    required_power_w: float
    wind_speed_ms: float
    wind_direction_rad: float
    temperature_k: float


class PredictResponse(BaseModel):
    calibration: CalibrationInfo
    # windows[0] EST la meilleure fenêtre (rank_forecast_windows_for_segment
    # renvoie déjà trié par temps croissant, T-27) — pas de champ "best"
    # séparé qui dupliquerait windows[0], c'est au frontend de le savoir.
    windows: list[Window]


# def, pas async def (nouveau concept) : le corps fait de l'I/O bloquant
# (DuckDB, httpx.Client synchrone) — FastAPI exécute automatiquement les
# routes "def" classiques dans un thread séparé, pour ne pas geler le
# serveur pendant cet appel. "async def" ne serait utile qu'avec des
# bibliothèques conçues pour ça (ex. httpx.AsyncClient), pas ici.
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        draft_ratio = draft_ratio_for_preset(request.draft_preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cp_fit = fit_current_cp(_connection)
    cda_crr_fit = calibrate_cda_crr_from_db(
        _connection, CSV_PATH, mass_kg=request.mass_kg, cp_fit=cp_fit
    )
    effective_cda_m2 = cda_crr_fit.cda_m2 * draft_ratio

    try:
        with httpx.Client(timeout=30.0) as client:
            windows = rank_forecast_windows_for_segment(
                client,
                _connection,
                request.segment_id,
                request.mass_kg,
                effective_cda_m2,
                cda_crr_fit.crr,
                cp_fit,
            )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not windows:
        raise HTTPException(
            status_code=404,
            detail="Aucun créneau exploitable sur les 10 prochains jours (6h-21h).",
        )

    return PredictResponse(
        calibration=CalibrationInfo(
            cp_watts=cp_fit.cp_watts,
            cp_watts_std=cp_fit.cp_watts_std,
            cda_m2=cda_crr_fit.cda_m2,
            crr=cda_crr_fit.crr,
        ),
        windows=[
            Window(
                time=w.time,
                predicted_time_s=w.predicted_time_s,
                required_power_w=w.required_power_w,
                wind_speed_ms=w.wind_speed_ms,
                wind_direction_rad=w.wind_direction_rad,
                temperature_k=w.temperature_k,
            )
            for w in windows
        ],
    )
