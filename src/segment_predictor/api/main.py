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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
from segment_predictor.predict.wind_scan import scan_segments_for_today

# 4 parents : main.py -> api/ -> segment_predictor/ -> src/ -> racine du
# projet (app.py, lui, est à la racine et n'a besoin que d'un seul .parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"
WEB_DIR = PROJECT_ROOT / "web"

# 300, pas 2000 (T-32, cf app.py) : chaque tirage simule TOUS les
# tronçons du polyline, jusqu'à ~340 sur un long segment — 300 suffit à
# stabiliser moyenne et écart-type (mesuré : <1s d'écart contre 2000
# tirages), pour un temps de calcul très inférieur.
N_MONTE_CARLO_SAMPLES = 300
# Hypothèse ASSUMÉE, pas mesurée (T-28) : pas d'historique
# prévision-vs-réalisé disponible pour la calibrer.
WIND_RELATIVE_STD = 0.20

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


class PacingInfo(BaseModel):
    """Une seule puissance, pas un profil variable (T-26 dans app.py) :
    aucun profil pente/distance détaillé n'est stocké au niveau segment
    (T-07b jamais fait), donc le segment est optimisé comme UN SEUL
    tronçon à pente moyenne — "pacing" ici veut dire "la puissance
    soutenable optimale pour ce profil simplifié", pas une vraie
    stratégie qui varierait dans le segment."""

    power_w: float


class KomInfo(BaseModel):
    seconds: int
    # Puissance estimée par TON modèle CP pour TENIR ce temps — pas la
    # puissance réelle du recordman (Strava ne la fournit pas).
    power_w: float
    # True si `seconds` tombe hors de la plage de durées sur laquelle
    # CP/W' ont été calibrés (fit_current_cp) — extrapolation, donc moins
    # fiable, signalé plutôt que présenté comme aussi sûr que dans la
    # plage calibrée.
    power_w_extrapolated: bool


class PrEffortInfo(BaseModel):
    activity_id: int
    start_date: datetime
    average_watts: float | None
    # False si average_watts n'est pas confirmé par un capteur de
    # puissance (ex. estimé par Strava depuis la vitesse) — None si
    # average_watts lui-même est absent (pas de capteur du tout ce
    # jour-là, T-07b).
    sensor_confirmed: bool | None
    # "solo" / "roue_collee" / "un_metre" / "groupe" / "unknown" (T-16) —
    # un PR obtenu dans une roue serait plus rapide qu'un effort solo à
    # puissance égale, la comparaison peut être biaisée si != "solo".
    draft_status: str


class PrInfo(BaseModel):
    seconds: int
    # None si l'effort correspondant n'a pas été retrouvé dans
    # segment_efforts (ne devrait pas arriver en usage normal, mais
    # possible si la base a été partiellement reconstruite).
    effort: PrEffortInfo | None


class UncertaintyInfo(BaseModel):
    """Sur la meilleure fenêtre (windows[0]) uniquement — recalculer pour
    chaque créneau du classement coûterait cher (Monte-Carlo) pour un
    intérêt marginal, même choix que app.py."""

    mean_time_s: float
    std_time_s: float
    n_samples: int
    n_excluded: int


class PredictResponse(BaseModel):
    calibration: CalibrationInfo
    # windows[0] EST la meilleure fenêtre (rank_forecast_windows_for_segment
    # renvoie déjà trié par temps croissant, T-27) — pas de champ "best"
    # séparé qui dupliquerait windows[0], c'est au frontend de le savoir.
    windows: list[Window]
    pacing: PacingInfo
    kom: KomInfo
    # None si jamais roulé ce segment (pr_seconds NULL en base) — pas une
    # erreur, un fait normal pour un segment jamais tenté.
    pr: PrInfo | None
    # None si pas assez d'efforts proches du maximum dans les 90 derniers
    # jours (T-23) pour estimer une distribution de forme à échantillonner.
    uncertainty: UncertaintyInfo | None


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

    # Un seul tronçon (comme app.py, T-26) : optimize_pacing simule à
    # vent nul, donc un cap unique ne change rien à son résultat — pas la
    # peine d'y payer le coût du découpage par polyline (T-32) pour zéro
    # différence.
    distance_m, average_grade, heading_rad, polyline, kom_seconds, pr_seconds = _connection.execute(
        "SELECT distance_m, average_grade, heading_rad, polyline, kom_seconds, pr_seconds "
        "FROM segments WHERE id = ?",
        [request.segment_id],
    ).fetchone()
    chunk = SegmentChunk(0.0, distance_m, average_grade, heading_rad)
    pacing_result = optimize_pacing(
        [chunk],
        cp_fit.cp_watts,
        cp_fit.w_prime_joules,
        request.mass_kg,
        effective_cda_m2,
        cda_crr_fit.crr,
    )

    kom_power_w = sustainable_power_w(cp_fit.cp_watts, cp_fit.w_prime_joules, kom_seconds)
    duration_min_s, duration_max_s = cp_fit.duration_range_s
    kom_info = KomInfo(
        seconds=kom_seconds,
        power_w=kom_power_w,
        power_w_extrapolated=not duration_min_s <= kom_seconds <= duration_max_s,
    )

    pr_info = None
    if pr_seconds is not None:
        # Retrouvé par (segment_id, elapsed_time_s) : segments.pr_seconds
        # ne porte pas l'id de l'effort correspondant, pas de jointure
        # directe possible (même limite que app.py).
        pr_effort_row = _connection.execute(
            "SELECT id, average_watts, device_watts, start_date, activity_id "
            "FROM segment_efforts WHERE segment_id = ? AND elapsed_time_s = ? "
            "ORDER BY start_date DESC LIMIT 1",
            [request.segment_id, pr_seconds],
        ).fetchone()
        effort_info = None
        if pr_effort_row is not None:
            pr_effort_id, average_watts, device_watts, start_date, activity_id = pr_effort_row
            draft_status = load_existing_annotations(CSV_PATH).get(pr_effort_id, "unknown")
            effort_info = PrEffortInfo(
                activity_id=activity_id,
                start_date=start_date,
                average_watts=average_watts,
                sensor_confirmed=bool(device_watts) if average_watts is not None else None,
                draft_status=draft_status,
            )
        pr_info = PrInfo(seconds=pr_seconds, effort=effort_info)

    # Incertitude (T-28) : cap réel par tronçon (T-32), pas le chunk
    # unique du pacing ci-dessus — le vent compte ici, un cap moyen
    # unique le fausserait (voir HBFH, T-32).
    performance_index_samples = recent_performance_index_values(_connection, cp_fit=cp_fit)
    uncertainty_info = None
    if performance_index_samples:
        best = windows[0]
        chunks = segment_chunks_from_polyline(decode_polyline(polyline), average_grade)
        uncertainty_result = propagate_uncertainty(
            chunks,
            cp_watts=cp_fit.cp_watts,
            cp_watts_std=cp_fit.cp_watts_std,
            w_prime_joules=cp_fit.w_prime_joules,
            w_prime_joules_std=cp_fit.w_prime_joules_std,
            mass_kg=request.mass_kg,
            cda_m2=effective_cda_m2,
            crr=cda_crr_fit.crr,
            performance_index_samples=performance_index_samples,
            wind_speed_ms=best.wind_speed_ms,
            wind_direction_rad=best.wind_direction_rad,
            wind_relative_std=WIND_RELATIVE_STD,
            n_samples=N_MONTE_CARLO_SAMPLES,
        )
        uncertainty_info = UncertaintyInfo(
            mean_time_s=uncertainty_result.mean_time_s,
            std_time_s=uncertainty_result.std_time_s,
            n_samples=uncertainty_result.n_samples,
            n_excluded=uncertainty_result.n_excluded,
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
        pacing=PacingInfo(power_w=pacing_result.power_profile_w[0]),
        kom=kom_info,
        pr=pr_info,
        uncertainty=uncertainty_info,
    )


class WindOpportunity(BaseModel):
    segment_id: int
    segment_name: str
    distance_m: float
    best_hour: datetime
    # m/s, signé (T-34) : positif = vent de dos en moyenne sur le
    # segment, négatif = vent de face en moyenne.
    average_tailwind_speed_ms: float
    wind_speed_ms: float
    wind_direction_rad: float


@app.get("/wind-scan", response_model=list[WindOpportunity])
def wind_scan() -> list[WindOpportunity]:
    """Équivalent JSON de pages/1_Segments_du_jour.py (T-33/T-34) :
    aucune calibration CP/CdA/Crr, juste la géométrie de chaque segment
    favori contre la météo du jour. Un appel Open-Meteo par segment
    (scan_segments_for_today) — peut prendre plusieurs secondes selon le
    nombre de segments favoris.
    """
    with httpx.Client(timeout=30.0) as client:
        opportunities = scan_segments_for_today(client, _connection)
    return [
        WindOpportunity(
            segment_id=o.segment_id,
            segment_name=o.segment_name,
            distance_m=o.distance_m,
            best_hour=o.best_hour,
            average_tailwind_speed_ms=o.average_tailwind_speed_ms,
            wind_speed_ms=o.wind_speed_ms,
            wind_direction_rad=o.wind_direction_rad,
        )
        for o in opportunities
    ]


# Montage APRÈS toutes les routes API ci-dessus (ordre significatif) :
# StaticFiles(html=True) sert index.html pour "/" et toute route inconnue
# lui est déléguée en dernier recours — monté avant /segments, /predict,
# /wind-scan, il les masquerait puisque "/" matche tout.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
