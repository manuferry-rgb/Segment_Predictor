"""T-36 : squelette FastAPI — remplace progressivement l'app Streamlit
(T-29). Même rôle que app.py avant lui : assemble des fonctions déjà
testées ailleurs (storage/, predict/, calibrate/, models/), aucune
logique métier nouvelle ici — juste du câblage HTTP et de la
sérialisation JSON. Pas de tests dédiés à ce fichier pour la même
raison que app.py (cf sa docstring, T-29).

Usage : uv run uvicorn segment_predictor.api.main:app --reload
"""

import secrets
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import duckdb
import httpx
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from segment_predictor.calibrate.cda_crr import calibrate_cda_crr_from_db
from segment_predictor.calibrate.draft_tagging import (
    DEFAULT_CP_FIT_DURATIONS_S,
    compute_aggregate_mmp_curve,
    fit_current_cp,
    load_existing_annotations,
)
from segment_predictor.calibrate.form import recent_performance_index_values
from segment_predictor.ingest.strava_activities import fetch_and_store_new_activities
from segment_predictor.ingest.strava_activity_details import fetch_and_store_activity_details
from segment_predictor.ingest.strava_auth import (
    exchange_authorization_code,
    get_valid_access_token_for_user,
)
from segment_predictor.ingest.strava_segments import (
    fetch_and_store_segments,
    list_starred_segment_ids,
)
from segment_predictor.ingest.strava_streams import (
    ensure_path_is_gitignored,
    fetch_and_store_streams,
    list_eligible_activity_ids,
)
from segment_predictor.models.draft import draft_ratio_for_preset
from segment_predictor.models.pacing import optimize_pacing
from segment_predictor.models.polyline import decode_polyline
from segment_predictor.models.power import interpolate_mmp_curve, sustainable_power_w
from segment_predictor.models.segment import (
    SegmentChunk,
    segment_chunks_from_polyline,
    simulate_segment_time_from_mmp_curve,
)
from segment_predictor.models.uncertainty import propagate_uncertainty
from segment_predictor.predict.forecast_window import rank_forecast_windows_for_segment
from segment_predictor.predict.wind_scan import scan_segments_for_today
from segment_predictor.storage.activities import build_activities_table
from segment_predictor.storage.segment_efforts import build_segment_efforts_table
from segment_predictor.storage.segments import (
    build_segments_table,
    build_user_segment_stats_table,
    build_user_starred_segments_table,
)
from segment_predictor.storage.streams import build_streams_table
from segment_predictor.storage.users import ensure_users_table, get_user, upsert_user

# 4 parents : main.py -> api/ -> segment_predictor/ -> src/ -> racine du
# projet (app.py, lui, est à la racine et n'a besoin que d'un seul .parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"
CSV_PATH = PROJECT_ROOT / "annotations" / "draft_status.csv"
WEB_DIR = PROJECT_ROOT / "web"
ENV_PATH = PROJECT_ROOT / ".env"
# T-46 : racine des dossiers bruts par utilisateur (T-44d) — chaque
# source y a son sous-dossier <user_id>/, sauf open_meteo (météo
# partagée), pas fetchée par /sync (voir sa docstring).
RAW_DIR_ROOT = PROJECT_ROOT / "data" / "raw"

_env_values = dotenv_values(ENV_PATH)
STRAVA_CLIENT_ID = _env_values.get("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = _env_values.get("STRAVA_CLIENT_SECRET")
# Signe (pas chiffre) le cookie de session (T-45, SessionMiddleware) :
# n'importe qui peut LIRE son contenu (juste du base64), mais pas le
# FORGER sans cette clé — c'est pour ça qu'on n'y met jamais de tokens
# Strava, seulement un user_id (voir storage/users.py pour les tokens).
SESSION_SECRET_KEY = _env_values.get("SESSION_SECRET_KEY")
if not SESSION_SECRET_KEY:
    raise RuntimeError(
        "SESSION_SECRET_KEY manquant dans .env — requis pour signer le cookie de session (T-45)"
    )

STRAVA_AUTHORIZE_URL = "https://www.strava.com/oauth/authorize"
# read : profil (id, prénom...) ; activity:read_all : mêmes données que
# les scripts d'ingestion CLI utilisent déjà pour un seul utilisateur.
STRAVA_OAUTH_SCOPE = "read,activity:read_all"

# 300, pas 2000 (T-32, cf app.py) : chaque tirage simule TOUS les
# tronçons du polyline, jusqu'à ~340 sur un long segment — 300 suffit à
# stabiliser moyenne et écart-type (mesuré : <1s d'écart contre 2000
# tirages), pour un temps de calcul très inférieur.
N_MONTE_CARLO_SAMPLES = 300
# Hypothèse ASSUMÉE, pas mesurée (T-28) : pas d'historique
# prévision-vs-réalisé disponible pour la calibrer.
WIND_RELATIVE_STD = 0.20

app = FastAPI(title="Segment Chaser API")

# SessionMiddleware (T-45, nouveau concept) : signe un cookie
# (`session_cookie=` ci-dessous, nommé explicitement plutôt que de
# laisser le nom générique "session" par défaut) contenant l'état de
# connexion (user_id) — sans lui, chaque requête serait anonyme,
# impossible de savoir "qui parle" entre /auth/strava/callback et
# /predict. https_only=False : correct en local (http://127.0.0.1) ;
# à repasser à True le jour où l'app tourne derrière un vrai domaine
# HTTPS (T-47), sinon le cookie ne serait plus envoyé du tout par le
# navigateur.
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    session_cookie="segment_chaser_session",
    https_only=False,
)

# Une connexion DuckDB par PROCESSUS uvicorn, ouverte une seule fois au
# chargement du module — pas par requête. Équivalent du st.cache_resource
# de app.py, mais ici la raison d'être est différente : Streamlit
# ré-exécute tout le script à chaque interaction (d'où le besoin d'un
# cache explicite), alors qu'un module Python n'est importé qu'une fois
# par processus de toute façon.
#
# PAS read_only (changé en T-45) : la connexion API doit désormais
# pouvoir écrire dans `users` à chaque connexion/rafraîchissement de
# token. Conséquence pratique inchangée depuis le début du projet : un
# script d'ingestion (écriture, lui aussi) ne peut toujours pas tourner
# EN MÊME TEMPS que le serveur (verrou DuckDB, un seul writer à la fois)
# — il faut l'arrêter, lancer le script, le relancer.
_connection = duckdb.connect(str(DUCKDB_PATH))
ensure_users_table(_connection)


def _db_cursor() -> duckdb.DuckDBPyConnection:
    """Un curseur DuckDB PAR REQUÊTE (nouveau concept, bug réel rencontré
    en testant T-45) : FastAPI exécute les routes "def" classiques dans
    un pool de threads (voir plus haut) — DuckDB ne garantit PAS qu'une
    même connexion supporte des requêtes lancées en parallèle par deux
    threads différents. Constaté en vrai : le navigateur a chargé
    `/auth/me` et `/segments` en même temps, et `/auth/me` a reçu la
    ligne à 4 colonnes de `/segments` au lieu de sa propre requête à 6
    colonnes sur `users` — les deux partageaient `_connection`.
    `.cursor()` "duplique" la connexion (même base, exécution
    indépendante) : chaque route appelle ceci UNE fois en tout début de
    fonction, jamais `_connection` directement après ce point.
    """
    return _connection.cursor()


def _require_user_id(request: Request) -> int:
    """Utilisateur de LA SESSION (T-44e — remplace l'ancienne constante
    `CURRENT_USER_ID` codée en dur, supprimée). 401 explicite si
    personne n'est connecté, plutôt qu'une réponse vide ou un id
    inventé (règle du projet : jamais de valeur par défaut silencieuse).
    """
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Connecte-toi avec Strava (/auth/strava/login) d'abord.",
        )
    return user_id


@app.get("/auth/strava/login")
def strava_login(request: Request) -> RedirectResponse:
    """Redirige vers la page d'autorisation Strava (T-45) — première
    étape du flux OAuth "Se connecter avec Strava".

    `state` (nouveau concept, protection CSRF) : une valeur aléatoire
    posée dans la session AVANT de partir chez Strava, revérifiée au
    retour sur `/auth/strava/callback` — sans ça, un tiers pourrait
    forger un lien de callback avec SON PROPRE code et faire connecter
    la victime à SON compte Strava à elle (attaque documentée du
    protocole OAuth), pas la moindre paranoïa excessive ici.

    `request.url_for("strava_callback")` (nouveau concept) construit
    l'URL absolue de la route ci-dessous à partir du NOM de la fonction
    Python — jamais une URL en dur, qui serait fausse dès que l'app
    tourne ailleurs qu'en local (T-47).
    """
    state = secrets.token_urlsafe(24)
    request.session["oauth_state"] = state
    params = {
        "client_id": STRAVA_CLIENT_ID,
        "redirect_uri": str(request.url_for("strava_callback")),
        "response_type": "code",
        "approval_prompt": "auto",
        "scope": STRAVA_OAUTH_SCOPE,
        "state": state,
    }
    return RedirectResponse(f"{STRAVA_AUTHORIZE_URL}?{urlencode(params)}")


@app.get("/auth/strava/callback")
def strava_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Strava revient ici après que la personne a accepté (ou refusé) sur
    son propre site — voir strava_login pour `state`.

    `error` (ex. "access_denied") : Strava le renvoie si la personne a
    cliqué "Refuser" plutôt que "Autoriser" — un cas normal, pas une
    erreur serveur, donc pas de 500 pour ça.
    """
    if error is not None:
        return RedirectResponse(f"/?auth_error={error}")

    expected_state = request.session.pop("oauth_state", None)
    if expected_state is None or state != expected_state:
        raise HTTPException(status_code=400, detail="state OAuth invalide ou expiré")
    if code is None:
        raise HTTPException(status_code=400, detail="code manquant dans la réponse Strava")

    with httpx.Client(timeout=30.0) as client:
        authorized = exchange_authorization_code(
            client, STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, code
        )

    upsert_user(
        _db_cursor(),
        id=authorized.athlete_id,
        firstname=authorized.firstname,
        lastname=authorized.lastname,
        access_token=authorized.token_state.access_token,
        refresh_token=authorized.token_state.refresh_token,
        expires_at=authorized.token_state.expires_at,
    )
    request.session["user_id"] = authorized.athlete_id
    return RedirectResponse("/")


@app.post("/auth/logout")
def logout(request: Request) -> dict[str, bool]:
    request.session.clear()
    return {"authenticated": False}


class CurrentUser(BaseModel):
    authenticated: bool
    firstname: str | None = None


@app.get("/auth/me", response_model=CurrentUser)
def me(request: Request) -> CurrentUser:
    """Le frontend l'appelle au chargement pour savoir s'afficher "Se
    connecter" ou "Connecté comme {firstname}" (T-45c)."""
    user_id = request.session.get("user_id")
    if user_id is None:
        return CurrentUser(authenticated=False)
    user = get_user(_db_cursor(), user_id)
    if user is None:
        return CurrentUser(authenticated=False)
    return CurrentUser(authenticated=True, firstname=user.firstname)


class SyncSummary(BaseModel):
    new_activities_fetched: bool
    streams_fetched: int
    streams_remaining: int
    streams_quota_reached: bool
    activity_details_fetched: int
    activity_details_remaining: int
    activity_details_quota_reached: bool
    segments_fetched: int
    segments_remaining: int
    segments_quota_reached: bool


@app.post("/sync", response_model=SyncSummary)
def sync(request: Request) -> SyncSummary:
    """Synchronise les données Strava de l'utilisateur CONNECTÉ (T-46) —
    remplace, pour un utilisateur du flux web, les scripts CLI
    (fetch_activities.py, etc.) qui ne savent lire que le .env de
    l'auteur.

    Ne fetche PAS la météo (`activity_weather`) ni le wellness
    (intervals.icu) : vérifié avant d'écrire ce ticket, `activity_
    weather` n'est lue par AUCUN module de calibrate/predict (juste
    construite, jamais consommée) — l'omettre ici ne change rien au
    résultat de /predict. intervals.icu resterait de toute façon propre
    à l'auteur (pas de compte par utilisateur) — hors périmètre.

    Synchrone, pas de file d'attente/tâche de fond (T-47 pourra revoir
    ça si le volume l'impose un jour) : peut prendre de quelques
    secondes à plusieurs minutes selon l'historique Strava de la
    personne — le frontend affiche un statut d'attente (T-46, comme
    /wind-scan avant lui).

    Le quota Strava est PAR APPLICATION, pas par utilisateur (voir
    ROADMAP.md, Phase 14) : un utilisateur peut donc être arrêté par du
    quota consommé par un AUTRE utilisateur — les indicateurs
    `*_quota_reached` le signalent plutôt que de le cacher.
    """
    user_id = _require_user_id(request)
    conn = _db_cursor()

    user_id_str = str(user_id)
    activities_raw_dir = RAW_DIR_ROOT / "strava_activities" / user_id_str
    streams_raw_dir = RAW_DIR_ROOT / "strava_streams" / user_id_str
    segments_raw_dir_root = RAW_DIR_ROOT / "strava_segments"
    segments_raw_dir = segments_raw_dir_root / user_id_str
    details_raw_dir = RAW_DIR_ROOT / "strava_activity_details" / user_id_str

    with httpx.Client(timeout=30.0) as client:
        access_token = get_valid_access_token_for_user(
            client, conn, STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, user_id
        )

        new_activities_path = fetch_and_store_new_activities(
            client, access_token, activities_raw_dir
        )

        ensure_path_is_gitignored(streams_raw_dir, PROJECT_ROOT)
        streams_summary = fetch_and_store_streams(
            client, access_token, activities_raw_dir, streams_raw_dir
        )

        eligible_ids = list_eligible_activity_ids(activities_raw_dir)
        details_summary = fetch_and_store_activity_details(
            client, access_token, eligible_ids, details_raw_dir
        )

        segment_ids = list_starred_segment_ids(client, access_token)
        segments_summary = fetch_and_store_segments(
            client, access_token, segment_ids, segments_raw_dir
        )

    build_activities_table(conn, activities_raw_dir, user_id=user_id)
    build_streams_table(conn, streams_raw_dir, user_id=user_id)
    build_segments_table(conn, segments_raw_dir_root)
    build_user_segment_stats_table(conn, segments_raw_dir, user_id=user_id)
    build_user_starred_segments_table(conn, segments_raw_dir, user_id=user_id)
    build_segment_efforts_table(conn, details_raw_dir, user_id=user_id)

    return SyncSummary(
        new_activities_fetched=new_activities_path is not None,
        streams_fetched=len(streams_summary.fetched_activity_ids),
        streams_remaining=len(streams_summary.remaining_activity_ids),
        streams_quota_reached=streams_summary.stopped_due_to_daily_quota,
        activity_details_fetched=len(details_summary.fetched_ids),
        activity_details_remaining=len(details_summary.remaining_ids),
        activity_details_quota_reached=details_summary.stopped_due_to_daily_quota,
        segments_fetched=len(segments_summary.fetched_ids),
        segments_remaining=len(segments_summary.remaining_ids),
        segments_quota_reached=segments_summary.stopped_due_to_daily_quota,
    )


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
def list_segments(request: Request) -> list[SegmentSummary]:
    """Équivalent de load_segments() dans app.py, mais avec distance et
    D+ inclus directement : app.py les récupérait après coup (au moment
    du choix du segment), ici le frontend a besoin de tout d'un coup
    pour peindre la liste déroulante sans un second aller-retour.

    Filtré aux favoris de l'utilisateur CONNECTÉ via `user_starred_
    segments` (T-44c/T-44e) — `segments` elle-même est partagée entre
    tous les utilisateurs, sans ce filtre chacun verrait aussi les
    segments favoris de tout le monde.
    """
    user_id = _require_user_id(request)
    rows = (
        _db_cursor()
        .execute(
            "SELECT s.id, s.name, s.distance_m, s.total_elevation_gain_m FROM segments s "
            "JOIN user_starred_segments u ON u.segment_id = s.id "
            "WHERE u.user_id = ? ORDER BY s.name",
            [user_id],
        )
        .fetchall()
    )
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


class RealPowerCurveEstimate(BaseModel):
    """Puissance lue directement sur la courbe MMP RÉELLEMENT MESURÉE
    (T-42a/T-42b), pas sur le modèle CP+W' lissé qui alimente `windows`
    ci-dessous — répond à "si je donnais vraiment ma meilleure puissance
    déjà atteinte pour cette durée, avec le vent de cette fenêtre, quel
    temps ça donnerait ?". Ajouté à côté du classement, ne le remplace
    pas (décision explicite, cf simulate_segment_time_from_mmp_curve)."""

    predicted_time_s: float
    power_w: float


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
    # None si le temps converge hors de la plage mesurée par la courbe MMP
    # (~3-20 min, T-42a) — real_power_curve_unavailable_reason explique
    # pourquoi plutôt qu'un null silencieux (message d'interpolate_mmp_curve
    # ou de simulate_segment_time_from_mmp_curve tel quel, pas reformulé).
    real_power_curve: RealPowerCurveEstimate | None
    real_power_curve_unavailable_reason: str | None


# def, pas async def (nouveau concept) : le corps fait de l'I/O bloquant
# (DuckDB, httpx.Client synchrone) — FastAPI exécute automatiquement les
# routes "def" classiques dans un thread séparé, pour ne pas geler le
# serveur pendant cet appel. "async def" ne serait utile qu'avec des
# bibliothèques conçues pour ça (ex. httpx.AsyncClient), pas ici.
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, http_request: Request) -> PredictResponse:
    # Deux paramètres "request" différents (nouveau piège, T-44e) :
    # `request` est le corps JSON envoyé par le frontend (segment_id,
    # etc.), `http_request` est LA requête HTTP elle-même, seule à
    # porter la session — FastAPI les distingue par leur TYPE
    # (PredictRequest vs Request), pas leur nom, mais un nom identique
    # aurait été trompeur à la lecture.
    user_id = _require_user_id(http_request)
    # Un seul curseur pour TOUTE la requête (voir _db_cursor) — pas un
    # nouveau à chaque appel : les appels de cette fonction sont
    # séquentiels sur le même thread, seul le PARTAGE entre requêtes
    # concurrentes est le problème.
    conn = _db_cursor()
    try:
        draft_ratio = draft_ratio_for_preset(request.draft_preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cp_fit = fit_current_cp(conn)
    cda_crr_fit = calibrate_cda_crr_from_db(conn, CSV_PATH, mass_kg=request.mass_kg, cp_fit=cp_fit)
    effective_cda_m2 = cda_crr_fit.cda_m2 * draft_ratio

    try:
        with httpx.Client(timeout=30.0) as client:
            windows = rank_forecast_windows_for_segment(
                client,
                conn,
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
    distance_m, average_grade, heading_rad, polyline, kom_seconds = conn.execute(
        "SELECT distance_m, average_grade, heading_rad, polyline, kom_seconds "
        "FROM segments WHERE id = ?",
        [request.segment_id],
    ).fetchone()
    # PR : table PAR ATHLÈTE depuis T-44c (segments.pr_seconds n'existe
    # plus, ce n'était jamais une propriété du segment lui-même — voir
    # storage/segments.py). `user_id` vient de LA SESSION (T-44e).
    pr_stats_row = conn.execute(
        "SELECT pr_seconds FROM user_segment_stats WHERE user_id = ? AND segment_id = ?",
        [user_id, request.segment_id],
    ).fetchone()
    pr_seconds = pr_stats_row[0] if pr_stats_row is not None else None
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
        pr_effort_row = conn.execute(
            "SELECT id, average_watts, device_watts, start_date, activity_id "
            "FROM segment_efforts WHERE user_id = ? AND segment_id = ? AND elapsed_time_s = ? "
            "ORDER BY start_date DESC LIMIT 1",
            [user_id, request.segment_id, pr_seconds],
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

    # Cap réel par tronçon (T-32), pas le chunk unique du pacing ci-dessus —
    # réutilisé par l'incertitude ET par la comparaison "courbe réelle"
    # (T-42) juste en dessous : le vent compte dans les deux cas, un cap
    # moyen unique le fausserait (voir HBFH, T-32).
    best = windows[0]
    chunks = segment_chunks_from_polyline(decode_polyline(polyline), average_grade)

    # Incertitude (T-28)
    performance_index_samples = recent_performance_index_values(conn, cp_fit=cp_fit)
    uncertainty_info = None
    if performance_index_samples:
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

    # Comparaison "courbe de puissance réelle" (T-42) : amorcée par
    # best.predicted_time_s (déjà une bonne estimation, cf modèle CP+W')
    # plutôt que l'amorce générique — réduit le risque de sortir de la
    # plage mesurée (~3-20 min) avant d'avoir convergé. N'affecte ni le
    # classement ni predicted_time_s ci-dessus, uniquement ce chiffre de
    # comparaison.
    mmp_curve = compute_aggregate_mmp_curve(conn, DEFAULT_CP_FIT_DURATIONS_S)
    real_power_curve_info = None
    real_power_curve_unavailable_reason = None
    try:
        real_time_s = simulate_segment_time_from_mmp_curve(
            chunks,
            mmp_curve,
            request.mass_kg,
            effective_cda_m2,
            cda_crr_fit.crr,
            wind_speed_ms=best.wind_speed_ms,
            wind_direction_rad=best.wind_direction_rad,
            initial_guess_s=best.predicted_time_s,
        )
        real_power_curve_info = RealPowerCurveEstimate(
            predicted_time_s=real_time_s,
            power_w=interpolate_mmp_curve(mmp_curve, real_time_s),
        )
    except ValueError as exc:
        real_power_curve_unavailable_reason = str(exc)

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
        real_power_curve=real_power_curve_info,
        real_power_curve_unavailable_reason=real_power_curve_unavailable_reason,
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
    # % PUR d'alignement tracé/vent (T-43), indépendant de la force du
    # vent — complète average_tailwind_speed_ms, ne remplace pas le
    # classement (toujours par vitesse réelle ci-dessus).
    wind_alignment_pct: float


@app.get("/wind-scan", response_model=list[WindOpportunity])
def wind_scan(request: Request) -> list[WindOpportunity]:
    """Équivalent JSON de pages/1_Segments_du_jour.py (T-33/T-34) :
    aucune calibration CP/CdA/Crr, juste la géométrie de chaque segment
    favori (de l'utilisateur CONNECTÉ, T-44e) contre la météo du jour.
    Un appel Open-Meteo par segment (scan_segments_for_today) — peut
    prendre plusieurs secondes selon le nombre de segments favoris.
    """
    user_id = _require_user_id(request)
    with httpx.Client(timeout=30.0) as client:
        opportunities = scan_segments_for_today(client, _db_cursor(), user_id)
    return [
        WindOpportunity(
            segment_id=o.segment_id,
            segment_name=o.segment_name,
            distance_m=o.distance_m,
            best_hour=o.best_hour,
            average_tailwind_speed_ms=o.average_tailwind_speed_ms,
            wind_speed_ms=o.wind_speed_ms,
            wind_direction_rad=o.wind_direction_rad,
            wind_alignment_pct=o.wind_alignment_pct,
        )
        for o in opportunities
    ]


# Montage APRÈS toutes les routes API ci-dessus (ordre significatif) :
# StaticFiles(html=True) sert index.html pour "/" et toute route inconnue
# lui est déléguée en dernier recours — monté avant /segments, /predict,
# /wind-scan, il les masquerait puisque "/" matche tout.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
