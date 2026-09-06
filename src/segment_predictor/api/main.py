"""T-36 : squelette FastAPI — remplace progressivement l'app Streamlit
(T-29). Même rôle que app.py avant lui : assemble des fonctions déjà
testées ailleurs (storage/, predict/, calibrate/, models/), aucune
logique métier nouvelle ici — juste du câblage HTTP et de la
sérialisation JSON. Pas de tests dédiés à ce fichier pour la même
raison que app.py (cf sa docstring, T-29).

Usage : uv run uvicorn segment_predictor.api.main:app --reload
"""

from pathlib import Path

import duckdb
from fastapi import FastAPI
from pydantic import BaseModel

# 4 parents : main.py -> api/ -> segment_predictor/ -> src/ -> racine du
# projet (app.py, lui, est à la racine et n'a besoin que d'un seul .parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "segment_predictor.duckdb"

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
