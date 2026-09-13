# T-47a : image de production pour l'API FastAPI (api/main.py), pas pour
# l'ancienne app Streamlit (app.py, racine du dépôt) — celle-ci n'a jamais
# besoin de tourner en dehors de mon poste, et sera supprimée (T-41,
# deuxième moitié, différée). Seuls src/, web/ et annotations/ sont donc
# copiés dans l'image : ce que l'API sert réellement.
#
# Image de base officielle uv (voir uv.lock) : Python 3.12 + le binaire
# `uv` déjà installés sur Debian slim — évite d'installer uv nous-mêmes
# dans une image python:3.12-slim classique.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# UV_LINK_MODE=copy : par défaut uv essaie de faire des hardlinks depuis
# son cache global vers .venv, impossible entre deux layers Docker
# différents (systèmes de fichiers distincts) — copy évite un warning
# inoffensif mais bruyant à chaque build.
ENV UV_LINK_MODE=copy

# Étape séparée pour les dépendances SEULES (avant de copier le code) :
# Docker met ce layer en cache tant que pyproject.toml/uv.lock ne changent
# pas, donc une modif de src/ ne réinstalle pas duckdb/fastapi/etc à
# chaque build. --no-dev exclut pytest/ruff (dependency-groups.dev,
# pyproject.toml) : inutiles pour faire tourner l'app en production.
# --no-install-project : installe seulement les dépendances, pas encore
# le code (qui n'est pas copié à cette étape).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src/ src/
COPY web/ web/
COPY annotations/ annotations/
# README.md : requis par pyproject.toml (readme = "README.md") pour que
# uv_build puisse construire le package `segment_predictor` ci-dessous
# — sans lui, `uv sync` échoue à "failed to open file README.md" (trouvé
# en simulant cette étape hors Docker avant de l'écrire ici).
COPY README.md ./

# Deuxième `uv sync` : installe maintenant `segment_predictor` lui-même
# (résolu en tant que package local par uv_build, cf pyproject.toml)
# dans le même environnement virtuel déjà créé ci-dessus.
RUN uv sync --locked --no-dev

# data/ : jamais commité (.gitignore — contient le DuckDB + les données
# Strava brutes de chaque utilisateur), donc absent de l'image de base.
# Créé ici pour que le premier démarrage fonctionne même avant qu'un
# volume persistant y soit monté (T-47b) : DuckDB crée le fichier .duckdb
# tout seul au premier `connect()`, mais seulement si le DOSSIER existe déjà.
RUN mkdir -p data

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080

# --host 0.0.0.0 obligatoire : le défaut 127.0.0.1 n'écoute que sur la
# boucle locale DU CONTENEUR, invisible depuis l'extérieur.
CMD ["uvicorn", "segment_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
