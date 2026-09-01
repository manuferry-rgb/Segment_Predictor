# Kompass — prédicteur de temps sur segment Strava

## Objectif du projet

Prédire le temps réalisable sur un segment Strava donné, en fonction de :
- la courbe de puissance de l'athlète (modèle Critical Power)
- le profil du segment (pente, cap, longueur)
- les conditions météo (vent, température, pression)
- l'état de forme du jour (charge d'entraînement, HRV, sommeil)
- le scénario de draft (solo, dans une roue, dans un groupe)

Puis identifier la meilleure fenêtre horaire sur les 10 prochains jours
pour tenter un record, et proposer une stratégie de pacing optimale.

Projet mono-utilisateur pour l'instant. Pas d'authentification,
pas de multi-tenant, pas d'application mobile.

## Contexte sur l'auteur — IMPORTANT

Je suis ingénieur BI/data avec ~18 ans d'expérience : SQL, modélisation,
ETL, data warehousing, Power BI, Databricks, un peu de Python et de
JavaScript. Je suis en revanche **débutant en développement d'application
Python moderne** et en outillage (uv, pytest, packaging, CI).

Ce projet est un projet d'apprentissage autant qu'un projet de portfolio.
Je dois pouvoir expliquer chaque ligne en entretien.

## Comment tu dois travailler avec moi

1. **Explique avant de coder.** Avant toute implémentation non triviale,
   décris en 5 lignes ce que tu vas faire et pourquoi cette approche.
   Attends ma validation.

2. **Petits incréments.** Maximum ~150 lignes de code par étape.
   Si une tâche est plus grosse, découpe-la et propose-moi le découpage.

3. **Jamais de dépendance sans me demander.** Explique à quoi elle sert
   et ce qu'on perdrait sans elle.

4. **Commente le "pourquoi", pas le "quoi".** Je lis le code.
   Ce que je ne devine pas, ce sont les décisions.

5. **Signale-moi les concepts nouveaux.** Quand tu utilises un pattern ou
   une notion que je n'ai probablement jamais croisée (dataclass, générateur,
   injection de dépendance, fixture pytest...), ajoute une phrase
   d'explication dans ta réponse — pas dans le code.

6. **Test avant code** pour toute la logique métier (physique, modèles,
   calibration). Le test doit échouer d'abord.

7. **Ne jamais inventer de données.** Pas de valeurs par défaut silencieuses,
   pas de `try/except` qui avale une erreur. Si une donnée manque,
   la fonction lève une exception explicite.

8. **Sois honnête sur les limites.** Si un modèle est bancal ou si une
   approximation est grossière, dis-le et documente-le.

## Stack

- Python 3.12, gestion de projet avec `uv`
- DuckDB pour le stockage analytique, Parquet pour les données brutes
- `httpx` pour les appels API, `pydantic` pour la validation des schémas
- `numpy` / `scipy` pour la physique et l'optimisation
- `pytest` pour les tests, `ruff` pour le lint et le format
- Streamlit pour l'interface (phase tardive uniquement)
- GitHub Actions pour la CI

## Structure

```
src/kompass/
  ingest/        # appels API bruts -> Parquet, aucune transformation
  storage/       # schéma DuckDB, chargement, requêtes
  models/
    power.py     # courbe de puissance, modèle Critical Power
    physics.py   # équation de puissance du cycliste, résolution en vitesse
    draft.py     # facteur de réduction aérodynamique
    wbal.py      # bilan W', modèle de Skiba
    pacing.py    # optimisation du profil de puissance (prog. dynamique)
    form.py      # indice de performance du jour
  calibrate/     # estimation de CdA, Crr, CP, W' sur données historiques
  predict/       # orchestration : segment + météo + forme -> temps prédit
tests/
notebooks/       # exploration uniquement, jamais de logique métier ici
```

Règle d'architecture : la couche `ingest` ne transforme rien.
La couche `models` ne fait aucun I/O — elle prend des structures en entrée
et retourne des structures. C'est ce qui la rend testable.

## Commandes

```bash
uv sync                  # installer les dépendances
uv run pytest            # lancer les tests
uv run pytest -k physics # lancer un sous-ensemble
uv run ruff check .      # lint
uv run ruff format .     # formatage
```

## Conventions

- Type hints partout dans `src/`.
- Unités SI en interne (mètres, secondes, watts, kg, m/s). La conversion
  vers km/h ou mm:ss se fait uniquement à l'affichage. Suffixer les noms
  de variables ambigus : `speed_ms`, `duration_s`, `distance_m`.
- Les secrets vont dans `.env`, jamais dans le code. `.env` est gitignored.
- Un commit par unité logique, message en anglais à l'impératif.

## Notes de domaine

- L'endpoint `/segments/{id}/leaderboard` de l'API Strava n'est plus
  accessible. Utiliser `GET /segments/{id}` : le champ `xoms.kom` donne
  le temps KOM sous forme de chaîne formatée ("50:03"), et
  `athlete_segment_stats` donne mon propre record.
- Strava ne fournit ni HRV ni sommeil. Ces données viennent de l'API
  intervals.icu.
- Météo historique et prévisionnelle : Open-Meteo (gratuit, sans clé).
- Les efforts réalisés en groupe faussent la calibration du CdA.
  Ils doivent être taggés et exclus de la calibration.
