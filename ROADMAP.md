# Roadmap — un ticket = une session Claude Code

Règle : un seul ticket à la fois. On ne passe au suivant que quand les
tests passent et que le commit est fait. Si un ticket prend plus de 2h,
c'est qu'il était trop gros — le redécouper.

---

## Phase 0 — Socle

**T-01 — Squelette du projet**
Initialiser le projet avec `uv`, layout `src/`, `pyproject.toml`,
`.gitignore`, `README.md` minimal, `pytest` et `ruff` configurés.
*Critère de fin* : `uv run pytest` passe (avec un seul test trivial),
`uv run ruff check .` est propre.

**T-02 — Intégration continue**
GitHub Actions : lint + tests sur chaque push.
*Critère de fin* : le badge est vert sur le repo.

---

## Phase 1 — Ingestion

**T-03 — Authentification Strava**
Flow OAuth, stockage du refresh token dans `.env`, rafraîchissement
automatique du token d'accès.
*Critère de fin* : un script affiche mon nom d'athlète.

**T-04 — Récupération des activités**
Lister mes activités, sauvegarder le JSON brut en Parquet daté.
Gérer la pagination et les limites de débit de l'API.
*Critère de fin* : N activités sur disque, relançable sans doublon.

**T-05 — Récupération des streams**
Pour chaque activité : puissance, altitude, latlng, temps, cadence, FC.
*Critère de fin* : streams stockés, taille du dataset documentée.

**T-06 — Segments et efforts**
`GET /segments/{id}` pour mes segments favoris, extraction du KOM
depuis `xoms` et de mon PR depuis `athlete_segment_stats`.
Parser le format "mm:ss" vers des secondes.
*Critère de fin* : une table `segments` en DuckDB avec KOM et PR.

**T-07 — Schéma DuckDB**
Modéliser proprement : `activities`, `streams`, `segments`,
`segment_efforts`. Séparer les vues brutes des vues transformées.
*Critère de fin* : le schéma est documenté dans le README.

---

## Phase 2 — Courbe de puissance

**T-08 — Mean maximal power**
Calculer la puissance maximale moyenne sur toutes les durées
(1 s à 3 h) sur l'ensemble des activités.
*Critère de fin* : test unitaire sur un signal synthétique connu.

**T-09 — Modèle Critical Power**
Ajuster `P(t) = CP + W'/t` sur la courbe. Sortir CP, W' et la qualité
de l'ajustement.
*Critère de fin* : mes valeurs de CP et W', comparées à mon ressenti.

---

## Phase 3 — Physique

**T-10 — Équation de puissance du cycliste**
Fonction pure : puissance requise pour une vitesse donnée, sur un
tronçon de pente et cap connus, avec vent.
*Critère de fin* : tests contre des valeurs de référence connues
(cas plat sans vent, cas montée sans aéro).

**T-11 — Résolution inverse**
Étant donné une puissance, trouver la vitesse. Cubique en v,
résolution par Brent.
*Critère de fin* : aller-retour puissance -> vitesse -> puissance stable.

**T-12 — Découpage du segment**
Depuis le stream d'altitude et latlng : tronçons de ~50 m avec pente
et cap. Lisser le bruit d'altitude GPS.
*Critère de fin* : le dénivelé total recalculé correspond à Strava.

**T-13 — Simulation d'un segment complet**
Itérer sur les tronçons, avec la boucle de convergence entre temps
prédit et puissance tenable.
*Critère de fin* : un temps prédit pour un segment réel.

---

## Phase 4 — Météo

**T-14 — Météo historique**
Open-Meteo archive : vent, température, pression pour chaque effort passé.
*Critère de fin* : chaque effort enrichi de ses conditions.

**T-15 — Vent effectif**
Projeter le vent sur le cap de chaque tronçon.
*Critère de fin* : tests sur les cas plein dos, plein face, travers.

---

## Phase 5 — Calibration

**T-16 — Tag des efforts en groupe**
Interface minimale (CSV ou CLI) pour marquer mes efforts groupés.
*Critère de fin* : colonne `is_drafted` renseignée.

**T-17 — Estimation de CdA et Crr**
Minimiser l'écart entre temps modélisés et temps réels sur les
efforts solo uniquement.
*Critère de fin* : valeurs plausibles (CdA entre 0.25 et 0.40).

**T-18 — Backtest**
Split temporel, erreur absolue médiane en secondes, graphique
prédit vs réel.
*Critère de fin* : le chiffre d'erreur est dans le README.

---

## Phase 6 — Draft

**T-19 — Facteur de draft**
Formule d'Olds, presets solo / roue collée / un mètre / groupe.
*Critère de fin* : tests montrant que le gain est fort sur le plat
et quasi nul en forte pente.

**T-20 — Comparaison de scénarios**
Simuler le même segment sous plusieurs scénarios de draft.
*Critère de fin* : un tableau comparatif des temps.

---

## Phase 7 — Forme

**T-21 — Charge d'entraînement**
Modèle de Banister : CTL, ATL, TSB à partir du TSS quotidien.
*Critère de fin* : courbes cohérentes avec mon vécu.

**T-22 — Données de wellness**
Ingestion intervals.icu : HRV, sommeil, FC de repos, poids.
*Critère de fin* : table `wellness` alimentée.

**T-23 — Indice de performance**
Ratio puissance réelle / puissance prédite par le modèle CP,
sur les efforts maximaux uniquement. Définir et documenter le
filtre "effort maximal".
*Critère de fin* : la série temporelle de l'indice.

**T-24 — Modèle de forme**
Régression régularisée de l'indice sur les features de forme,
validation croisée temporelle.
*Critère de fin* : R² honnête, y compris s'il est faible.

---

## Phase 8 — Pacing

**T-25 — Bilan W'**
Modèle de Skiba : consommation au-dessus de CP, recharge en dessous.
*Critère de fin* : tests sur profils synthétiques.

**T-26 — Optimisation par programmation dynamique**
État = W' restant, décision = puissance par tronçon, objectif =
temps minimal sous contrainte W' >= 0.
*Critère de fin* : un profil de puissance optimal, plus rapide qu'un
profil à puissance constante.

---

## Phase 9 — Produit

**T-27 — Prévision sur 10 jours**
Open-Meteo forecast, évaluation créneau par créneau, classement.
*Critère de fin* : "meilleure fenêtre : jeudi 17h".

**T-28 — Propagation de l'incertitude**
Monte-Carlo sur l'incertitude du CP, de la forme et de la météo.
Sortir un intervalle, pas un point.
*Critère de fin* : "4:12 ± 8 s".

**T-29 — Interface Streamlit**
Choix du segment, scénario de draft, affichage de la fenêtre
optimale et de la stratégie de pacing.
*Critère de fin* : utilisable sans toucher au code.

**T-30 — README de portfolio**
Schéma d'architecture, décisions techniques et leurs justifications,
chiffres de performance du modèle, limites connues.
*Critère de fin* : lisible par un recruteur en 3 minutes.
