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
Modéliser proprement : `activities`, `streams`, `segments`.
Séparer les vues brutes (schéma `raw`) des tables transformées
(schéma `main`).
*Critère de fin* : le schéma est documenté dans le README.

**T-07b — Ingestion des segment_efforts**
`GET /segments/{id}` ne donne que le PR agrégé, pas l'historique des
passages. Ingérer `GET /activities/{id}` en vue détaillée (tableau
`segment_efforts` embarqué) pour les activités qui recoupent mes
segments favoris, puis construire la table `segment_efforts`
(PK `id`, FK `segment_id`, FK `activity_id`) — reporté depuis T-07.
*Critère de fin* : une table `segment_efforts` en DuckDB, avec au moins
un effort par segment déjà roulé.

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

---

## Phase 10 — Fidélité du modèle

**T-32 — Cap réel par tronçon depuis le polyline**
Depuis T-27, le vent est projeté sur un seul cap moyen par segment
(`heading_rad`, calculé start->end à vol d'oiseau) : correct pour un
segment globalement rectiligne, faux pour un virage serré ou une
boucle (ex. HBFH, où start et end ne sont distants que de 124m pour
15km de tracé réel — le "cap" actuel n'y veut rien dire).
`segments.polyline` (déjà stocké tel quel, étape préparatoire de ce
ticket) contient le tracé
réel encodé (format Google Polyline). Le décoder en séquence de
points GPS, en tirer un cap par tronçon (réutiliser `chunk_segment`,
T-12, avec une altitude constante faute de profil d'altitude au
niveau segment), puis brancher ces tronçons dans `forecast_window.py`
(vent) au lieu de l'unique `SegmentChunk` actuel. Ne résout pas le
dénivelé (aucun profil d'altitude par segment disponible) — seulement
l'orientation face au vent.
*Critère de fin* : sur un segment en boucle ou tortueux, le vent
prédit n'est plus un seul vent de face/dos appliqué à toute la
distance, mais varie tronçon par tronçon comme en vrai.

**T-33 — Balayage "segments du jour" par orientation du vent**
Module indépendant, page Streamlit séparée (`pages/`) : pas "quelle est
la meilleure fenêtre pour CE segment sur 10 jours" (T-27), mais "parmi
TOUS mes segments favoris, lesquels ont le vent dans le bon sens
AUJOURD'HUI". Réutilise le découpage en tronçons de T-32
(`segment_chunks_from_polyline`) pour calculer, par segment, la
fraction de la DISTANCE (pas juste un compte de tronçons) où le vent
est de dos (`tailwind_fraction`) — un filtre géométrique pur, sans
aucune calibration CP/CdA/Crr. Segments mis en avant dans un tableau
quand au moins 3/4 de la distance est favorable.
*Critère de fin* : un tableau de tous les segments favoris, triés par
fraction de vent favorable décroissante, pour l'heure restante
d'aujourd'hui.

**T-34 — Vitesse de vent favorable réelle plutôt qu'une fraction**
`tailwind_fraction` (T-33) était binaire tronçon par tronçon : 2 km/h
et 20 km/h de vent de dos sur la même moitié d'un segment comptaient
pareil. Remplacée par `average_tailwind_speed_ms` : moyenne pondérée
par la longueur des tronçons de la vitesse de vent favorable (m/s,
signée — positive aide, négative freine), une vraie grandeur physique.
Le seuil arbitraire "3/4 de la distance" (T-33) devient "vitesse
moyenne > 0" dans `pages/1_Segments_du_jour.py`.
*Critère de fin* : la colonne "Vent favorable" affiche un km/h signé
(ex. "+17 km/h"), le tri et le seuil "favorable" utilisent cette même
valeur.

---

## Phase 11 — Migration Streamlit -> FastAPI + HTML/CSS/JS

Streamlit dessine tout côté Python : aucun contrôle sur le DOM, la mise
en page ou les micro-interactions, ce qui plafonne le rendu visuel très
en dessous d'une vraie web app "pro". Migration vers une API JSON
(FastAPI) consommée par une page HTML/CSS/JS statique (vanilla, sans
framework ni build step). Streamlit (`app.py`, `pages/`) reste en place
et fonctionnel jusqu'à T-40 inclus ; supprimé à T-41 une fois
l'équivalence fonctionnelle validée.

**T-36 — Squelette FastAPI + `GET /segments`**
Application FastAPI minimale (`src/segment_predictor/api/`), servie par
`uvicorn`. Un seul endpoint pour commencer : liste des segments
(id, nom, distance, D+) depuis DuckDB — assemble `storage/segments.py`,
aucune nouvelle logique.
*Critère de fin* : `uvicorn segment_predictor.api.main:app` répond en
JSON sur `/segments`.

**T-37 — Endpoint `POST /predict`**
Porte la logique de `app.py` (T-29) en backend pur : calibration
CP/CdA/Crr, classement des fenêtres sur 10 jours, pacing, comparaison
KOM/PR, incertitude. Découpé en sous-étapes si besoin (fenêtres
d'abord, pacing/KOM/PR/incertitude ensuite). Zéro rendu — uniquement
des structures sérialisées en JSON.
*Critère de fin* : la même réponse (aux arrondis d'affichage près)
qu'obtiendrait `app.py` pour un même segment/scénario/poids.

**T-38 — Endpoint `GET /wind-scan`**
Porte `pages/1_Segments_du_jour.py` (T-33/T-34) : scan des segments
favoris pour la météo du jour, sans calibration.
*Critère de fin* : équivalent JSON du tableau actuel de "Segments du
jour".

**T-39 — Page HTML/CSS/JS "Kompass"**
`web/index.html` + `app.js` + `style.css`, servis statiquement par
FastAPI (`StaticFiles`). Reproduit le flux de la page principale
(sélection segment/scénario/poids -> `fetch('/predict')` -> affichage)
sans polish visuel — on valide la mécanique avant le design.
*Critère de fin* : utilisable de bout en bout dans un navigateur, sans
Python visible côté rendu.

**T-40 — Design system**
Reprise visuelle complète une fois T-39 fonctionnel : typographie,
grille, cartes, palette, micro-animations. C'est cette étape qui
répond au "plus moderne, plus design, plus pro" — les précédentes ne
posent que la mécanique.
*Critère de fin* : rendu jugé "pro" par l'auteur, cohérent sur les deux
pages.

**T-41 — Page "Segments du jour" + suppression de Streamlit**
Deuxième page statique (même traitement que T-39/T-40) branchée sur
`/wind-scan`. Une fois les deux pages validées comme équivalentes à
Streamlit, suppression de `app.py`, `pages/`, `.streamlit/` et de la
dépendance `streamlit`.
*Critère de fin* : Streamlit n'est plus une dépendance du projet,
`uv run pytest` et `uv run ruff check .` passent toujours.

---

## Phase 12 — Courbe de puissance réelle, à côté du modèle CP+W'

**T-42 — Temps prédit à partir de la courbe MMP réellement mesurée**
Le temps prédit et la puissance requise (T-27/T-31) viennent tous deux
du modèle CP+W' (2 paramètres, lissé, extrapolable à toute durée mais
moins fidèle hors de sa plage calibrée). `compute_aggregate_mmp_curve`
(T-09) donne pourtant la courbe RÉELLEMENT mesurée (7 points, 3-20 min)
mais ne servait qu'à caler ce modèle, jamais à prédire directement.
Ajoute, à CÔTÉ du modèle (ne le remplace ni dans le classement des
créneaux ni dans `predicted_time_s`, décision explicite) : "si je
donnais vraiment ma meilleure puissance déjà atteinte pour cette durée,
avec le vent de cette fenêtre, quel temps ça donnerait ?"
- `interpolate_mmp_curve` (models/power.py) : lecture directe entre deux
  points mesurés (droite, pas un modèle ajusté) — ValueError explicite
  hors de la plage mesurée, pas d'extrapolation inventée.
- `simulate_segment_time_from_mmp_curve` (models/segment.py) : même
  boucle de convergence que `simulate_segment_time` (factorisée,
  `_simulate_time_with_power_curve`), mais la puissance vient de la
  courbe réelle. Amorçable par le temps déjà prédit par CP+W'
  (`initial_guess_s`) pour limiter le risque de sortir de la plage
  mesurée avant d'avoir convergé.
- `POST /predict` : nouveau champ `real_power_curve` (ou
  `real_power_curve_unavailable_reason` si hors plage), affiché en
  petit sous le héro.
*Critère de fin* : sur un segment dont le temps prédit tombe dans
[3, 20] min, un second chiffre "avec ta courbe mesurée" apparaît à côté
du temps prédit ; en dehors, un message explique pourquoi plutôt qu'un
silence ou un crash.

---

## Phase 13 — % d'alignement vent/tracé, et la page "Segments du jour" manquante

**T-43 — Top segments du jour avec % d'alignement, page web/segments-du-jour.html**
`average_tailwind_speed_ms` (T-34) donne une vraie vitesse (km/h) mais
pas d'intuition directe sur l'ORIENTATION seule. Ajoute
`average_wind_alignment_pct` (models/segment.py) : même pondération par
longueur, mais indépendant de la force du vent — +100% vent de dos pur,
-100% vent de face pur. Affiché à CÔTÉ du classement réel (km/h), ne le
remplace pas (T-34 avait déjà abandonné un % seul comme critère de tri).

Complète aussi la moitié "page" de T-41 (jamais faite) :
`web/segments-du-jour.html` + `.js`, même design system que T-40, top
10 cliquable (redirige vers `index.html?segment=<id>`, remplace le
`st.session_state` de T-35 par un paramètre d'URL — deux pages
statiques indépendantes n'ont pas de session partagée). La suppression
de Streamlit (l'autre moitié de T-41) reste à faire séparément.
*Critère de fin* : `GET /wind-scan` expose `wind_alignment_pct` ; la
page affiche le top 10 trié par vent favorable réel, avec le %
d'alignement en colonne, clic sur une ligne -> segment présélectionné
dans Kompass. Vérifié en vrai contre les 78 segments favoris.

---

## Phase 14 — Multi-utilisateur : connexion Strava pour d'autres personnes

Changement de nature du projet (CLAUDE.md disait "mono-utilisateur,
pas d'authentification, pas de multi-tenant" — à mettre à jour une fois
la phase terminée). Objectif à terme : ~100 utilisateurs, chacun
connecté à SON propre compte Strava, données isolées. Contraintes
identifiées avant de commencer : le quota Strava (100 req/15min,
1000/jour) est PAR APPLICATION, pas par utilisateur — partagé entre
tout le monde ; DuckDB n'accepte qu'un seul writer à la fois (déjà vu
en pratique, T-40) ; Strava impose l'affichage "Powered by Strava" pour
toute appli à plusieurs utilisateurs. Découpage validé avec l'auteur :

- **T-44a — table `users`** ✅ Un compte par athlète Strava connecté
  (`storage/users.py`) : id = l'id athlète Strava lui-même (pas d'id
  interne séparé), tokens, upsert (reconnexion/rafraîchissement ne
  duplique jamais un compte). Table VIVANTE (upserts au fil du temps),
  pas reconstruite depuis du Parquet comme le reste de storage/.
- **T-44b — `user_id` sur les tables personnelles** : `activities`,
  `streams`, `activity_weather`, `segment_efforts`, `wellness` (toutes
  déjà 100% propres à un athlète) + migration des données existantes
  (rattachées à l'auteur comme premier utilisateur).
- **T-44c — séparation de `segments`** : `segments` reste partagée
  (distance, tracé, KOM — des faits physiques identiques pour tout le
  monde) ; `pr_seconds`/`pr_date`/`effort_count`, aujourd'hui stockés à
  tort dans `segments` comme s'il n'y avait qu'un PR possible par
  segment, déménagent vers une nouvelle `user_segment_stats`
  (user_id, segment_id, ...) ; nouvelle `user_starred_segments`
  (user_id, segment_id) remplace l'hypothèse actuelle "la table
  `segments` = mes favoris".
- **T-44d — chemins bruts par utilisateur** : `data/raw/.../<user_id>/`
  plutôt qu'un dossier partagé — sinon la synchro d'un utilisateur
  écraserait les fichiers d'un autre.
- **T-44e — filtrage par utilisateur partout** : chaque requête de
  `storage/`, `calibrate/`, `predict/`, `api/` doit filtrer par
  utilisateur. Le plus gros morceau, le plus risqué (une requête
  oubliée = fuite de données entre utilisateurs).
- **T-45 — OAuth "Se connecter avec Strava"** : `/auth/strava/login` +
  `/auth/strava/callback`, session par cookie signé, bouton dans l'UI.
- **T-46 — Ingestion à la demande** : remplacer les scripts CLI
  mono-utilisateur par un flux déclenché après connexion.
- **T-47 — Hébergement** : l'app tourne aujourd'hui uniquement en local
  (127.0.0.1) — prérequis bloquant pour que quelqu'un d'autre puisse
  réellement s'y connecter (callback OAuth accessible depuis Internet).
