# Kompass — prédicteur de temps sur segment Strava

Voir `CLAUDE.md` pour le contexte du projet, la stack et les conventions.

## Schéma de données (T-07)

Toutes les données vivent dans `data/` (gitignoré). Le Parquet brut
écrit par `ingest/` (`data/raw/strava_*/`) est la seule source de
vérité ; le fichier DuckDB (`data/segment_predictor.duckdb`) n'est
qu'une projection reconstructible — `uv run python scripts/build_database.py`
le refait entièrement à partir du Parquet, sans appel réseau.

Deux schémas SQL, pas juste des tables :

- **`raw`** — des **vues** (pas des tables), passthrough sur
  `read_parquet(...)` : aucune donnée dupliquée, aucun cast, toujours à
  jour. Sert à explorer le JSON brut en SQL (`SELECT * FROM raw.segments`).
- **`main`** (schéma par défaut) — les tables **transformées**, typées,
  utilisées par le reste du projet.

### `main.activities`

PK `id`. Sélection curée d'un sous-ensemble des ~59 champs bruts de
Strava — le reste reste consultable via `raw.activities`.

| colonne | type | note |
|---|---|---|
| id | BIGINT | PK |
| name, type, sport_type | VARCHAR | |
| start_date | TIMESTAMP | UTC, naïf (Strava ne fournit que de l'UTC pour ce champ) |
| distance_m | DOUBLE | |
| moving_time_s, elapsed_time_s | BIGINT | |
| total_elevation_gain_m | DOUBLE | |
| average_watts, device_watts | DOUBLE, BOOLEAN | NULL si pas de capteur (ex. randonnée) |
| average_heartrate, max_heartrate | DOUBLE | NULL si pas de ceinture cardio |
| average_cadence | DOUBLE | NULL si pas de capteur |

### `main.streams`

PK `(activity_id, sample_index)`, FK `activity_id → activities.id`.
**Format long** (1 ligne = 1 échantillon temporel), obtenu par
dépivotage du brut (1 ligne = 1 activité, colonnes-listes). Hypothèse
vérifiée sur les 497 activités réelles : tous les streams présents pour
une activité ont la même longueur que `time` — sinon la construction
lève plutôt que de décaler les points silencieusement.

| colonne | type | note |
|---|---|---|
| activity_id | BIGINT | FK -> activities.id |
| sample_index | BIGINT | position dans la série, 0-indexé |
| t_s | BIGINT | secondes depuis le début de l'activité |
| watts, altitude_m, distance_m, heartrate, cadence | DOUBLE | NULL si le stream n'existe pas pour cette activité |
| lat, lng | DOUBLE | éclatés depuis `latlng` (paires `[lat, lng]`), NULL si absent |

### `main.segments`

PK `id`.

| colonne | type | note |
|---|---|---|
| id | BIGINT | PK |
| name, distance_m | | |
| average_grade | DOUBLE | **fraction** rise/run (0.002 = 0.2%), pas le pourcent brut de Strava — converti ici (÷100) pour rester cohérent avec `grade` dans `models/physics.py` depuis T-10. Bug réel trouvé tardivement (T-16, en branchant enfin cette colonne sur un temps prédit) : resté non converti, un 0.2% était traité comme 20% par la physique |
| kom_seconds | BIGINT | parsé depuis `xoms.kom` ("mm:ss" / "h:mm:ss") |
| pr_seconds, pr_date | BIGINT, VARCHAR | NULL si le segment n'a jamais été roulé (`effort_count = 0`) |
| effort_count | BIGINT | nombre de mes passages sur ce segment |

### `main.segment_efforts` (T-07b)

PK `id` (id d'effort Strava), FK `segment_id → segments.id`, FK
`activity_id → activities.id`. Tous les efforts trouvés dans les
activités détaillées (`GET /activities/{id}`, `ingest/strava_activity_
details.py`) — Strava matche automatiquement tous les segments publics
croisés, pas seulement ceux suivis dans `main.segments` ; c'est T-16 qui
filtre aux segments suivis via la jointure.

| colonne | type | note |
|---|---|---|
| id | BIGINT | PK |
| segment_id, activity_id | BIGINT | FK |
| start_date | TIMESTAMP | UTC, naïf, même convention que `activities.start_date` |
| elapsed_time_s, moving_time_s | BIGINT | |
| distance_m | DOUBLE | |
| average_watts, average_heartrate | DOUBLE | NULL si pas de capteur ce jour-là |
| device_watts | BOOLEAN | |
| pr_rank, kom_rank | BIGINT | NULL si l'effort n'est ni un PR ni un KOM |

**Limite de couverture connue** : seules les activités avec capteur de
puissance (même périmètre que `main.streams`, T-05 — 497/735 sorties
vélo) ont leur vue détaillée récupérée. Sur le segment 7722237
(`effort_count=16`), seuls 4 efforts sont couverts ici — les 12 autres
ont eu lieu sur des sorties sans capteur, invisibles à cette ingestion.
Assumé : un effort sans puissance ne sert de toute façon à rien pour la
calibration (T-17), pas la peine d'étendre l'ingest pour les couvrir.

### `main.activity_weather` (T-14)

PK/FK `activity_id → activities.id`. Une ligne par sortie vélo
(`Ride`/`VirtualRide` uniquement — hors périmètre pour la marche/course)
géolocalisée dont la zone météo a été téléchargée. Météo Open-Meteo
(archive, gratuite, sans clé) interpolée linéairement entre les deux
relevés horaires encadrant le `start_date` exact de l'activité.

| colonne | type | note |
|---|---|---|
| activity_id | BIGINT | PK, FK -> activities.id |
| temperature_k | DOUBLE | converti depuis °C |
| relative_humidity_pct | DOUBLE | 0-100 |
| pressure_pa | DOUBLE | converti depuis hPa |
| wind_speed_ms | DOUBLE | converti depuis km/h |
| wind_direction_rad | DOUBLE | 0 = nord, croît vers l'est — même convention que `heading_rad` (T-12). Interpolé en composantes (cos, sin), pas en degrés bruts (l'angle traverse 0°/360°) |

Ingest (`ingest/open_meteo.py`) : un Parquet par **zone géographique**
(grille 0.1° ≈ 11km, arrondi à la résolution native des données de
réanalyse Open-Meteo), pas par activité — `compute_weather_zones`
regroupe les activités par zone et calcule la plage de dates à couvrir.
Sur les 699 sorties vélo géolocalisées du jeu de données actuel, ça
donne 133 zones (le vélo se pratique sur des lieux plus dispersés que
prévu au départ — voyages, essais de nouveaux itinéraires — d'où
133 plutôt que les ~68 estimés en ne comptant que les zones les plus
fréquentées). Une zone déjà téléchargée n'est jamais redemandée, y
compris si de nouvelles activités dans cette zone tombent hors de sa
plage de dates déjà couverte (il faut supprimer le fichier pour forcer
un refetch).

**Limitation connue, non corrigée** : le vent Open-Meteo (`wind_speed_10m`)
est mesuré à 10m de hauteur, pas au niveau du cycliste. Le vent réel
ressenti près du sol est généralement plus faible (frottement de
surface, effet de couche limite) — utiliser `wind_speed_ms` tel quel
tend à **surestimer** le vent effectif. Une correction nécessiterait un
profil vertical du vent (loi logarithmique ou en puissance), pas fait
ici : le vent est simplement absent des calculs de puissance jusqu'à
T-15/T-19.

## Tri des efforts en groupe (T-16)

`annotations/draft_status.csv` (versionné dans Git, contrairement à
`data/` — c'est une donnée d'entrée du projet, pas un artefact
régénérable à l'identique) liste tous mes efforts sur les segments
suivis (`main.segments`), avec un temps prédit et un écart, pour aider
à repérer ceux à tagger `solo`/`drafted` avant la calibration CdA/Crr
(T-17, qui doit exclure les efforts en groupe).

`uv run python scripts/generate_draft_tagging_csv.py` régénère le
fichier. **Seule la colonne `draft_status` est préservée** d'un
lancement à l'autre (fusion par `effort_id`) — `predicted_time_s` et
`gap_s` sont toujours recalculés depuis "le modèle actuel" (CP/W'
réajustés à chaque appel sur tout l'historique). Rien n'est jamais
supprimé, y compris si un effort disparaît de la source. Remplis
`draft_status` (`solo`/`drafted`, par défaut `unknown`) à la main, puis
commit.

`gap_s = predicted_time_s - actual_time_s` : positif = tu es allé
**plus vite** que la prédiction, le signal à regarder en premier. Trié
par écart décroissant.

**⚠️ Le temps prédit n'est PAS une détection de drafting, juste une aide
au tri** :
- Le modèle physique n'est pas calibré : CdA=0.32 m² et Crr=0.005 sont
  des valeurs génériques de littérature, pas les tiennes (c'est
  justement ce que T-17 calibrera, à partir des efforts tagués `solo`
  ici — la boucle est assumée).
- Le segment est traité comme **un seul tronçon à pente moyenne
  constante** (`main.segments.average_grade`), pas son profil réel
  tronçon par tronçon — on n'a pas de quoi le reconstruire par effort
  (`start_index`/`end_index` non extraits en T-07b).
- Sans vent (T-15 existe mais n'est pas branché ici) ni draft (T-19).

Un grand écart peut donc venir d'un vrai drafting, d'un bon jour, ou
juste des approximations ci-dessus — à vérifier au cas par cas, pas à
prendre pour argent comptant.

## Calibration CdA/Crr (T-17)

`calibrate_cda_crr_from_db` (`calibrate/cda_crr.py`) ajuste CdA et Crr
en minimisant l'écart relatif entre `simulate_segment_time` (T-13) et le
temps réel, sur les efforts tagués `solo` dans
`annotations/draft_status.csv`. Deux filtres, trouvés nécessaires en
conditions réelles (320 efforts solo bruts), en plus du tri manuel
solo/drafted :

- **Plage de validité du modèle CP** (`cp_fit.duration_range_s`,
  180–1200s) : sous 180s, `CP + W'/T` diverge vers l'infini quand T→0
  (déjà documenté en T-09) et surestime largement la puissance
  soutenable — inclure ces efforts ne calibre pas un mauvais CdA/Crr, ça
  fait plafonner l'optimiseur à ses bornes pour compenser un biais de
  durée qu'aucun CdA/Crr ne peut corriger.
- **`pr_rank` non nul** : le modèle prédit le temps *atteignable en
  effort maximal*. Un effort solo à rythme d'entraînement (la majorité
  des 320, une fois le filtre durée appliqué) n'a aucune raison de s'en
  approcher. `pr_rank` (position au classement perso Strava sur ce
  segment) sert de proxy simple pour "effort quasi maximal" — une
  approximation documentée, pas une vraie mesure d'intensité (VO2/FTP) :
  un effort peut être un record perso sans être poussé à bloc, et
  inversement sur un segment rarement emprunté.

Dernier résultat officiel (mass=91kg, 2026) : **CdA=0.441 m², Crr=0.0105,
RMSE relative=8.7%, n=77** (189 exclus hors plage durée, 54 exclus faute
de `pr_rank`). Crr converge à 0.0105 quelle que soit la largeur des
bornes testées (vérifié jusqu'à 0.03) — ce n'est pas un plafonnement
artificiel, `DEFAULT_CRR_BOUNDS` a été élargi à `(0.002, 0.012)` en
conséquence pour ne plus friser sa borne. Cette valeur reste un peu
élevée pour du bitume lisse (0.003–0.005 typique) : plausiblement le
revêtement réel de mes segments, et/ou un résidu de vent absorbé par Crr
plutôt que CdA — **le vent n'est pas encore branché dans cette
calibration** (T-15 existe, pas intégré ici ; piste pour T-19).

Avec seulement 77 points malgré 9267 `segment_efforts` en base, ce
calibrage reste basé sur peu de données — à reconsidérer si le tag
manuel s'étoffe, ou si le vent est intégré et change le résidu.

## Backtest (T-18)

`backtest_cda_crr_from_db` (`calibrate/backtest.py`) évalue la
calibration ci-dessus honnêtement : split **temporel** (pas aléatoire)
des mêmes 77 efforts filtrés — les plus anciens calibrent CdA/Crr, les
plus récents évaluent la prédiction. Un split aléatoire mélangerait
passé et futur dans le train et donnerait une erreur artificiellement
optimiste, puisque le modèle "connaîtrait" déjà des efforts postérieurs
à ceux qu'il prédit — pas une situation réaliste (prédire une fenêtre à
venir, T-14).

`uv run python scripts/backtest_cda_crr.py` régénère le résultat et le
graphique.

Dernier résultat officiel (mass=91kg, 2026, split 80/20 par défaut) :
**erreur absolue médiane = 14.9 s sur 15 efforts de test** (train : 62).
Le nuage de points prédit vs réel (`data/backtest_t18.png`, gitignoré —
artefact dérivé, pas une donnée d'entrée) suit bien la diagonale y=x sur
toute la plage observée (~200s à ~790s), sans dérive systématique
visible à l'œil (le modèle ne surestime ni ne sous-estime plus sur les
longs efforts que sur les courts).

15 points de test reste peu pour juger de la robustesse dans le temps —
comme pour T-17, à reconsidérer avec plus de tags `solo`/`pr_rank`
disponibles.

## Scénarios de draft (T-19/T-20)

`models/draft.py` calcule un ratio de réduction du CdA solo pour 4
scénarios (`draft_ratio_for_preset`) : `solo` (1.0, pas de draft),
`roue_collee` et `un_metre` via la formule d'Olds (1995, paceline à un
seul leader), `groupe` via Blocken et al. (2018, peloton — physique
différente, blindage multi-coureurs, pas dérivée d'Olds). Détail des
sources et de la plage de validité dans le docstring du module.

`predict/scenarios.py` (`compare_draft_scenarios_for_segment`) simule un
segment sous les 4 scénarios et calcule le gain de chacun par rapport au
solo. `uv run python scripts/compare_draft_scenarios.py <segment_id>`
affiche le tableau comparatif — le critère de fin de T-20.

Le contraste attendu (gain fort à plat, quasi nul en forte pente) se
confirme sur de vrais segments : `groupe` donne 47% de gain sur un
segment quasi plat (0.1%) contre 9% sur une montée à 6.7%.

## Charge d'entraînement (T-21)

`models/power.py` calcule la puissance normalisée (`normalized_power`,
algorithme de Coggan : moyenne glissante 30s, puissance 4, racine 4e) et
le TSS (`training_stress_score`) par activité, avec le CP calibré (T-16)
comme proxy de FTP — approximation documentée dans le docstring (pas de
tests FTP historiques datés pour faire autrement).

`models/training_load.py` implémente le modèle de Banister : CTL
(forme, constante de temps 42j), ATL (fatigue, 7j), TSB = CTL - ATL de
la veille (forme avec laquelle on aborde la séance du jour, pas
influencée par elle). `calibrate/training_load.py` agrège le TSS réel
jour par jour depuis `main.activities`/`main.streams` (plusieurs
sorties le même jour s'additionnent, un jour sans sortie compte 0 et
décharge quand même la forme) et fait tourner la récursion sur tout
l'historique.

`uv run python scripts/plot_training_load.py` génère les courbes
(`data/training_load_t21.png`, gitignoré). Critère de fin du ticket
("courbes cohérentes avec mon vécu") validé à l'œil sur l'historique
réel (2021-2026) : quasi rien avant fin 2023 (peu de sorties à capteur
de puissance sur cette période), un rythme annuel de blocs/récup net
ensuite, et un gros creux de TSB (~-45) suivi d'un rebond marqué
mi-2026, reconnu comme cohérent.

## Wellness intervals.icu (T-22)

`ingest/intervals_icu.py` récupère HRV, sommeil, FC de repos et poids
via l'API intervals.icu (Basic Auth : utilisateur `API_KEY`, mot de
passe la clé générée dans Réglages > Developer Settings — pas d'OAuth
contrairement à Strava). Un seul appel couvre toute la plage de dates
(pas de pagination), et le fichier Parquet est **écrasé** à chaque
fetch plutôt qu'accumulé : le wellness peut être corrigé
rétroactivement (ex. saisie HRV en retard), donc un refetch complet
reste plus correct qu'un "depuis la dernière fois".

`storage/wellness.py` construit `main.wellness` (date, hrv, sleep_s,
resting_heart_rate_bpm, weight_kg). `weight_kg` est vérifié en kg contre
une vraie valeur du compte (83.0 — 83 lb serait beaucoup trop léger pour
un cycliste adulte), pas juste supposé. `hrv`/`sleep_s` restent en
revanche **non vérifiés** : jamais rencontrés renseignés dans
l'historique réel disponible pour confirmer leur unité au-delà du nom
du champ intervals.icu lui-même.

`uv run python scripts/fetch_wellness.py` (prérequis :
`INTERVALS_ICU_ATHLETE_ID` et `INTERVALS_ICU_API_KEY` dans `.env`).

Critère de fin du ticket (table alimentée) atteint : **392 lignes**
réelles (2025-08-07 → 2026-09-02). La couverture reste creuse
honnêtement — 0 valeur `hrv`, 0 `sleep_s`, 49 jours de
`resting_heart_rate_bpm` (constant à 55), 1 seule entrée `weight_kg`.
Pas un bug : reflète l'usage actuel du compte intervals.icu (pas
d'appareil connecté pour HRV/sommeil pour l'instant). À reconsidérer
pour T-24 (modèle de forme), qui devra composer avec des features très
manquantes.

## Indice de performance (T-23)

`models/form.py` : `performance_index(actual_power_w, cp_fit, duration_s)`
= puissance réelle / (CP + W'/durée) — même plage de validité que le
modèle CP (T-09), `ValueError` explicite en dehors.

Le filtre "effort maximal" (demandé et documenté par le ticket) :
`is_near_maximal_effort` compare la MMP de l'effort au **record glissant
à cette date** (le meilleur jamais vu jusque-là, pas le record absolu
toutes dates confondues), seuil à 95%. Comparer au record absolu aurait
fait disparaître rétroactivement de vrais efforts à bloc anciens dès
qu'un record plus récent les dépasse — biais discuté et écarté avant de
coder.

`calibrate/form.py` (`compute_performance_index_series`) parcourt les
activités par date croissante, calcule la MMP à chaque durée de
référence (mêmes durées que le fit CP, T-16), tient le record glissant à
jour au fil de l'eau, et calcule l'indice pour les efforts qui passent le
filtre. `uv run python scripts/plot_performance_index.py` sort la série
(critère de fin du ticket).

Résultat réel (2021-2026) : 225 points sur 7 durées, indice moyen 0.87.
Tendance nette de ~0.5-0.7 en 2021 vers ~0.95-1.0 en 2025-2026 —
cohérent avec le choix (déjà documenté ailleurs, T-17/T-21) d'utiliser
le CP *actuel*, pas un CP recalculé à chaque date : les anciens "records
du moment" paraissent logiquement loin du plafond de forme le plus
récent, ce n'est pas un signe de mauvaise forme passée mais l'effet
mécanique de comparer à un plafond qui n'existait pas encore.

## Modèle de forme (T-24)

`models/form_regression.py` : Ridge (`fit_ridge`/`predict_ridge`, solution
fermée β = (XᵀX + alpha·I)⁻¹Xᵀy sur features standardisées) et R²
(`r_squared`), implémentés à la main avec numpy/scipy plutôt qu'avec
scikit-learn — pas de nouvelle dépendance pour une résolution d'algèbre
linéaire de quelques lignes, choix fait avec toi avant de coder.
`temporal_cross_validate_ridge` découpe en blocs chronologiques
successifs (comme T-18) : chaque fold entraîne sur le passé, teste sur
un futur qu'il n'a jamais vu.

`calibrate/form.py` (`build_form_regression_dataset`) joint l'indice de
performance (T-23) au CTL/ATL/TSB de la même date exacte (T-21), plus
`duration_s` comme 4e feature (absorbe le biais résiduel du modèle CP
selon la durée, visible dans le graphique T-23). Les 4 champs wellness
(T-22) sont **volontairement exclus** : quasi tous vides en pratique
(0 `hrv`, 0 `sleep_s`, `resting_heart_rate_bpm` constant, 1 seule
`weight_kg` — voir T-22), les inclure aurait demandé une imputation
inventant des données absentes.

`uv run python scripts/fit_form_model.py` — critère de fin du ticket.

**Résultat réel, honnête : R² moyen en validation croisée temporelle =
-3.47** (5 folds, de -0.49 à -5.36) — nettement pire que de prédire la
moyenne d'entraînement. Diagnostiqué, pas juste rapporté tel quel : le
R² **en échantillon** (train = test, tout l'historique) est correct
(0.70), donc CTL/ATL/TSB/durée portent un vrai signal — mais l'indice de
performance dérive fortement par année (moyenne 0.62 en 2021, 0.81 en
2023, 0.90 en 2024, 0.96 en 2025-2026, cf T-23) et CTL/ATL/TSB
n'encodent pas "combien de temps s'est écoulé". Un modèle entraîné sur
les années anciennes (indice bas) sous-estime donc systématiquement les
années récentes (indice haut) — exactement ce qu'une validation croisée
*temporelle* est censée révéler, contrairement à une validation
aléatoire qui aurait caché ce problème. Piste pour plus tard, pas
traitée ici : la dérive vient surtout du choix d'un CP fixe (T-17/T-21)
comme référence de `performance_index`, pas forcément d'une vraie
absence de lien forme/performance.

### Limites connues

- Une seule activité (`10066651328`) a 63 échantillons `heartrate = -1`
  sur ses ~2,15M au total, dans `main.streams` — un décrochage capteur
  ponctuel, pas nettoyé ici : le nettoyage/filtrage de données est un
  problème de couche `models`/`calibrate`, pas de stockage.
- `main.activities` ne recopie qu'un sous-ensemble des champs bruts.
  Le reste (kudos, athlète, polyline de la carte...) reste accessible
  via `raw.activities` si besoin.
- Le vent Open-Meteo est mesuré à 10m, pas au niveau du cycliste (détail
  ci-dessus, section `main.activity_weather`).
