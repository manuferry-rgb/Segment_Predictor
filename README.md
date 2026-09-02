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
| name, distance_m, average_grade | | |
| kom_seconds | BIGINT | parsé depuis `xoms.kom` ("mm:ss" / "h:mm:ss") |
| pr_seconds, pr_date | BIGINT, VARCHAR | NULL si le segment n'a jamais été roulé (`effort_count = 0`) |
| effort_count | BIGINT | nombre de mes passages sur ce segment |

### `segment_efforts` — pas encore construite

Nécessite l'historique individuel de mes passages sur un segment, que
`GET /segments/{id}` ne fournit pas (seulement le PR agrégé). Ça
demande un nouvel ingest (`GET /activities/{id}` en vue détaillée, qui
embarque un tableau `segment_efforts` par activité) — voir T-07b dans
`ROADMAP.md`.

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
