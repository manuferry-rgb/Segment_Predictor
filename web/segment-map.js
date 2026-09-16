// T-52 : carte 3D du segment (MapLibre GL JS + tuiles-terrain MapTiler).
// Fichier séparé de app.js : cette logique (config MapLibre, relief 3D)
// n'a rien à voir avec le flux prédiction/formulaire — chargé après
// maplibre-gl (CDN, index.html) et après app.js (utilise segmentSelect).
//
// /config expose MAPTILER_API_KEY (contrairement aux secrets Strava, une
// clé MapTiler est FAITE pour voyager jusqu'au navigateur — voir
// api/main.py). Si elle est absente (MapTiler pas configuré), la carte
// est simplement omise plutôt que de planter le reste de la page.

let currentMap = null;

async function renderSegmentMap(segmentId) {
  const container = document.getElementById("segment-map");
  if (!container) return;

  // Changement de segment : MapLibre ne réutilise pas proprement un
  // conteneur déjà initialisé, on repart d'une carte neuve à chaque fois.
  if (currentMap) {
    currentMap.remove();
    currentMap = null;
  }

  let apiKey;
  try {
    const configResponse = await fetch("/config");
    apiKey = (await configResponse.json()).maptiler_api_key;
  } catch {
    apiKey = null;
  }
  if (!apiKey) {
    container.innerHTML = '<p class="card-note">Carte indisponible (MapTiler non configuré).</p>';
    return;
  }

  let points;
  try {
    const geometryResponse = await fetch(`/segments/${segmentId}/geometry`);
    if (!geometryResponse.ok) throw new Error("géométrie indisponible");
    points = (await geometryResponse.json()).points;
  } catch {
    container.innerHTML = '<p class="card-note">Tracé du segment indisponible.</p>';
    return;
  }
  if (points.length < 2) {
    container.innerHTML = '<p class="card-note">Tracé trop court pour être affiché.</p>';
    return;
  }

  // GeoJSON attend [lng, lat] ; decode_polyline (Google/Strava, côté
  // serveur) renvoie (lat, lng) — inversion nécessaire ici, pas plus tôt.
  const coordinates = points.map(([lat, lng]) => [lng, lat]);

  currentMap = new maplibregl.Map({
    container: "segment-map",
    style: `https://api.maptiler.com/maps/outdoor-v2/style.json?key=${apiKey}`,
    // pitch à 0 ici (pas 60) : cadrer les bornes ET incliner la caméra en
    // une seule étape donne un zoom aberrant (constaté en vrai, T-52) —
    // fitBounds calcule mal l'échelle quand la caméra est déjà penchée.
    // L'inclinaison est appliquée séparément APRÈS le cadrage, plus bas.
    attributionControl: false,
  });
  currentMap.addControl(new maplibregl.NavigationControl(), "top-right");

  currentMap.on("load", () => {
    // Tuiles-terrain (élévation), séparées du fond de carte lui-même —
    // setTerrain est ce qui donne le relief 3D réel, pas juste un
    // ombrage 2D peint sur la texture.
    currentMap.addSource("segment-terrain", {
      type: "raster-dem",
      url: `https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json?key=${apiKey}`,
      tileSize: 256,
    });
    currentMap.setTerrain({ source: "segment-terrain", exaggeration: 1.4 });

    currentMap.addSource("segment-route", {
      type: "geojson",
      data: { type: "Feature", geometry: { type: "LineString", coordinates } },
    });
    currentMap.addLayer({
      id: "segment-route-line",
      type: "line",
      source: "segment-route",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#c9862f", "line-width": 4 },
    });

    const bounds = coordinates.reduce(
      (b, coord) => b.extend(coord),
      new maplibregl.LngLatBounds(coordinates[0], coordinates[0])
    );
    // "idle" (pas tout de suite après setTerrain) : bug connu MapLibre —
    // positionner la caméra avant que les tuiles d'élévation terrain
    // aient fini de charger corrompt le centre de la carte en NaN, et ça
    // reste cassé pour toute interaction suivante (zoom, molette...).
    // Trouvé en testant en vrai (T-52) : les contrôles s'affichaient mais
    // la carte restait blanche, chaque clic relançait l'erreur.
    currentMap.once("idle", () => {
      // Cadrage à plat d'abord (pas de `pitch` ici, voir le commentaire du
      // constructeur plus haut) — fitBounds calcule le zoom correctement
      // seulement à plat. Une fois posé, `once("moveend", ...)` incline la
      // caméra à 60° séparément ("relief inclinable" demandé), sans
      // perturber le calcul de zoom déjà fait.
      currentMap.fitBounds(bounds, { padding: 40, duration: 0 });
      currentMap.once("moveend", () => {
        currentMap.easeTo({ pitch: 60, duration: 600 });
      });
    });
  });
}
