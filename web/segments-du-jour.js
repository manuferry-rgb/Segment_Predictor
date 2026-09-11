// T-41/T-43 : "Segments du jour" — parmi tes segments favoris, lesquels ont
// le vent dans le bon sens aujourd'hui. Page indépendante de index.html
// (comme pages/1_Segments_du_jour.py l'était de app.py côté Streamlit) :
// aucune calibration, juste la géométrie du tracé contre la météo du jour.

const scanButton = document.getElementById("scan-button");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

const TOP_N = 10;

// Dupliqué depuis app.js (même convention, T-27/T-32/T-33) plutôt
// qu'extrait dans un module partagé : une seule petite fonction, pas
// de justification à un système de modules ES pour ça sur un projet
// sans build step.
const COMPASS_LABELS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"];

function compassLabel(directionRad) {
  const degrees = ((directionRad * 180) / Math.PI + 360) % 360;
  const index = Math.round(degrees / 45) % COMPASS_LABELS.length;
  return COMPASS_LABELS[index];
}

function formatHour(isoString) {
  // Toujours aujourd'hui par construction (scan_segments_for_today, T-33) :
  // pas besoin d'afficher la date, juste l'heure — même choix que
  // pages/1_Segments_du_jour.py ("%Hh%M").
  return new Date(isoString).toLocaleTimeString("fr-FR", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderRows(opportunities) {
  return opportunities
    .slice(0, TOP_N)
    .map((o, index) => {
      const tailwindKmh = o.average_tailwind_speed_ms * 3.6;
      const windKmh = Math.round(o.wind_speed_ms * 3.6);
      return `<tr class="row-clickable" data-segment-id="${o.segment_id}">
        <td>${index + 1}</td>
        <td>${o.segment_name}</td>
        <td>${(o.distance_m / 1000).toFixed(1)} km</td>
        <td>${formatHour(o.best_hour)}</td>
        <td>${tailwindKmh >= 0 ? "+" : ""}${tailwindKmh.toFixed(0)} km/h</td>
        <td>${Math.round(o.wind_alignment_pct)}%</td>
        <td>${windKmh} km/h ${compassLabel(o.wind_direction_rad)}</td>
      </tr>`;
    })
    .join("");
}

function renderResults(opportunities) {
  if (opportunities.length === 0) {
    resultsEl.innerHTML = `<p class="card-note">
      Aucun créneau exploitable aujourd'hui (6h-21h, heures déjà passées exclues) sur aucun segment.
    </p>`;
    resultsEl.hidden = false;
    return;
  }

  // "Favorable" = vent de dos EN MOYENNE (positif) — même seuil naturel
  // que pages/1_Segments_du_jour.py (T-34), pas un pourcentage arbitraire.
  const nGood = opportunities.filter((o) => o.average_tailwind_speed_ms > 0).length;
  const shown = Math.min(TOP_N, opportunities.length);

  resultsEl.innerHTML = `
    <p class="card-note">
      <strong>${nGood}</strong> segment(s) sur <strong>${opportunities.length}</strong> avec un vent
      favorable en moyenne aujourd'hui — top ${shown} ci-dessous, classé par vent favorable réel
      (km/h). La colonne "Alignement" est un % pur d'orientation tracé/vent, indépendant de sa
      force : un vent fort mal aligné peut valoir plus qu'un vent faible parfaitement aligné,
      d'où deux colonnes plutôt qu'une seule.
    </p>
    <article class="card">
      <h2>Top ${shown} — clique une ligne pour ouvrir ce segment dans Kompass</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Segment</th><th>Distance</th><th>Meilleure heure</th>
              <th>Vent favorable</th><th>Alignement</th><th>Vent</th>
            </tr>
          </thead>
          <tbody>${renderRows(opportunities)}</tbody>
        </table>
      </div>
    </article>
  `;
  resultsEl.hidden = false;

  for (const row of resultsEl.querySelectorAll("tr[data-segment-id]")) {
    row.addEventListener("click", () => {
      window.location.href = `index.html?segment=${row.dataset.segmentId}`;
    });
  }
}

scanButton.addEventListener("click", async () => {
  statusEl.textContent = "Récupération de la météo du jour pour chaque segment favori...";
  resultsEl.hidden = true;
  scanButton.disabled = true;

  try {
    const response = await fetch("/wind-scan");
    if (!response.ok) {
      const error = await response.json();
      statusEl.textContent = `Erreur : ${error.detail}`;
      return;
    }
    const opportunities = await response.json();
    // Déjà trié par vent favorable décroissant (scan_segments_for_today,
    // T-34) : pas de tri côté client, on affiche tel quel.
    statusEl.textContent = "";
    renderResults(opportunities);
  } catch (err) {
    statusEl.textContent = `Erreur réseau : ${err.message}`;
  } finally {
    scanButton.disabled = false;
  }
});
