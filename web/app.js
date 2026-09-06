// T-39 : squelette mécanique du frontend — aucun style poussé ici (T-40).
// Différence principale avec app.py (Streamlit) : ici c'est CE fichier,
// exécuté dans le navigateur, qui reconstruit le HTML des résultats à
// partir du JSON reçu — Python ne dessine plus rien, il ne fait que
// répondre à des requêtes HTTP (voir api/main.py).

const segmentSelect = document.getElementById("segment-select");
const draftSelect = document.getElementById("draft-select");
const massInput = document.getElementById("mass-input");
const form = document.getElementById("predict-form");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

// Même convention que api/main.py / physics.py : direction D'OÙ VIENT
// le vent, Nord -> Nord-Est -> ... -> Nord-Ouest.
const COMPASS_LABELS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"];

function compassLabel(directionRad) {
  const degrees = ((directionRad * 180) / Math.PI + 360) % 360;
  const index = Math.round(degrees / 45) % COMPASS_LABELS.length;
  return COMPASS_LABELS[index];
}

function formatMmSs(seconds) {
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${minutes}:${String(secs).padStart(2, "0")}`;
}

function formatDayHour(isoString) {
  // toLocaleString plutôt qu'un tableau de jours en dur (contrairement à
  // _format_day_hour dans app.py) : le navigateur connaît déjà les noms
  // de jours en français, pas la peine de les redupliquer ici.
  return new Date(isoString).toLocaleString("fr-FR", {
    weekday: "long",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function loadSegments() {
  const response = await fetch("/segments");
  const segments = await response.json();
  for (const segment of segments) {
    const option = document.createElement("option");
    option.value = segment.id;
    const km = (segment.distance_m / 1000).toFixed(1);
    option.textContent = `${segment.name} (${km} km, D+ ${Math.round(segment.elevation_gain_m)} m)`;
    segmentSelect.appendChild(option);
  }
}

function renderWindowRows(windows) {
  return windows
    .map((w) => {
      const windKmh = Math.round(w.wind_speed_ms * 3.6);
      const tempC = Math.round(w.temperature_k - 273.15);
      return `<tr>
        <td>${formatDayHour(w.time)}</td>
        <td>${formatMmSs(w.predicted_time_s)}</td>
        <td>${Math.round(w.required_power_w)} W</td>
        <td>${windKmh} km/h ${compassLabel(w.wind_direction_rad)}</td>
        <td>${tempC}°C</td>
      </tr>`;
    })
    .join("");
}

function renderResults(data) {
  const best = data.windows[0];
  const windKmh = Math.round(best.wind_speed_ms * 3.6);
  const tempC = Math.round(best.temperature_k - 273.15);

  const uncertaintyHtml = data.uncertainty
    ? `<p>Avec incertitude (CP, forme, vent, ${data.uncertainty.n_samples} tirages) : ` +
      `<strong>${formatMmSs(data.uncertainty.mean_time_s)} ± ${data.uncertainty.std_time_s.toFixed(0)}s</strong></p>`
    : `<p>Pas assez d'efforts récents proches du maximum pour estimer l'incertitude de forme.</p>`;

  const prHtml = data.pr
    ? `<p>Mon PR : ${formatMmSs(data.pr.seconds)}` +
      (data.pr.effort
        ? ` — ${data.pr.effort.average_watts ?? "puissance non disponible"} W` +
          (data.pr.effort.draft_status !== "solo"
            ? ` (⚠️ statut draft : ${data.pr.effort.draft_status})`
            : "")
        : "") +
      `</p>`
    : `<p>Mon PR : jamais roulé ce segment</p>`;

  resultsEl.innerHTML = `
    <h2>Meilleure fenêtre : ${formatDayHour(best.time)}</h2>
    <p>Temps prédit : <strong>${formatMmSs(best.predicted_time_s)}</strong>
       (puissance requise, vent inclus : ${Math.round(best.required_power_w)} W)</p>
    <p>Vent : ${windKmh} km/h du ${compassLabel(best.wind_direction_rad)} · Température : ${tempC}°C</p>
    <p>CP=${data.calibration.cp_watts.toFixed(0)}±${data.calibration.cp_watts_std.toFixed(0)} W ·
       CdA=${data.calibration.cda_m2.toFixed(3)} m² · Crr=${data.calibration.crr.toFixed(4)}</p>

    <h3>Stratégie de pacing (sans vent)</h3>
    <p>${Math.round(data.pacing.power_w)} W</p>

    <h3>KOM du segment</h3>
    <p>${formatMmSs(data.kom.seconds)} — puissance estimée pour toi : ${Math.round(data.kom.power_w)} W
       ${data.kom.power_w_extrapolated ? "⚠️ hors de la plage calibrée du modèle CP" : ""}</p>

    ${prHtml}
    ${uncertaintyHtml}

    <h3>Classement des créneaux</h3>
    <table>
      <thead>
        <tr><th>Créneau</th><th>Temps</th><th>Puissance</th><th>Vent</th><th>Température</th></tr>
      </thead>
      <tbody>${renderWindowRows(data.windows)}</tbody>
    </table>
  `;
  resultsEl.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusEl.textContent = "Calibration et prévision météo...";
  resultsEl.hidden = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        segment_id: Number(segmentSelect.value),
        draft_preset: draftSelect.value,
        mass_kg: Number(massInput.value),
      }),
    });
    if (!response.ok) {
      const error = await response.json();
      statusEl.textContent = `Erreur : ${error.detail}`;
      return;
    }
    statusEl.textContent = "";
    renderResults(await response.json());
  } catch (err) {
    statusEl.textContent = `Erreur réseau : ${err.message}`;
  }
});

loadSegments();
