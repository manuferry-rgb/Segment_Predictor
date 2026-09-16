// T-39/T-40 : squelette mécanique + rendu visuel (direction validée par
// maquette, cf ROADMAP.md T-40). Différence principale avec app.py
// (Streamlit) : ici c'est CE fichier, exécuté dans le navigateur, qui
// reconstruit le HTML des résultats à partir du JSON reçu — Python ne
// dessine plus rien, il ne fait que répondre à des requêtes HTTP
// (voir api/main.py).

const segmentSelect = document.getElementById("segment-select");
const draftSelect = document.getElementById("draft-select");
const massInput = document.getElementById("mass-input");
const form = document.getElementById("predict-form");
const submitButton = form.querySelector("button[type=submit]");
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

// Rotation CSS (deg) pour l'aiguille de la boussole — même angle que
// compassLabel, juste sans l'arrondi aux 8 points cardinaux : le
// dessin peut être précis là où le libellé texte doit rester lisible.
function compassRotationDeg(directionRad) {
  return ((directionRad * 180) / Math.PI + 360) % 360;
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

// Zones d'effort (% de CP) pour les pastilles de couleur — calquées sur
// les zones Coggan usuelles en cyclisme. Approximation ASSUMÉE et
// documentée (cf web/style.css, même docstring) : CP (modèle du
// projet) sert ici de référence de seuil, pas une vraie mesure de FTP.
const POWER_ZONES = [
  { max: 0.75, label: "Endurance", className: "zone-endurance" },
  { max: 0.9, label: "Tempo", className: "zone-tempo" },
  { max: 1.03, label: "Seuil", className: "zone-seuil" },
  { max: 1.15, label: "VO2max", className: "zone-vo2" },
  { max: Infinity, label: "Anaérobie", className: "zone-anaerobie" },
];

function powerZone(watts, cpWatts) {
  const ratio = watts / cpWatts;
  return POWER_ZONES.find((zone) => ratio <= zone.max);
}

function zonePillHtml(watts, cpWatts) {
  const zone = powerZone(watts, cpWatts);
  return `<span class="zone-pill ${zone.className}">${zone.label}</span>`;
}

// Bande d'incertitude à l'échelle (mean ± std) avec un repère pour le
// temps de la fenêtre retenue — bornes calculées dynamiquement à partir
// des données reçues, jamais de plage fixe en dur.
function computeRangeBar(predictedTimeS, meanTimeS, stdTimeS) {
  const lo = Math.min(predictedTimeS, meanTimeS - 2 * stdTimeS);
  const hi = Math.max(predictedTimeS, meanTimeS + 2 * stdTimeS);
  const span = hi - lo || 1;
  const pct = (s) => ((s - lo) / span) * 100;
  return {
    axisMin: formatMmSs(lo),
    axisMax: formatMmSs(hi),
    bandLeftPct: pct(meanTimeS - stdTimeS),
    bandWidthPct: pct(meanTimeS + stdTimeS) - pct(meanTimeS - stdTimeS),
    markerLeftPct: pct(predictedTimeS),
  };
}

async function loadSegments() {
  const response = await fetch("/segments");
  if (!response.ok) {
    // Le cas normal ici (T-44e) : pas encore connecté, /segments répond
    // 401 — sans ce message, le menu resterait vide sans explication
    // (auth.js affiche déjà le bouton "Se connecter", mais ce menu-ci
    // ne le répète pas).
    const error = await response.json();
    statusEl.textContent = `Erreur : ${error.detail}`;
    return;
  }
  const segments = await response.json();
  for (const segment of segments) {
    const option = document.createElement("option");
    option.value = segment.id;
    const km = (segment.distance_m / 1000).toFixed(1);
    option.textContent = `${segment.name} (${km} km, D+ ${Math.round(segment.elevation_gain_m)} m)`;
    segmentSelect.appendChild(option);
  }

  // Pré-sélection venant de "Segments du jour" (clic sur une ligne, T-43) :
  // ?segment=<id> dans l'URL plutôt qu'un état serveur (st.session_state
  // côté Streamlit, T-35) — deux pages statiques indépendantes n'ont pas
  // de session partagée, l'URL est le seul canal entre les deux.
  const preselectedId = new URLSearchParams(window.location.search).get("segment");
  if (preselectedId !== null) {
    segmentSelect.value = preselectedId; // no-op si l'id n'existe pas parmi les options
  }
}

const MAX_WINDOWS_PER_DAY = 2;

// Demandé : l'écart entre deux créneaux voisins du même jour est souvent
// minime (quelques secondes) — afficher les 6+ heures d'une même journée
// n'apporte rien, seuls les 2 meilleurs créneaux de chaque jour comptent.
// `windows` est déjà trié par temps prédit croissant (l'API, T-27) : les
// 2 premiers rencontrés pour un jour calendaire donné SONT ses 2
// meilleurs, pas 2 au hasard — pas besoin de retrier ici.
function topWindowsPerDay(windows, maxPerDay = MAX_WINDOWS_PER_DAY) {
  const countByDay = new Map();
  return windows.filter((w) => {
    const day = w.time.slice(0, 10); // "2026-09-13T14:00:00" -> "2026-09-13"
    const count = countByDay.get(day) ?? 0;
    if (count >= maxPerDay) return false;
    countByDay.set(day, count + 1);
    return true;
  });
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

function renderPacingCard(pacing, cpWatts) {
  return `
    <article class="card">
      <h2>Stratégie de pacing</h2>
      <p class="big-figure">${Math.round(pacing.power_w)} W ${zonePillHtml(pacing.power_w, cpWatts)}</p>
      <p class="card-note">
        Puissance constante optimale, vent mis à part — un seul tronçon à pente
        moyenne, pas un profil variable : aucun relevé pente/distance détaillé
        n'existe au niveau segment.
      </p>
    </article>`;
}

function renderKomCard(kom, cpWatts) {
  const warning = kom.power_w_extrapolated
    ? `<p class="warn">⚠ ${kom.seconds}s hors de la plage calibrée du modèle de puissance seuil — estimation moins fiable</p>`
    : "";
  return `
    <article class="card">
      <h2>KOM du segment</h2>
      <p class="big-figure">${formatMmSs(kom.seconds)}</p>
      <p class="card-note">
        Puissance estimée pour toi : <strong>${Math.round(kom.power_w)} W</strong>
        ${zonePillHtml(kom.power_w, cpWatts)}
      </p>
      <p class="card-note">Ton modèle de puissance seuil, pas la puissance réelle du recordman.</p>
      ${warning}
    </article>`;
}

function renderPrCard(pr, cpWatts) {
  if (pr === null) {
    return `
      <article class="card">
        <h2>Mon PR</h2>
        <p class="card-note">Jamais roulé ce segment.</p>
      </article>`;
  }
  const effort = pr.effort;
  const wattsHtml =
    effort && effort.average_watts !== null
      ? `${effort.average_watts} W ${zonePillHtml(effort.average_watts, cpWatts)}`
      : "puissance non disponible";
  const draftWarning =
    effort && effort.draft_status !== "solo"
      ? `<p class="warn">⚠ statut draft de cet effort : ${effort.draft_status} — comparaison possiblement biaisée</p>`
      : "";
  return `
    <article class="card">
      <h2>Mon PR</h2>
      <p class="big-figure">${formatMmSs(pr.seconds)}</p>
      <p class="card-note">${wattsHtml}</p>
      ${draftWarning}
    </article>`;
}

function renderUncertaintyCard(uncertainty, predictedTimeS) {
  if (uncertainty === null) {
    return `
      <article class="card">
        <h2>Avec incertitude</h2>
        <p class="card-note">
          Pas assez d'efforts récents proches du maximum (90 derniers jours) pour
          estimer une distribution de forme à échantillonner.
        </p>
      </article>`;
  }
  const range = computeRangeBar(predictedTimeS, uncertainty.mean_time_s, uncertainty.std_time_s);
  return `
    <article class="card card--accent">
      <h2>Avec incertitude</h2>
      <p class="big-figure">
        ${formatMmSs(uncertainty.mean_time_s)}
        <span class="figure-sub">± ${uncertainty.std_time_s.toFixed(0)} s</span>
      </p>
      <p class="card-note">
        Puissance seuil, forme récente (${uncertainty.n_samples} tirages Monte-Carlo), vent
        perturbé — écart-type ASSUMÉ à 20%, pas mesuré.
      </p>
      <div class="range-bar">
        <div class="range-track">
          <div class="range-band" style="left: ${range.bandLeftPct}%; width: ${range.bandWidthPct}%"></div>
          <div class="range-mark" style="left: ${range.markerLeftPct}%"></div>
        </div>
        <div class="range-labels"><span>${range.axisMin}</span><span>${range.axisMax}</span></div>
        <div class="range-callouts">
          <span>Repère : <strong>${formatMmSs(predictedTimeS)}</strong> (fenêtre retenue)</span>
          <span>Bande : <strong>${formatMmSs(uncertainty.mean_time_s)} ± ${uncertainty.std_time_s.toFixed(0)} s</strong> (forme actuelle)</span>
        </div>
      </div>
    </article>`;
}

// "Courbe de puissance réelle" (T-42) : comparaison additive, ne remplace
// pas le temps prédit du modèle CP+W' ci-dessus dans le héro — répond à
// "si je donnais vraiment ma meilleure puissance déjà atteinte pour cette
// durée, avec ce vent, quel temps ça donnerait ?". `null` la plupart du
// temps hors de la plage mesurée (~3-20 min) : le message vient tel quel
// de l'API (interpolate_mmp_curve/simulate_segment_time_from_mmp_curve),
// pas reformulé ici.
function renderRealPowerCurveNote(realPowerCurve, reason, cpWatts) {
  if (realPowerCurve === null) {
    return `<p class="hero-note">Courbe de puissance réelle : ${reason}</p>`;
  }
  return `<p class="hero-note">
    Avec ta courbe de puissance mesurée (pas le modèle) : ${formatMmSs(realPowerCurve.predicted_time_s)}
    (${Math.round(realPowerCurve.power_w)} W ${zonePillHtml(realPowerCurve.power_w, cpWatts)}), vent inclus
  </p>`;
}

function renderResults(data) {
  const best = data.windows[0];
  const windKmh = Math.round(best.wind_speed_ms * 3.6);
  const tempC = Math.round(best.temperature_k - 273.15);
  const cpWatts = data.calibration.cp_watts;
  // "real_curve" (T-49c) : segment trop court pour le modèle CP+W'
  // (T-48a) — la puissance vient d'un effort RÉELLEMENT déjà atteint,
  // pas d'une extrapolation "requise". Le libellé doit refléter cette
  // différence, pas la maquiller derrière le même mot que d'habitude.
  const isRealCurve = best.power_source === "real_curve";
  const powerLabel = isRealCurve ? "W déjà atteints (mesurés)" : "W requis";

  resultsEl.innerHTML = `
    <section class="hero">
      <div class="hero-main">
        <p class="eyebrow">Meilleure fenêtre</p>
        <h1 class="hero-time">${formatMmSs(best.predicted_time_s)}</h1>
        <p class="hero-sub">
          ${formatDayHour(best.time)}
          ${zonePillHtml(best.required_power_w, cpWatts)} ${Math.round(best.required_power_w)} ${powerLabel}
        </p>
        ${
          isRealCurve
            ? `<p class="hero-note">Segment trop court pour le modèle CP+W' (moins de 3 min) —
               estimation basée sur ta courbe de puissance réellement mesurée, pas sur un modèle extrapolé.</p>`
            : renderRealPowerCurveNote(
                data.real_power_curve,
                data.real_power_curve_unavailable_reason,
                cpWatts
              )
        }
      </div>
      <div class="hero-stats">
        <div class="stat-tile">
          <div class="compass" style="--deg: ${compassRotationDeg(best.wind_direction_rad)}deg">
            <svg viewBox="0 0 40 40" fill="none" aria-hidden="true">
              <circle cx="20" cy="20" r="17" stroke="var(--border)" stroke-width="1.5" />
              <text x="20" y="7" text-anchor="middle" font-size="6" fill="var(--text-muted)" font-family="var(--font-body)">N</text>
              <g class="needle"><path d="M20 6 L23 20 L20 17 L17 20 Z" fill="var(--accent)" /></g>
            </svg>
          </div>
          <div>
            <p class="stat-value">${windKmh} km/h</p>
            <p class="stat-label">Vent du ${compassLabel(best.wind_direction_rad)}</p>
          </div>
        </div>
        <div class="stat-tile">
          <div>
            <p class="stat-value">${tempC}°C</p>
            <p class="stat-label">Température</p>
          </div>
        </div>
        <div class="stat-tile mono">
          <div>
            <p class="stat-value">${cpWatts.toFixed(0)} ± ${data.calibration.cp_watts_std.toFixed(0)} W</p>
            <p class="stat-label">Puissance seuil · CdA ${data.calibration.cda_m2.toFixed(3)} m² · Crr ${data.calibration.crr.toFixed(4)}</p>
          </div>
        </div>
      </div>
    </section>

    <section class="body-grid">
      <div class="col-left">
        ${renderPacingCard(data.pacing, cpWatts)}
        <div class="pair">
          ${renderKomCard(data.kom, cpWatts)}
          ${renderPrCard(data.pr, cpWatts)}
        </div>
      </div>
      <div class="col-right">
        ${renderUncertaintyCard(data.uncertainty, best.predicted_time_s)}
        <article class="card">
          <h2>Classement des créneaux</h2>
          <p class="card-note">
            Au plus ${MAX_WINDOWS_PER_DAY} par jour — l'écart entre deux heures voisines
            du même jour est souvent minime, inutile de lister chaque heure.
          </p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Créneau</th><th>Temps</th><th>Puissance</th><th>Vent</th><th>Temp.</th></tr>
              </thead>
              <tbody>${renderWindowRows(topWindowsPerDay(data.windows))}</tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  `;
  resultsEl.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusEl.textContent = "Calibration et prévision météo...";
  resultsEl.hidden = true;
  submitButton.disabled = true;

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
  } finally {
    submitButton.disabled = false;
  }
});

loadSegments();
