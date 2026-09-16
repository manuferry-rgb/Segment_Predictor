// T-45c/T-46 : widget "Se connecter avec Strava" + "Synchroniser mes
// données" — partagé par index.html et segments-du-jour.html via un
// <script> classique (pas un module ES, projet sans build step) chargé
// AVANT le script propre à chaque page, pour que l'un n'ait pas à
// réimplémenter l'autre. Utilise l'élément #status DE LA PAGE (présent
// sur les deux) pour les messages de synchro, pas un élément propre au
// widget — pas besoin de dupliquer une zone de statut.

async function renderAuthWidget() {
  const container = document.getElementById("auth-widget");
  if (!container) return;

  const response = await fetch("/auth/me");
  const me = await response.json();

  if (me.authenticated) {
    container.innerHTML = `
      <span class="auth-status">Connecté comme ${me.firstname ?? "toi"}</span>
      <button type="button" class="auth-sync">Synchroniser mes données</button>
      <button type="button" class="auth-logout">Se déconnecter</button>
    `;
    container.querySelector(".auth-sync").addEventListener("click", runSync);
    container.querySelector(".auth-logout").addEventListener("click", async () => {
      await fetch("/auth/logout", { method: "POST" });
      window.location.reload();
    });
  } else {
    // Lien classique, pas fetch() : /auth/strava/login redirige vers
    // Strava, un fetch() suivrait la redirection en arrière-plan sans
    // jamais faire naviguer le navigateur lui-même.
    container.innerHTML = `<a class="auth-login" href="/auth/strava/login">Se connecter avec Strava</a>`;
  }
}

async function runSync() {
  const statusEl = document.getElementById("status");
  const button = document.querySelector(".auth-sync");
  button.disabled = true;
  if (statusEl) {
    statusEl.textContent =
      "Synchronisation avec Strava en cours (peut prendre plusieurs minutes selon ton historique)...";
  }

  try {
    const response = await fetch("/sync", { method: "POST" });
    const summary = await response.json();
    if (!response.ok) {
      if (statusEl) statusEl.textContent = `Erreur : ${summary.detail}`;
      return;
    }

    // Le quota Strava est PAR APPLICATION (T-46, voir la docstring de
    // /sync) : le signaler plutôt que de laisser croire que tout a été
    // récupéré alors qu'il reste des données à demain.
    const quotaReached =
      summary.streams_quota_reached ||
      summary.activity_details_quota_reached ||
      summary.segments_quota_reached ||
      summary.segment_streams_quota_reached;
    const quotaNote = quotaReached
      ? " ⚠ Quota Strava atteint (partagé entre tous les utilisateurs) — reviens demain pour continuer."
      : "";

    if (statusEl) {
      statusEl.textContent =
        `Synchronisé : ${summary.streams_fetched} nouvelle(s) sortie(s) avec capteur, ` +
        `${summary.segments_fetched} nouveau(x) segment(s) favori(s).${quotaNote}`;
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = `Erreur réseau : ${err.message}`;
  } finally {
    button.disabled = false;
  }
}

renderAuthWidget();
