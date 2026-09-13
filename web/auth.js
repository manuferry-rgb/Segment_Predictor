// T-45c : widget "Se connecter avec Strava" — partagé par index.html et
// segments-du-jour.html via un <script> classique (pas un module ES,
// projet sans build step) chargé AVANT le script propre à chaque page,
// pour que l'un n'ait pas à réimplémenter l'autre.

async function renderAuthWidget() {
  const container = document.getElementById("auth-widget");
  if (!container) return;

  const response = await fetch("/auth/me");
  const me = await response.json();

  if (me.authenticated) {
    container.innerHTML = `
      <span class="auth-status">Connecté comme ${me.firstname ?? "toi"}</span>
      <button type="button" class="auth-logout">Se déconnecter</button>
    `;
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

renderAuthWidget();
