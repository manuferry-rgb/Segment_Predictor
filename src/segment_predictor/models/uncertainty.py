"""Propagation d'incertitude par Monte-Carlo (T-28).

Fonction pure — aucun I/O. Trois sources d'incertitude, échantillonnées
indépendamment à chaque tirage :

- **CP/W'** : Normal(cp_watts, cp_watts_std) / Normal(w_prime_joules,
  w_prime_joules_std) — écarts-types réels, issus de la covariance de la
  régression du fit (T-09/T-28, `fit_critical_power`), pas inventés.
- **Forme** : tirage AVEC REMISE (bootstrap) dans `performance_index_
  samples` — la distribution empirique récente de l'indice de
  performance (T-23), pas un modèle paramétrique. Choix délibéré plutôt
  que de réutiliser la régression de T-24 : on a déjà trouvé qu'elle
  généralise mal (R² négatif en validation croisée temporelle), s'appuyer
  sur son résidu comme incertitude calibrée n'aurait pas de sens.
  L'échantillon `performance_index_sample` remet à l'échelle CP et W'
  ensemble (`cp_watts_sample * performance_index_sample`, idem W') :
  cohérent avec la définition de l'indice (T-23) comme ratio puissance
  réelle / puissance prédite par CP+W'/T.
- **Vent** : Normal(wind_speed_ms, wind_relative_std * wind_speed_ms),
  tronqué à 0 (pas de vitesse de vent négative). `wind_relative_std` est
  une hypothèse ASSUMÉE et documentée par l'appelant, pas mesurée : le
  projet n'a pas d'historique prévision-vs-réalisé pour la calibrer.

Un tirage dont la vitesse n'a pas de solution (`simulate_segment_time`,
T-13 — ex. vent de face extrême après perturbation) est exclu du calcul
plutôt que de faire planter toute la propagation ; le nombre d'exclusions
est rapporté, pas caché.
"""

from dataclasses import dataclass

import numpy as np

from segment_predictor.models.segment import SegmentChunk, simulate_segment_time


@dataclass(frozen=True)
class UncertaintyResult:
    """`samples_s` : tous les temps prédits valides, pour qui voudrait
    tracer un histogramme plutôt que se limiter à moyenne ± écart-type."""

    mean_time_s: float
    std_time_s: float
    n_samples: int
    n_excluded: int
    samples_s: np.ndarray


def propagate_uncertainty(
    chunks: list[SegmentChunk],
    cp_watts: float,
    cp_watts_std: float,
    w_prime_joules: float,
    w_prime_joules_std: float,
    mass_kg: float,
    cda_m2: float,
    crr: float,
    performance_index_samples: list[float],
    wind_speed_ms: float,
    wind_direction_rad: float,
    wind_relative_std: float,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> UncertaintyResult:
    """`n_samples` tirages Monte-Carlo de (CP, W', forme, vent), un temps
    prédit par tirage valide (voir docstring du module pour chaque
    source). `rng` injectable pour des tests reproductibles ; laissé à
    None en usage normal pour un tirage réellement aléatoire à chaque
    appel.
    """
    if not performance_index_samples:
        raise ValueError(
            "performance_index_samples vide : aucune distribution de forme à échantillonner"
        )
    if n_samples <= 0:
        raise ValueError(f"n_samples doit être positif, reçu {n_samples}")

    if rng is None:
        rng = np.random.default_rng()

    performance_index_samples = np.asarray(performance_index_samples, dtype=float)

    cp_draws = (
        rng.normal(cp_watts, cp_watts_std, size=n_samples)
        if cp_watts_std > 0
        else np.full(n_samples, cp_watts)
    )
    w_prime_draws = (
        rng.normal(w_prime_joules, w_prime_joules_std, size=n_samples)
        if w_prime_joules_std > 0
        else np.full(n_samples, w_prime_joules)
    )
    performance_draws = rng.choice(performance_index_samples, size=n_samples, replace=True)
    wind_draws = (
        np.clip(
            rng.normal(wind_speed_ms, wind_relative_std * wind_speed_ms, size=n_samples), 0.0, None
        )
        if wind_relative_std > 0
        else np.full(n_samples, wind_speed_ms)
    )

    predicted_times_s = []
    n_excluded = 0
    for i in range(n_samples):
        try:
            predicted_time_s = simulate_segment_time(
                chunks,
                cp_draws[i] * performance_draws[i],
                w_prime_draws[i] * performance_draws[i],
                mass_kg,
                cda_m2,
                crr,
                wind_speed_ms=wind_draws[i],
                wind_direction_rad=wind_direction_rad,
            )
        except ValueError:
            n_excluded += 1
            continue
        predicted_times_s.append(predicted_time_s)

    samples_s = np.array(predicted_times_s)
    return UncertaintyResult(
        mean_time_s=float(np.mean(samples_s)),
        std_time_s=float(np.std(samples_s)),
        n_samples=n_samples,
        n_excluded=n_excluded,
        samples_s=samples_s,
    )
