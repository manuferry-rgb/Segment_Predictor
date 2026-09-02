"""Tests de la propagation d'incertitude par Monte-Carlo (T-28).

Trois sources échantillonnées à chaque tirage : CP/W' (écart-type de la
régression, T-09/T-28), forme (tirage avec remise dans la distribution
empirique récente de l'indice de performance, T-23), vent (écart-type
relatif assumé et documenté, pas mesuré — pas d'historique prévision vs
réalisé disponible pour le calibrer).
"""

import numpy as np
import pytest

from segment_predictor.models.segment import SegmentChunk, simulate_segment_time
from segment_predictor.models.uncertainty import propagate_uncertainty

_MASS_KG = 75.0
_CDA_M2 = 0.30
_CRR = 0.005
_CHUNK = [SegmentChunk(0.0, 2000.0, 0.0, heading_rad=0.0)]
_CP_WATTS = 250.0
_W_PRIME_JOULES = 20_000.0


def test_propagate_uncertainty_has_near_zero_spread_with_no_uncertainty_sources() -> None:
    """Toutes les sources à variance nulle : les tirages doivent converger
    vers (quasi) la même valeur que simulate_segment_time direct."""
    baseline_s = simulate_segment_time(_CHUNK, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)

    result = propagate_uncertainty(
        _CHUNK,
        cp_watts=_CP_WATTS,
        cp_watts_std=0.0,
        w_prime_joules=_W_PRIME_JOULES,
        w_prime_joules_std=0.0,
        mass_kg=_MASS_KG,
        cda_m2=_CDA_M2,
        crr=_CRR,
        performance_index_samples=[1.0],
        wind_speed_ms=0.0,
        wind_direction_rad=0.0,
        wind_relative_std=0.0,
        n_samples=200,
        rng=np.random.default_rng(0),
    )

    assert result.mean_time_s == pytest.approx(baseline_s, rel=1e-6)
    assert result.std_time_s == pytest.approx(0.0, abs=1e-6)
    assert result.n_samples == 200
    assert result.n_excluded == 0


def test_propagate_uncertainty_spread_grows_with_cp_uncertainty() -> None:
    kwargs = dict(
        mass_kg=_MASS_KG,
        cda_m2=_CDA_M2,
        crr=_CRR,
        performance_index_samples=[1.0],
        wind_speed_ms=0.0,
        wind_direction_rad=0.0,
        wind_relative_std=0.0,
        n_samples=500,
    )
    low_std = propagate_uncertainty(
        _CHUNK,
        cp_watts=_CP_WATTS,
        cp_watts_std=1.0,
        w_prime_joules=_W_PRIME_JOULES,
        w_prime_joules_std=0.0,
        rng=np.random.default_rng(1),
        **kwargs,
    )
    high_std = propagate_uncertainty(
        _CHUNK,
        cp_watts=_CP_WATTS,
        cp_watts_std=15.0,
        w_prime_joules=_W_PRIME_JOULES,
        w_prime_joules_std=0.0,
        rng=np.random.default_rng(1),
        **kwargs,
    )

    assert high_std.std_time_s > low_std.std_time_s


def test_propagate_uncertainty_spread_grows_with_performance_index_spread() -> None:
    kwargs = dict(
        cp_watts=_CP_WATTS,
        cp_watts_std=0.0,
        w_prime_joules=_W_PRIME_JOULES,
        w_prime_joules_std=0.0,
        mass_kg=_MASS_KG,
        cda_m2=_CDA_M2,
        crr=_CRR,
        wind_speed_ms=0.0,
        wind_direction_rad=0.0,
        wind_relative_std=0.0,
        n_samples=500,
    )
    narrow = propagate_uncertainty(
        _CHUNK,
        performance_index_samples=[0.99, 1.0, 1.01],
        rng=np.random.default_rng(2),
        **kwargs,
    )
    wide = propagate_uncertainty(
        _CHUNK,
        performance_index_samples=[0.7, 1.0, 1.3],
        rng=np.random.default_rng(2),
        **kwargs,
    )

    assert wide.std_time_s > narrow.std_time_s


def test_propagate_uncertainty_spread_grows_with_wind_uncertainty() -> None:
    kwargs = dict(
        cp_watts=_CP_WATTS,
        cp_watts_std=0.0,
        w_prime_joules=_W_PRIME_JOULES,
        w_prime_joules_std=0.0,
        mass_kg=_MASS_KG,
        cda_m2=_CDA_M2,
        crr=_CRR,
        performance_index_samples=[1.0],
        wind_speed_ms=5.0,
        wind_direction_rad=0.0,
        n_samples=500,
    )
    no_wind_uncertainty = propagate_uncertainty(
        _CHUNK, wind_relative_std=0.0, rng=np.random.default_rng(3), **kwargs
    )
    with_wind_uncertainty = propagate_uncertainty(
        _CHUNK, wind_relative_std=0.3, rng=np.random.default_rng(3), **kwargs
    )

    assert with_wind_uncertainty.std_time_s > no_wind_uncertainty.std_time_s


def test_propagate_uncertainty_raises_on_empty_performance_index_samples() -> None:
    with pytest.raises(ValueError, match="performance_index_samples"):
        propagate_uncertainty(
            _CHUNK,
            cp_watts=_CP_WATTS,
            cp_watts_std=0.0,
            w_prime_joules=_W_PRIME_JOULES,
            w_prime_joules_std=0.0,
            mass_kg=_MASS_KG,
            cda_m2=_CDA_M2,
            crr=_CRR,
            performance_index_samples=[],
            wind_speed_ms=0.0,
            wind_direction_rad=0.0,
            wind_relative_std=0.0,
        )


def test_propagate_uncertainty_raises_on_non_positive_n_samples() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        propagate_uncertainty(
            _CHUNK,
            cp_watts=_CP_WATTS,
            cp_watts_std=0.0,
            w_prime_joules=_W_PRIME_JOULES,
            w_prime_joules_std=0.0,
            mass_kg=_MASS_KG,
            cda_m2=_CDA_M2,
            crr=_CRR,
            performance_index_samples=[1.0],
            wind_speed_ms=0.0,
            wind_direction_rad=0.0,
            wind_relative_std=0.0,
            n_samples=0,
        )
