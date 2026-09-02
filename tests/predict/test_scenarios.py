"""Tests de la comparaison de scénarios de draft sur un même segment (T-20)."""

import duckdb
import pyarrow as pa
import pytest

from segment_predictor.models.draft import draft_ratio_for_preset
from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time
from segment_predictor.predict.scenarios import (
    ALL_DRAFT_PRESETS,
    compare_draft_scenarios,
    compare_draft_scenarios_for_segment,
)

_CP_FIT = CriticalPowerFit(
    cp_watts=280.0, w_prime_joules=18_000.0, r_squared=0.9, n_points=5, duration_range_s=(180, 1200)
)
_MASS_KG = 80.0
_CDA_M2 = 0.30
_CRR = 0.005
_CHUNKS = [SegmentChunk(0.0, 2000.0, 0.0, 0.0)]


# ---- compare_draft_scenarios (fonction pure) --------------------------------------------------


def test_compare_draft_scenarios_returns_one_result_per_preset() -> None:
    results = compare_draft_scenarios(_CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    assert [r.preset for r in results] == list(ALL_DRAFT_PRESETS)


def test_compare_draft_scenarios_times_match_direct_simulation() -> None:
    """Pas de logique dupliquée : le temps par preset doit être exactement
    ce que donnerait simulate_segment_time avec le CdA réduit à la main."""
    results = compare_draft_scenarios(_CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    for result in results:
        ratio = draft_ratio_for_preset(result.preset)
        expected_time_s = simulate_segment_time(
            _CHUNKS, _CP_FIT.cp_watts, _CP_FIT.w_prime_joules, _MASS_KG, _CDA_M2 * ratio, _CRR
        )
        assert result.predicted_time_s == pytest.approx(expected_time_s)


def test_compare_draft_scenarios_solo_has_zero_gain() -> None:
    results = compare_draft_scenarios(_CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)

    solo = next(r for r in results if r.preset == "solo")
    assert solo.gain_vs_solo_s == pytest.approx(0.0)
    assert solo.gain_vs_solo_fraction == pytest.approx(0.0)


def test_compare_draft_scenarios_gain_grows_with_draft_benefit() -> None:
    """Plus le ratio de draft_ratio_for_preset est petit (plus de
    réduction du CdA), plus le gain doit être grand — vérifié avec les
    vrais ratios plutôt que supposé : roue_collee (gap≈0.2m) réduit LÉGÈREMENT
    plus le CdA que un_metre (gap=1m), donc gagne plus, et groupe (0.10,
    peloton) gagne largement plus que les deux presets Olds."""
    results = {
        r.preset: r for r in compare_draft_scenarios(_CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)
    }

    assert results["groupe"].gain_vs_solo_s > results["roue_collee"].gain_vs_solo_s
    assert results["roue_collee"].gain_vs_solo_s > results["un_metre"].gain_vs_solo_s > 0


def test_compare_draft_scenarios_raises_when_solo_not_in_presets() -> None:
    with pytest.raises(ValueError, match="solo"):
        compare_draft_scenarios(
            _CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR, presets=("roue_collee", "groupe")
        )


# ---- compare_draft_scenarios_for_segment (DB) --------------------------------------------------


def _make_db(segments):
    conn = duckdb.connect(":memory:")
    conn.register("_rows", pa.Table.from_pylist(segments))
    conn.execute("CREATE TABLE segments AS SELECT * FROM _rows")
    conn.unregister("_rows")
    return conn


def test_compare_draft_scenarios_for_segment_uses_segment_distance_and_grade() -> None:
    conn = _make_db([{"id": 100, "name": "Plat", "distance_m": 2000.0, "average_grade": 0.0}])

    results = compare_draft_scenarios_for_segment(
        conn, segment_id=100, mass_kg=_MASS_KG, cda_m2=_CDA_M2, crr=_CRR, cp_fit=_CP_FIT
    )

    expected = compare_draft_scenarios(_CHUNKS, _CP_FIT, _MASS_KG, _CDA_M2, _CRR)
    assert [r.predicted_time_s for r in results] == pytest.approx(
        [r.predicted_time_s for r in expected]
    )


def test_compare_draft_scenarios_for_segment_raises_when_segment_not_found() -> None:
    conn = _make_db([{"id": 100, "name": "Plat", "distance_m": 2000.0, "average_grade": 0.0}])

    with pytest.raises(ValueError, match="404"):
        compare_draft_scenarios_for_segment(
            conn, segment_id=404, mass_kg=_MASS_KG, cda_m2=_CDA_M2, crr=_CRR, cp_fit=_CP_FIT
        )
