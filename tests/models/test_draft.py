"""Tests du facteur de réduction aérodynamique par drafting (T-19).

Deux régimes distincts, testés séparément :
- olds_draft_ratio / draft_ratio_for_preset("roue_collee"|"un_metre") :
  paceline à un seul leader (Olds 1995).
- draft_ratio_for_preset("groupe") : peloton, physique différente
  (Blocken et al. 2018), pas dérivée d'Olds.
Le critère de fin du ticket (gain fort sur le plat, quasi nul en forte
pente) est vérifié en bout de chaîne avec simulate_segment_time (T-13).
"""

import pytest

from segment_predictor.models.draft import (
    GROUP_DRAFT_RATIO,
    OLDS_MAX_GAP_M,
    draft_ratio_for_preset,
    olds_draft_ratio,
)
from segment_predictor.models.power import CriticalPowerFit
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time

_CP_FIT = CriticalPowerFit(
    cp_watts=280.0, w_prime_joules=18_000.0, r_squared=0.9, n_points=5, duration_range_s=(180, 1200)
)
_MASS_KG = 80.0
_CDA_M2 = 0.30
_CRR = 0.005


# ---- olds_draft_ratio -----------------------------------------------------------------------


def test_olds_draft_ratio_matches_known_values() -> None:
    # Valeurs calculées indépendamment depuis la formule (0.62 - 0.0104*g + 0.0452*g^2),
    # pas re-dérivées de l'implémentation testée.
    assert olds_draft_ratio(0.0) == pytest.approx(0.62, abs=1e-6)
    assert olds_draft_ratio(1.0) == pytest.approx(0.6548, abs=1e-6)
    assert olds_draft_ratio(2.0) == pytest.approx(0.78, abs=1e-6)


def test_olds_draft_ratio_stays_below_one_within_valid_range() -> None:
    """1.0 = aucun bénéfice de draft : le ratio ne doit jamais le dépasser
    sur la plage documentée comme valide, sinon la formule prédirait plus
    de traînée qu'en solo (non-sens physique)."""
    for gap_m in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, OLDS_MAX_GAP_M]:
        assert olds_draft_ratio(gap_m) <= 1.0


def test_olds_draft_ratio_raises_outside_valid_range() -> None:
    with pytest.raises(ValueError, match="plage de validité"):
        olds_draft_ratio(-0.1)
    with pytest.raises(ValueError, match="plage de validité"):
        olds_draft_ratio(OLDS_MAX_GAP_M + 0.1)


# ---- draft_ratio_for_preset ------------------------------------------------------------------


def test_draft_ratio_for_preset_solo_is_one() -> None:
    assert draft_ratio_for_preset("solo") == 1.0


def test_draft_ratio_for_preset_matches_olds_formula_for_paceline_presets() -> None:
    assert draft_ratio_for_preset("roue_collee") == pytest.approx(olds_draft_ratio(0.2))
    assert draft_ratio_for_preset("un_metre") == pytest.approx(olds_draft_ratio(1.0))


def test_draft_ratio_for_preset_groupe_is_not_derived_from_olds() -> None:
    """ "groupe" (peloton) n'est physiquement pas un cas particulier d'Olds
    (un seul leader) : sa valeur doit être nettement plus faible (plus de
    bénéfice) que même le meilleur cas Olds (roue collée), signe qu'elle
    vient bien d'une source différente (Blocken et al. 2018), pas d'un
    gap_m minuscule glissé dans la formule d'Olds."""
    assert draft_ratio_for_preset("groupe") == GROUP_DRAFT_RATIO
    assert GROUP_DRAFT_RATIO < olds_draft_ratio(0.0)


def test_draft_ratio_for_preset_raises_on_unknown_preset() -> None:
    with pytest.raises(ValueError, match="draft"):
        draft_ratio_for_preset("en_danseuse")


# ---- critère de fin du ticket : gain fort sur le plat, quasi nul en forte pente --------------


def test_draft_gain_is_large_on_flat_and_negligible_on_steep_climb() -> None:
    flat_chunk = [SegmentChunk(0.0, 2000.0, 0.0, 0.0)]
    climb_chunk = [SegmentChunk(0.0, 2000.0, 0.09, 0.0)]  # 9% : la gravité domine largement l'aéro

    def time_s(chunks: list[SegmentChunk], draft_ratio: float) -> float:
        return simulate_segment_time(
            chunks, _CP_FIT.cp_watts, _CP_FIT.w_prime_joules, _MASS_KG, _CDA_M2 * draft_ratio, _CRR
        )

    solo_ratio = draft_ratio_for_preset("solo")
    group_ratio = draft_ratio_for_preset("groupe")

    flat_gain_fraction = 1 - time_s(flat_chunk, group_ratio) / time_s(flat_chunk, solo_ratio)
    climb_gain_fraction = 1 - time_s(climb_chunk, group_ratio) / time_s(climb_chunk, solo_ratio)

    # Seuils vérifiés contre les vrais nombres (pas des valeurs supposées) :
    # même avec "groupe" (le preset qui réduit le plus l'aéro, 90%), le
    # gain en forte pente reste sous 5% quand le gain à plat dépasse 15%.
    assert flat_gain_fraction > 0.15
    assert climb_gain_fraction < 0.05  # quasi nul : la gravité écrase le terme aéro
    assert flat_gain_fraction > 5 * climb_gain_fraction  # le contraste demandé par le ticket
