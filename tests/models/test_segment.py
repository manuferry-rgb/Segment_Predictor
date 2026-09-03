"""Tests du découpage de segment en tronçons ~50m (T-12) et de la simulation
d'un segment complet (T-13).

Fonctions pures, aucun I/O. `smooth_altitude` est testée seule (sur un
profil synthétique bruité de vérité connue) avant `chunk_segment`.
"""

import math

import numpy as np
import pytest

from segment_predictor.models.physics import cyclist_speed_from_power
from segment_predictor.models.polyline import decode_polyline
from segment_predictor.models.segment import (
    STANDARD_AIR_DENSITY_KG_M3,
    SegmentChunk,
    bearing_rad,
    chunk_segment,
    haversine_distance_m,
    segment_chunks_from_polyline,
    simulate_segment_time,
    smooth_altitude,
)

METERS_PER_DEGREE = 111_320.0  # approximation usuelle à l'équateur

# ---- smooth_altitude ------------------------------------------------------------------


def test_smooth_altitude_leaves_a_noiseless_constant_profile_unchanged() -> None:
    distance_m = np.arange(0, 200, 5.0)
    altitude_m = np.full_like(distance_m, 150.0)

    smoothed = smooth_altitude(distance_m, altitude_m, window_m=30.0)

    np.testing.assert_allclose(smoothed, 150.0)


def test_smooth_altitude_reduces_noise_on_a_known_linear_profile() -> None:
    """Bruit GPS synthétique (+/-3m, centré) sur une pente constante connue :
    le profil lissé doit se rapprocher nettement plus du vrai profil que le
    profil brut — c'est la justification quantitative du choix de méthode."""
    rng = np.random.default_rng(seed=7)
    distance_m = np.arange(0, 500, 2.0)  # échantillonnage dense, comme un vrai stream GPS
    true_altitude_m = 100.0 + 0.04 * distance_m  # pente constante 4%
    noisy_altitude_m = true_altitude_m + rng.normal(0.0, 3.0, size=distance_m.shape)

    smoothed = smooth_altitude(distance_m, noisy_altitude_m, window_m=30.0)

    raw_mean_error_m = np.abs(noisy_altitude_m - true_altitude_m).mean()
    smoothed_mean_error_m = np.abs(smoothed - true_altitude_m).mean()
    assert smoothed_mean_error_m < raw_mean_error_m * 0.5


def test_smooth_altitude_blurs_a_sharp_step() -> None:
    """Limite assumée, pas cachée : une vraie cassure nette est adoucie par
    la fenêtre, pas préservée — le prix à payer pour annuler le bruit GPS."""
    distance_m = np.arange(0, 200, 2.0)
    altitude_m = np.where(distance_m < 100, 0.0, 20.0)  # marche nette de 20m à 100m

    smoothed = smooth_altitude(distance_m, altitude_m, window_m=30.0)

    idx_at_step = np.argmin(np.abs(distance_m - 100))
    assert 0.0 < smoothed[idx_at_step] < 20.0


def test_smooth_altitude_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="longueur"):
        smooth_altitude(np.array([0.0, 10.0, 20.0]), np.array([100.0, 101.0]))


def test_smooth_altitude_raises_on_decreasing_distance() -> None:
    with pytest.raises(ValueError, match="croissante"):
        smooth_altitude(np.array([0.0, 10.0, 5.0]), np.array([100.0, 101.0, 102.0]))


# ---- chunk_segment ----------------------------------------------------------------------


def test_chunk_segment_recovers_exact_constant_grade_without_noise() -> None:
    distance_m = np.arange(0, 201, 10.0)  # 0..200m, 200m total, multiple exact de 50
    true_grade = 0.05
    altitude_m = distance_m * true_grade
    # chemin plein nord : longitude constante, latitude qui avance avec la distance
    lat = distance_m / METERS_PER_DEGREE
    lng = np.zeros_like(distance_m)

    chunks = chunk_segment(distance_m, altitude_m, lat, lng, chunk_length_m=50.0)

    assert len(chunks) == 4
    assert [c.start_distance_m for c in chunks] == [0.0, 50.0, 100.0, 150.0]
    for chunk in chunks:
        assert chunk.length_m == pytest.approx(50.0)
        assert chunk.grade == pytest.approx(true_grade)
        assert chunk.heading_rad == pytest.approx(0.0, abs=1e-6)  # plein nord


def test_chunk_segment_handles_a_remainder_shorter_last_chunk() -> None:
    distance_m = np.arange(0, 121, 10.0)  # 120m total, pas un multiple de 50
    altitude_m = distance_m * 0.02
    lat = distance_m / METERS_PER_DEGREE
    lng = np.zeros_like(distance_m)

    chunks = chunk_segment(distance_m, altitude_m, lat, lng, chunk_length_m=50.0)

    assert len(chunks) == 3
    assert [c.length_m for c in chunks] == pytest.approx([50.0, 50.0, 20.0])
    assert sum(c.length_m for c in chunks) == pytest.approx(120.0)


def test_chunk_segment_headings_match_known_compass_directions() -> None:
    """1er tronçon plein nord, 2e plein est — vérifiable à la main via la
    définition même du cap (0 = nord, croissant vers l'est)."""
    distance_m = np.array([0.0, 50.0, 100.0])
    lat = np.array([0.0, 50.0 / METERS_PER_DEGREE, 50.0 / METERS_PER_DEGREE])
    lng = np.array([0.0, 0.0, 50.0 / METERS_PER_DEGREE])
    altitude_m = np.array([0.0, 0.0, 0.0])

    chunks = chunk_segment(distance_m, altitude_m, lat, lng, chunk_length_m=50.0)

    assert len(chunks) == 2
    assert chunks[0].heading_rad == pytest.approx(0.0, abs=1e-6)  # nord
    assert chunks[1].heading_rad == pytest.approx(math.pi / 2, abs=1e-3)  # est


def test_bearing_rad_matches_known_compass_directions() -> None:
    """Publique (T-27) : testée directement, pas seulement via
    chunk_segment — c'est ce qui donnera le cap global d'un segment
    depuis ses start/end latlng."""
    assert bearing_rad(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=1e-6)  # nord
    assert bearing_rad(0.0, 0.0, 0.0, 1.0) == pytest.approx(math.pi / 2, abs=1e-6)  # est
    assert bearing_rad(0.0, 0.0, -1.0, 0.0) == pytest.approx(math.pi, abs=1e-6)  # sud
    assert bearing_rad(0.0, 0.0, 0.0, -1.0) == pytest.approx(3 * math.pi / 2, abs=1e-6)  # ouest


def test_chunk_segment_raises_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="longueur"):
        chunk_segment(
            np.array([0.0, 50.0]),
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.1]),
            np.array([0.0, 0.0]),
        )


def test_chunk_segment_raises_on_non_positive_chunk_length() -> None:
    distance_m = np.array([0.0, 50.0, 100.0])
    zeros = np.zeros_like(distance_m)
    with pytest.raises(ValueError, match="chunk_length_m"):
        chunk_segment(distance_m, zeros, zeros, zeros, chunk_length_m=0.0)


# ---- haversine_distance_m / segment_chunks_from_polyline (T-32) ------------------------


def test_haversine_distance_m_matches_known_distance_at_equator() -> None:
    """1° de latitude ou de longitude à l'équateur vaut ~111.32km — même
    approximation que METERS_PER_DEGREE, déjà utilisée plus haut pour
    construire des fixtures de cap connu."""
    north = haversine_distance_m(0.0, 0.0, 1.0, 0.0)
    east = haversine_distance_m(0.0, 0.0, 0.0, 1.0)
    assert north == pytest.approx(METERS_PER_DEGREE, rel=0.01)
    assert east == pytest.approx(METERS_PER_DEGREE, rel=0.01)


def test_haversine_distance_m_is_zero_for_identical_points() -> None:
    assert haversine_distance_m(45.5, 7.4, 45.5, 7.4) == pytest.approx(0.0, abs=1e-9)


def test_segment_chunks_from_polyline_headings_match_known_compass_directions() -> None:
    """Mêmes points que test_chunk_segment_headings_match_known_compass_directions
    (nord puis est), même vérité terrain — mais depuis un polyline décodé
    (lat/lng seuls) plutôt qu'un stream distance/altitude/latlng complet."""
    points = [
        (0.0, 0.0),
        (50.0 / METERS_PER_DEGREE, 0.0),
        (50.0 / METERS_PER_DEGREE, 50.0 / METERS_PER_DEGREE),
    ]

    chunks = segment_chunks_from_polyline(points, average_grade=0.03)

    assert len(chunks) == 2
    assert chunks[0].heading_rad == pytest.approx(0.0, abs=1e-6)  # nord
    assert chunks[1].heading_rad == pytest.approx(math.pi / 2, abs=1e-3)  # est


def test_segment_chunks_from_polyline_applies_average_grade_to_every_chunk() -> None:
    """Limite ASSUMÉE du ticket T-32 : pas de profil d'altitude par tronçon
    disponible depuis un polyline (lat/lng seulement, pas d'élévation) —
    la même pente moyenne du segment entier est appliquée partout, seul
    le cap varie réellement d'un tronçon à l'autre."""
    points = [(0.0, 0.0), (50.0 / METERS_PER_DEGREE, 0.0), (100.0 / METERS_PER_DEGREE, 0.0)]

    chunks = segment_chunks_from_polyline(points, average_grade=0.07)

    assert all(c.grade == pytest.approx(0.07) for c in chunks)


def test_segment_chunks_from_polyline_start_distances_are_cumulative() -> None:
    points = [(0.0, 0.0), (50.0 / METERS_PER_DEGREE, 0.0), (100.0 / METERS_PER_DEGREE, 0.0)]

    chunks = segment_chunks_from_polyline(points, average_grade=0.0)

    assert chunks[0].start_distance_m == pytest.approx(0.0)
    assert chunks[1].start_distance_m == pytest.approx(chunks[0].length_m)
    assert chunks[1].start_distance_m == pytest.approx(50.0, rel=0.01)


def test_segment_chunks_from_polyline_skips_duplicate_consecutive_points() -> None:
    """Deux points GPS identiques d'affilée (arrondi à 1e-5° du polyline,
    T-32 — peut arriver quand le tracé revient exactement sur un point
    déjà visité) donneraient un tronçon de longueur nulle et un cap
    indéfini (atan2(0, 0)) : ignoré plutôt que produit."""
    points = [
        (0.0, 0.0),
        (0.0, 0.0),  # doublon
        (50.0 / METERS_PER_DEGREE, 0.0),
    ]

    chunks = segment_chunks_from_polyline(points, average_grade=0.0)

    assert len(chunks) == 1
    assert chunks[0].length_m == pytest.approx(50.0, rel=0.01)


def test_segment_chunks_from_polyline_raises_on_fewer_than_two_points() -> None:
    with pytest.raises(ValueError, match="2 points"):
        segment_chunks_from_polyline([(0.0, 0.0)], average_grade=0.0)


def test_segment_chunks_from_polyline_raises_when_all_points_identical() -> None:
    with pytest.raises(ValueError, match="identiques"):
        segment_chunks_from_polyline([(0.0, 0.0), (0.0, 0.0)], average_grade=0.0)


def test_segment_chunks_from_polyline_matches_real_strava_segment() -> None:
    """Regression T-32, même polyline réel que
    tests/models/test_polyline.py (segment HBFH, id 31159007).

    Deux vérifications croisées avec des données stockées indépendamment
    du décodage : (1) la somme des longueurs de tronçons doit retomber
    près de `segments.distance_m` (15042.6m, rapporté par Strava
    lui-même) ; (2) le cap doit varier largement sur ce segment en
    boucle (départ et arrivée à 124m l'un de l'autre pour 15km de
    tracé), PAS rester proche de l'unique heading_rad stocké pour lui
    (262.96°, un cap quasi arbitraire vu la boucle) — c'est exactement
    ce que ce ticket corrige.
    """
    encoded = (
        "okaaH{usl@|ChChB`CdC`CLVC\\Wr@Oj@QtBAfALdCN~AL~@VxAp@dDh@zB\\~@zBxDN\\Jd@PlA^rBf@lJ"
        "\\hCt@|Dt@zBVl@R`AZ~Ed@vEX|BDr@LtEShLUxI@dBJfHAp@Ij@Y~@g@r@g@ZoGpDm@d@a@f@wApCm@zAe@"
        "dBu@~CKd@ATA`AHn@Nn@Rh@TXLHPDX?^Sx@u@VOn@Ob@Bh@VRZRp@NdA?f@OlA}@dEQdAGv@Qv@Gv@YfAOr@"
        "AN@ZDPAj@d@~Ax@fBd@vAHv@CNUn@@h@C\\Ol@Q~A[xG]bGA|@@ZBRN`@|@bBfFdJrBxCt@lA`JhQ|DlHr@"
        "`BrCfIt@hBf@r@pAxAb@t@Rl@lA|Er@~Bl@jBjBjFnAbDvA~CbAfCF^?p@MrDKfA?\\n@|Af@bABRAf@I^W"
        "`@UJuBH_@EmEDmADyALi@B{@CsB]s@Ie@Ai@@g@McAKmAIoAOcA[MKMQ_@q@QQOEQAaANaAT_Cl@qAb@WFc@"
        "Bk@Jm@A[Ca@E_@QaCW{A]w@KkCm@{Bk@m@IsA[i@QSMc@KKAw@Ye@WyAcAy@]g@?KCqAkBqDyCmBkB}BeCyA"
        "wBkAwB}@uBw@gC_@{@aAwAqA_B_@a@k@i@aAk@_Ac@{@m@yBkBqB{A{@y@gImJ_BwB_E{F}AwBsQ{TYU]K]@"
        "yAT[?i@Mu@_@oAw@oGcFcAq@[M]EiA?]Ci@WMKy@u@}@cAgA{Ai@iAkAoBSk@Mq@Gu@EkFG}@kAyFSs@iAwC"
        "Qm@UsAIs@Ao@DSJKNEv@MdBc@h@Gl@Aj@KfDcBXKhBWbCKXEVKTMjEaExCcBdBkAdF_EvCcCbAmAtHoKRS^Yn"
        "@WzCw@dCu@d@YNOh@iABUAWIaAa@qBg@uASe@Q[Si@MaBAuA_@wEJmACaA@U@SN_@IQ?YL}@XiANcADOJML]"
        "LoALc@Be@C_ANoAHoANYLq@HQBWp@oCX{A`@wAR[^WRAfDPj@JfCp@dA^TLj@b@TRpAxATRVLVHZ@XERQL[H"
        "c@tEcZ`@cBL_@jFaN`B}DpEkJ`AkB\\e@PMx@Yz@SPKNONWTo@zCcJ"
    )
    points = decode_polyline(encoded)

    chunks = segment_chunks_from_polyline(points, average_grade=0.001)

    total_length_m = sum(c.length_m for c in chunks)
    assert total_length_m == pytest.approx(15042.6, rel=0.005)

    headings_deg = [math.degrees(c.heading_rad) for c in chunks]
    assert max(headings_deg) - min(headings_deg) > 180  # loin d'un cap unique


# ---- simulate_segment_time (T-13) --------------------------------------------------------

_CP_WATTS = 250.0
_W_PRIME_JOULES = 20_000.0
_MASS_KG = 75.0
_CDA_M2 = 0.30
_CRR = 0.005


def _cp_only_time_s(chunks: list[SegmentChunk]) -> float:
    """Temps si on roulait à CP constant (sans le supplément W'/T) — calculable
    directement, sans boucle : sert de référence indépendante pour les tests."""
    total_time_s = 0.0
    for chunk in chunks:
        speed_ms = cyclist_speed_from_power(
            _CP_WATTS, chunk.grade, 0.0, _MASS_KG, _CDA_M2, _CRR, STANDARD_AIR_DENSITY_KG_M3
        )
        total_time_s += chunk.length_m / speed_ms
    return total_time_s


def test_simulate_segment_time_is_faster_than_riding_at_cp_alone() -> None:
    """W' > 0 => la puissance soutenable (CP + W'/T) est toujours > CP pour un T
    fini => le temps simulé doit être strictement inférieur au temps à CP seul."""
    chunks = [SegmentChunk(0.0, 1000.0, 0.0, 0.0)]

    simulated_time_s = simulate_segment_time(
        chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )

    assert simulated_time_s < _cp_only_time_s(chunks)
    # ordre de grandeur : ~250W sur plat, CdA/Crr routiers -> autour de 30km/h,
    # donc 1km en 100-150s environ (cf. sanity check T-10 : 250W ~ 302W/40km/h)
    assert 60.0 < simulated_time_s < 180.0


def test_simulate_segment_time_w_prime_contribution_shrinks_for_longer_segments() -> None:
    """W'/T devient négligeable quand T grandit : l'écart relatif avec le temps
    à CP seul doit être plus petit sur un segment long que sur un segment court."""
    short_chunks = [SegmentChunk(0.0, 500.0, 0.0, 0.0)]
    long_chunks = [SegmentChunk(0.0, 20_000.0, 0.0, 0.0)]

    def gap_ratio(chunks: list[SegmentChunk]) -> float:
        cp_only = _cp_only_time_s(chunks)
        simulated = simulate_segment_time(
            chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
        )
        return (cp_only - simulated) / cp_only

    assert gap_ratio(long_chunks) < gap_ratio(short_chunks)


def test_simulate_segment_time_is_self_consistent_at_the_fixed_point() -> None:
    """Au point fixe, la puissance soutenable qu'implique le temps trouvé doit
    redonner (à la tolérance de convergence près) ce même temps."""
    chunks = [SegmentChunk(0.0, 3000.0, 0.02, 0.0)]  # légère montée

    predicted_time_s = simulate_segment_time(
        chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )

    sustainable_power_w = _CP_WATTS + _W_PRIME_JOULES / predicted_time_s
    speed_ms = cyclist_speed_from_power(
        sustainable_power_w, 0.02, 0.0, _MASS_KG, _CDA_M2, _CRR, STANDARD_AIR_DENSITY_KG_M3
    )
    recomputed_time_s = 3000.0 / speed_ms

    assert recomputed_time_s == pytest.approx(predicted_time_s, abs=0.1)


def test_simulate_segment_time_climbing_takes_longer_than_flat() -> None:
    flat_chunks = [SegmentChunk(0.0, 2000.0, 0.0, 0.0)]
    climb_chunks = [SegmentChunk(0.0, 2000.0, 0.06, 0.0)]

    flat_time_s = simulate_segment_time(
        flat_chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )
    climb_time_s = simulate_segment_time(
        climb_chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )

    assert climb_time_s > flat_time_s


def test_simulate_segment_time_is_consistent_across_chunking_granularity() -> None:
    """Pente constante : découper le même tronçon en 10 morceaux plutôt qu'1
    ne doit rien changer au temps total (vitesse constante à pente constante)."""
    one_chunk = [SegmentChunk(0.0, 1000.0, 0.03, 0.0)]
    ten_chunks = [SegmentChunk(i * 100.0, 100.0, 0.03, 0.0) for i in range(10)]

    time_one = simulate_segment_time(one_chunk, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)
    time_ten = simulate_segment_time(
        ten_chunks, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR
    )

    assert time_one == pytest.approx(time_ten, rel=1e-6)


def test_simulate_segment_time_raises_when_not_converged() -> None:
    chunks = [SegmentChunk(0.0, 1000.0, 0.0, 0.0)]
    with pytest.raises(ValueError, match="convergence"):
        simulate_segment_time(
            chunks,
            _CP_WATTS,
            _W_PRIME_JOULES,
            _MASS_KG,
            _CDA_M2,
            _CRR,
            max_iterations=5,
            convergence_tolerance_s=-1.0,  # tolérance impossible à satisfaire
        )


def test_simulate_segment_time_raises_on_empty_chunks() -> None:
    with pytest.raises(ValueError, match="tronçon"):
        simulate_segment_time([], _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)


# ---- vent (T-27 : effective_headwind_speed_ms, T-15, enfin branché) --------------------------


def test_simulate_segment_time_default_has_no_wind() -> None:
    """Appel sans vent == appel avec wind_speed_ms=0.0 explicite — le
    défaut ne doit rien changer au comportement d'avant T-27."""
    chunk = [SegmentChunk(0.0, 2000.0, 0.0, heading_rad=0.0)]
    baseline = simulate_segment_time(chunk, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)
    explicit_no_wind = simulate_segment_time(
        chunk,
        _CP_WATTS,
        _W_PRIME_JOULES,
        _MASS_KG,
        _CDA_M2,
        _CRR,
        wind_speed_ms=0.0,
        wind_direction_rad=1.7,  # n'importe quelle direction : sans vitesse, aucun effet
    )
    assert explicit_no_wind == pytest.approx(baseline)


def test_simulate_segment_time_headwind_is_slower_than_no_wind() -> None:
    # cap plein nord (0.0), vent qui VIENT du nord (0.0) : face pleine.
    chunk = [SegmentChunk(0.0, 2000.0, 0.0, heading_rad=0.0)]
    no_wind = simulate_segment_time(chunk, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)
    headwind = simulate_segment_time(
        chunk,
        _CP_WATTS,
        _W_PRIME_JOULES,
        _MASS_KG,
        _CDA_M2,
        _CRR,
        wind_speed_ms=5.0,
        wind_direction_rad=0.0,
    )
    assert headwind > no_wind


def test_simulate_segment_time_tailwind_is_faster_than_no_wind() -> None:
    # cap plein nord (0.0), vent qui VIENT du sud (pi) : dos plein.
    chunk = [SegmentChunk(0.0, 2000.0, 0.0, heading_rad=0.0)]
    no_wind = simulate_segment_time(chunk, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)
    tailwind = simulate_segment_time(
        chunk,
        _CP_WATTS,
        _W_PRIME_JOULES,
        _MASS_KG,
        _CDA_M2,
        _CRR,
        wind_speed_ms=5.0,
        wind_direction_rad=math.pi,
    )
    assert tailwind < no_wind


def test_simulate_segment_time_crosswind_is_close_to_no_wind() -> None:
    # cap plein nord, vent qui vient de l'est (pi/2) : travers pur, pas de composante de face/dos.
    chunk = [SegmentChunk(0.0, 2000.0, 0.0, heading_rad=0.0)]
    no_wind = simulate_segment_time(chunk, _CP_WATTS, _W_PRIME_JOULES, _MASS_KG, _CDA_M2, _CRR)
    crosswind = simulate_segment_time(
        chunk,
        _CP_WATTS,
        _W_PRIME_JOULES,
        _MASS_KG,
        _CDA_M2,
        _CRR,
        wind_speed_ms=5.0,
        wind_direction_rad=math.pi / 2,
    )
    assert crosswind == pytest.approx(no_wind, rel=1e-6)


def test_simulate_segment_time_wind_applies_per_chunk_heading() -> None:
    """Deux tronçons de caps opposés sous le même vent absolu : l'un doit
    subir un vent de face, l'autre un vent de dos — pas le même vent
    appliqué en bloc à tout le segment."""
    north_then_south = [
        SegmentChunk(0.0, 1000.0, 0.0, heading_rad=0.0),  # va vers le nord
        SegmentChunk(1000.0, 1000.0, 0.0, heading_rad=math.pi),  # fait demi-tour, va vers le sud
    ]
    # vent qui vient du nord : face sur le 1er tronçon, dos sur le 2e
    time_s = simulate_segment_time(
        north_then_south,
        _CP_WATTS,
        _W_PRIME_JOULES,
        _MASS_KG,
        _CDA_M2,
        _CRR,
        wind_speed_ms=5.0,
        wind_direction_rad=0.0,
    )

    # même vent, mais un aller simple entièrement plein nord (face sur tout le trajet)
    all_north = [SegmentChunk(0.0, 2000.0, 0.0, heading_rad=0.0)]
    all_headwind_time_s = simulate_segment_time(
        all_north,
        _CP_WATTS,
        _W_PRIME_JOULES,
        _MASS_KG,
        _CDA_M2,
        _CRR,
        wind_speed_ms=5.0,
        wind_direction_rad=0.0,
    )

    # le trajet aller-retour (face puis dos) doit être plus rapide que
    # face sur toute la distance : le vent de dos du 2e tronçon compense.
    assert time_s < all_headwind_time_s
