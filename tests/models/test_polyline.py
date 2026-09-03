"""Tests du décodage de polyline encodé (T-32, préparatoire).

Les vecteurs de test à un ou deux octets par valeur sont construits À LA
MAIN suivant l'algorithme documenté (delta -> zigzag -> groupes de 5 bits
-> +63 -> ASCII), pas produits en exécutant decode_polyline lui-même
(ça vérifierait le code contre lui-même, pas contre la spec) — voir le
détail du calcul dans les docstrings de chaque test.
"""

import pytest

from segment_predictor.models.polyline import decode_polyline

# ---- decode_polyline -----------------------------------------------------------------


def test_decode_polyline_empty_string_returns_empty_list() -> None:
    assert decode_polyline("") == []


def test_decode_polyline_single_byte_positive_and_negative_delta() -> None:
    """Premier point = (lat_delta, lng_delta) directement, l'origine
    implicite étant (0, 0). lat_delta=+1 : zigzag = 1<<1 = 2 (pair, donc
    positif au décodage) -> 5 bits '00010', pas de suite -> +63 -> chr(65)
    = 'A'. lng_delta=-2 : zigzag = ~(-2<<1) = ~(-4) = 3 (impair, donc
    négatif au décodage) -> 5 bits '00011' -> +63 -> chr(66) = 'B'.
    Chaque delta tient sur un seul octet (valeur < 32), pas de bit de
    continuation à poser."""
    assert decode_polyline("AB") == [(pytest.approx(0.00001), pytest.approx(-0.00002))]


def test_decode_polyline_multi_byte_chunk_with_continuation_bit() -> None:
    """lat_delta=100 : zigzag = 100<<1 = 200 = 0b11001000. Groupé par 5
    bits en partant du poids faible : chunk0 = 0b01000 = 8 (il reste des
    bits non nuls au-dessus -> bit de continuation 0x20 posé -> 8|32=40
    -> +63 -> chr(103) = 'g'), chunk1 = 0b00110 = 6 (dernier groupe, pas
    de continuation -> +63 -> chr(69) = 'E'). lng_delta=0 -> un seul
    groupe nul -> chr(63) = '?'. Vérifie le ré-assemblage multi-octets
    (poids faible en premier) que le test à un octet ne couvre pas."""
    assert decode_polyline("gE?") == [(pytest.approx(0.001), pytest.approx(0.0))]


def test_decode_polyline_accumulates_deltas_across_points() -> None:
    """Chaque point encode un DELTA par rapport au précédent, pas une
    coordonnée absolue (c'est tout l'intérêt de l'encodage : des petits
    entiers plutôt que des lat/lng complets à chaque point). "AB" (point 1,
    voir test single_byte) suivi d'un second "AB" doit donc donner un
    second point à (2, -4) en unités de 1e-5, pas (1, -2) répété."""
    assert decode_polyline("ABAB") == [
        (pytest.approx(0.00001), pytest.approx(-0.00002)),
        (pytest.approx(0.00002), pytest.approx(-0.00004)),
    ]


def test_decode_polyline_raises_on_truncated_input() -> None:
    """'g' seul (voir test multi_byte) porte un bit de continuation posé
    (>= 0x20 une fois l'offset -63 retiré) : le décodeur attend un octet
    de plus qui n'existe pas. Une erreur explicite, pas un résultat
    tronqué silencieux (règle du projet : pas de donnée inventée)."""
    with pytest.raises(ValueError, match="tronqué"):
        decode_polyline("g")


def test_decode_polyline_matches_a_real_strava_segment() -> None:
    """Regression : polyline réel du segment HBFH (id 31159007, T-31/T-32),
    tel que stocké dans `segments.polyline` — pas juste des vecteurs
    construits à la main. Le premier point décodé doit tomber sur
    segments.start_lat/start_lng déjà en base pour ce même segment
    (47.525849, 7.478863) : deux champs extraits indépendamment du même
    JSON Strava (l'un tel quel, l'autre décodé ici), qui doivent
    forcément coïncider si le décodage est correct."""
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
    first_point = decode_polyline(encoded)[0]
    assert first_point == pytest.approx((47.525849, 7.478863))
