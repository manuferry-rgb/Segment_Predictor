"""Facteur de réduction aérodynamique par drafting (T-19).

Fonction pure — aucun I/O. Le ratio retourné se multiplie directement au
CdA solo calibré (T-17) avant de simuler (T-13) : le drafting ne change
que le terme aéro de l'équation de puissance (physics.py), pas la masse
ni le roulement.

Deux régimes physiquement différents, pas un seul modèle continu :

- "roue collée" / "un mètre" : un seul rider derrière un seul leader
  (paceline), modélisé par la formule empirique d'Olds (1995) — un fit
  quadratique sur les mesures en soufflerie de Kyle (1979), en fonction
  de l'écart entre roues. Voir olds_draft_ratio.
- "groupe" : peloton bien formé, où la réduction vient du blindage
  latéral ET arrière de plusieurs coureurs à la fois. La formule d'Olds
  ne modélise qu'un seul leader devant — elle ne s'applique PAS ici,
  même avec un gap_m minuscule. Valeur reprise d'une étude CFD/soufflerie
  dédiée au peloton (Blocken et al. 2018, "Aerodynamic drag in cycling
  pelotons: New insights by CFD simulation and wind tunnel testing"),
  qui mesure une traînée réduite à 5-10% de celle d'un rider isolé au
  milieu d'un peloton de 121 coureurs. Voir GROUP_DRAFT_RATIO.
"""

# Coefficients du fit quadratique d'Olds (1995) sur les données de Kyle
# (1979) : ratio de traînée (drafteur / rider solo) en fonction de
# l'écart roue-à-roue en mètres.
_OLDS_A = 0.62
_OLDS_B = -0.0104
_OLDS_C = 0.0452
# Plage de validité du fit. Au-delà, le polynôme n'a plus de sens
# physique : ratio(3.0) ≈ 0.996 (bénéfice déjà quasi nul), et il continue
# à augmenter au-delà de 1.0 pour gap_m plus grand — plus de traînée
# qu'en solo, ce que la physique réelle ne prédit pas (le bénéfice
# tendrait simplement vers 0, pas au-delà).
OLDS_MAX_GAP_M = 3.0

# Réduction observée au milieu d'un peloton bien formé (borne basse de la
# fourchette 5-10% de la traînée d'un rider isolé mesurée par Blocken et
# al. 2018) — volontairement le côté le moins optimiste de la fourchette,
# pas la meilleure valeur possible.
GROUP_DRAFT_RATIO = 0.10

# Écarts (m) associés aux presets "paceline" du ticket T-19. 0.2m plutôt
# que 0.0m pour "roue collée" : le minimum du fit d'Olds tombe vers
# 0.115m (sqrt du vertex de la parabole), la formule est quasi plate sur
# tout [0, 0.5]m — le choix précis dans cette zone ne change quasiment
# rien au résultat (voir test_olds_draft_ratio_matches_known_values).
_ROUE_COLLEE_GAP_M = 0.2
_UN_METRE_GAP_M = 1.0


def olds_draft_ratio(gap_m: float) -> float:
    """Ratio de traînée aéro (drafteur / rider solo) selon Olds (1995), en
    fonction de l'écart roue-à-roue `gap_m`. 1.0 = aucun bénéfice ; 0.62 =
    38% d'économie, le maximum du fit, obtenu très près de la roue.

    Valide seulement sur [0, OLDS_MAX_GAP_M] — ValueError explicite en
    dehors plutôt qu'un ratio silencieusement faux.
    """
    if not 0.0 <= gap_m <= OLDS_MAX_GAP_M:
        raise ValueError(
            f"gap_m={gap_m} hors de la plage de validité du fit d'Olds "
            f"[0, {OLDS_MAX_GAP_M}] m — au-delà, le polynôme n'a plus de sens physique"
        )
    return _OLDS_A + _OLDS_B * gap_m + _OLDS_C * gap_m**2


def draft_ratio_for_preset(preset: str) -> float:
    """Facteur multiplicatif à appliquer au CdA solo, pour les 4 scénarios
    du ticket T-19 : "solo", "roue_collee", "un_metre", "groupe".

    "roue_collee" et "un_metre" passent par olds_draft_ratio (paceline à
    un seul leader) ; "groupe" utilise GROUP_DRAFT_RATIO (peloton, source
    différente, cf. docstring du module) ; "solo" vaut 1.0 par définition
    (rien à modéliser).
    """
    if preset == "solo":
        return 1.0
    if preset == "roue_collee":
        return olds_draft_ratio(_ROUE_COLLEE_GAP_M)
    if preset == "un_metre":
        return olds_draft_ratio(_UN_METRE_GAP_M)
    if preset == "groupe":
        return GROUP_DRAFT_RATIO
    raise ValueError(
        f"scénario de draft inconnu : {preset!r} "
        "(attendu : 'solo', 'roue_collee', 'un_metre', 'groupe')"
    )
