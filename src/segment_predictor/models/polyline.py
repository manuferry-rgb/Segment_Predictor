"""Décodage du polyline encodé Strava/Google (T-32, préparatoire).

Fonction pure — aucun I/O, prend une chaîne, retourne des structures.
`storage/segments.py` (T-31) stocke déjà `segments.polyline` tel quel ;
ce module en fait le premier usage réel : le décoder en points GPS pour,
à terme, calculer un cap par tronçon (au lieu de l'unique cap moyen
start->end de `heading_rad`) et projeter le vent correctement le long
d'un segment qui tourne ou boucle — pas encore branché ici, c'est la
suite du ticket T-32.

Format "Google Encoded Polyline Algorithm" : chaque point encode un
DELTA (lat, lng) par rapport au point précédent (pas une coordonnée
absolue), en unités de 1e-5 degré. Chaque delta passe par trois étapes :

1. Zigzag : les entiers signés deviennent des entiers non signés en
   alternant positifs/négatifs (0, -1, 1, -2, 2, ... -> 0, 1, 2, 3, 4,
   ...), pour que la représentation binaire d'un petit delta négatif
   reste courte (le complément à deux d'un petit négatif est long).
2. Découpage en groupes de 5 bits, poids faible en premier, chaque
   groupe sauf le dernier ayant son bit de poids fort (0x20) posé pour
   signaler "il reste un groupe" — un varint, essentiellement.
3. Chaque groupe de 5 bits + 63 devient un caractère ASCII imprimable.

Voir les docstrings de tests/models/test_polyline.py pour un calcul à la
main de chacune de ces étapes sur un exemple court.
"""

# Offset ajouté à chaque groupe de 5 bits avant conversion en caractère
# ASCII — choisi par le format pour que la plage encodée tombe dans les
# caractères imprimables (33 à 126), pas de raison physique.
_ASCII_OFFSET = 63
# Bit de poids fort d'un groupe de 5 bits : posé si un autre groupe suit.
_CONTINUATION_BIT = 0x20
_VALUE_BITS_MASK = 0x1F
# Précision standard du format (1e-5 degré par unité) — celle utilisée
# par Strava, vérifiée sur un polyline réel (voir le test de régression).
_DEFAULT_PRECISION = 5


def decode_polyline(encoded: str, precision: int = _DEFAULT_PRECISION) -> list[tuple[float, float]]:
    """Décode `encoded` en liste de points (lat, lng), en degrés décimaux.

    Lève `ValueError` si la chaîne se termine au milieu d'un point (bit
    de continuation posé sur le dernier caractère lu) plutôt que de
    renvoyer un point tronqué silencieusement.
    """
    if not encoded:
        return []

    factor = 10**precision
    lat = 0
    lng = 0
    coordinates: list[tuple[float, float]] = []
    index = 0
    length = len(encoded)

    while index < length:
        deltas = []
        for _ in range(2):  # dans l'ordre : delta latitude, puis delta longitude
            shift = 0
            value = 0
            while True:
                if index >= length:
                    raise ValueError(
                        "polyline tronqué : fin de chaîne atteinte au milieu d'un point "
                        f"(index {index} sur {length})"
                    )
                byte = ord(encoded[index]) - _ASCII_OFFSET
                index += 1
                value |= (byte & _VALUE_BITS_MASK) << shift
                shift += 5
                if not byte & _CONTINUATION_BIT:
                    break
            # Zigzag inverse : bit de poids faible = signe (impair = négatif).
            delta = ~(value >> 1) if value & 1 else value >> 1
            deltas.append(delta)

        lat += deltas[0]
        lng += deltas[1]
        coordinates.append((lat / factor, lng / factor))

    return coordinates
