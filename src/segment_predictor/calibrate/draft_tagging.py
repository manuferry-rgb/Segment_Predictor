"""Génère le CSV de tri pour le tag manuel des efforts en groupe (T-16).

⚠️ Le temps prédit ici vient d'un modèle NON CALIBRÉ : CdA/Crr génériques
(la calibration réelle, T-17, dépend justement du tri qu'on prépare ici —
boucle assumée), et une approximation supplémentaire (le segment est
traité comme un seul tronçon à pente moyenne constante, pas son profil
réel tronçon par tronçon, qu'on n'a pas au niveau segment). L'écart
(prédit − réel) est une AIDE AU TRI pour repérer les efforts à regarder
en premier, pas une détection : un grand écart peut venir d'un draft,
d'un bon jour, ou juste des approximations du modèle.

Le CSV produit est une donnée d'entrée versionnée, pas un artefact
jetable : `draft_status` est la seule colonne jamais écrasée d'un
lancement à l'autre (fusion par `effort_id`). Tout le reste est recalculé
à chaque appel pour refléter "le modèle actuel".
"""

import csv
from collections.abc import Iterable
from pathlib import Path

import duckdb
import numpy as np

from segment_predictor.models.power import (
    CriticalPowerFit,
    fit_critical_power,
    mean_maximal_power_curve,
    resample_to_uniform_seconds,
)
from segment_predictor.models.segment import SegmentChunk, simulate_segment_time

DEFAULT_CP_FIT_DURATIONS_S = (180, 240, 300, 420, 600, 900, 1200)
DEFAULT_DRAFT_STATUS = "unknown"
# ';' plutôt que ',' : Excel en locale française écrit (et attend, au
# double-clic) des CSV séparés par ';' — un ',' obligerait à repasser par
# l'assistant d'import à chaque ouverture. load_existing_annotations
# détecte le séparateur au lieu de le supposer, donc un fichier ',' déjà
# existant (ancien commit, ou un autre tableur) reste lisible.
CSV_DELIMITER = ";"
CSV_FIELDNAMES = (
    "effort_id",
    "date",
    "activity_name",
    "segment_id",
    "segment_name",
    "actual_time_s",
    "predicted_time_s",
    "gap_s",
    "draft_status",
)


def compute_aggregate_mmp_curve(
    conn: duckdb.DuckDBPyConnection, durations_s: Iterable[int]
) -> dict[int, float]:
    """Meilleure puissance moyenne jamais atteinte, par durée, tous mes
    efforts Ride/VirtualRide à capteur de puissance confondus — même
    calcul que la vérification manuelle faite en T-09/T-13, formalisé ici
    pour être réutilisable.
    """
    durations_s = list(durations_s)
    activity_ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM activities "
            "WHERE type IN ('Ride', 'VirtualRide') AND device_watts = true"
        ).fetchall()
    ]

    best_mmp = dict.fromkeys(durations_s, float("-inf"))
    for activity_id in activity_ids:
        rows = conn.execute(
            "SELECT t_s, watts FROM streams WHERE activity_id = ? ORDER BY sample_index",
            [activity_id],
        ).fetchall()
        if not rows:
            continue
        t_s = np.array([r[0] for r in rows])
        watts = np.array([r[1] if r[1] is not None else np.nan for r in rows])
        try:
            uniform = resample_to_uniform_seconds(t_s, watts)
        except ValueError:
            continue  # stream malformé (ex. timestamp dupliqué, cf T-09) : on saute cette activité

        curve = mean_maximal_power_curve(uniform, durations_s)
        for duration_s, value in curve.items():
            if not np.isnan(value) and value > best_mmp[duration_s]:
                best_mmp[duration_s] = value

    return {d: (v if v != float("-inf") else float("nan")) for d, v in best_mmp.items()}


def fit_current_cp(
    conn: duckdb.DuckDBPyConnection, durations_s: Iterable[int] = DEFAULT_CP_FIT_DURATIONS_S
) -> CriticalPowerFit:
    """Ajuste CP/W' sur la meilleure courbe MMP jamais atteinte à date —
    "le modèle actuel" : recalculé à chaque appel, jamais une valeur figée
    d'une session précédente.
    """
    durations_s = list(durations_s)
    curve = compute_aggregate_mmp_curve(conn, durations_s)
    durations_arr = np.array(durations_s, dtype=float)
    mmp_arr = np.array([curve[d] for d in durations_s], dtype=float)
    return fit_critical_power(durations_arr, mmp_arr)


def predict_segment_time_s(
    distance_m: float,
    average_grade: float,
    cp_fit: CriticalPowerFit,
    mass_kg: float,
    cda_m2: float,
    crr: float,
) -> float:
    """Temps prédit pour le segment, approximé comme UN SEUL tronçon à pente
    moyenne constante — pas le profil réel, qu'on n'a pas au niveau segment
    (le relier à un passage précis demanderait start_index/end_index, pas
    extraits en T-07b). Sans vent (T-15 n'est pas encore branché ici).
    """
    chunk = SegmentChunk(
        start_distance_m=0.0, length_m=distance_m, grade=average_grade, heading_rad=0.0
    )
    return simulate_segment_time(
        [chunk], cp_fit.cp_watts, cp_fit.w_prime_joules, mass_kg, cda_m2, crr
    )


def load_existing_annotations(csv_path: Path) -> dict[int, str]:
    """draft_status déjà annoté par effort_id — la seule chose qu'on ne réécrit jamais.

    Détecte le séparateur (`,` ou `;`) plutôt que de supposer une virgule :
    Excel en locale française réenregistre les CSV avec `;`, rencontré en
    conditions réelles. On ne va pas demander à l'utilisateur de lutter
    contre son tableur à chaque fois qu'il tague une ligne.
    """
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel  # échantillon trop court/ambigu : virgule par défaut
        return {
            int(row["effort_id"]): row["draft_status"] for row in csv.DictReader(f, dialect=dialect)
        }


def generate_draft_tagging_csv(
    conn: duckdb.DuckDBPyConnection,
    csv_path: Path,
    mass_kg: float,
    cda_m2: float = 0.32,
    crr: float = 0.005,
    cp_fit: CriticalPowerFit | None = None,
) -> None:
    """(Re)génère le CSV de tri, trié par écart décroissant (les efforts les
    plus rapides que le modèle prédit en premier — le signal à regarder pour
    repérer un drafting possible).

    `cp_fit` est injectable pour les tests ; en usage normal, laissé à None
    pour être recalculé depuis `conn` à chaque appel (voir fit_current_cp).
    """
    if cp_fit is None:
        cp_fit = fit_current_cp(conn)

    existing_annotations = load_existing_annotations(csv_path)

    efforts = conn.execute(
        "SELECT se.id, se.start_date, a.name, se.segment_id, s.name, se.elapsed_time_s, "
        "s.distance_m, s.average_grade "
        "FROM segment_efforts se "
        "JOIN segments s ON s.id = se.segment_id "
        "JOIN activities a ON a.id = se.activity_id"
    ).fetchall()

    rows = []
    for (
        effort_id,
        start_date,
        activity_name,
        segment_id,
        segment_name,
        actual_time_s,
        distance_m,
        average_grade,
    ) in efforts:
        try:
            predicted_time_s = predict_segment_time_s(
                distance_m, average_grade, cp_fit, mass_kg, cda_m2, crr
            )
            gap_s = predicted_time_s - actual_time_s
        except ValueError:
            # simulate_segment_time n'a pas convergé pour ce cas précis :
            # on le signale plutôt que d'inventer un temps ou de planter
            # tout le CSV pour une seule ligne.
            predicted_time_s = None
            gap_s = None

        rows.append(
            {
                "effort_id": effort_id,
                "date": start_date.isoformat() if hasattr(start_date, "isoformat") else start_date,
                "activity_name": activity_name,
                "segment_id": segment_id,
                "segment_name": segment_name,
                "actual_time_s": actual_time_s,
                "predicted_time_s": predicted_time_s,
                "gap_s": gap_s,
                "draft_status": existing_annotations.get(effort_id, DEFAULT_DRAFT_STATUS),
            }
        )

    rows.sort(key=lambda row: (row["gap_s"] is None, -(row["gap_s"] or 0)))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, delimiter=CSV_DELIMITER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
