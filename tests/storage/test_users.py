"""Tests de la table DuckDB `users` (T-44a) : un compte par athlète Strava
connecté. Contrairement aux autres tables de storage/, `users` est VIVANTE
(upserts au fil des connexions/rafraîchissements de token), pas reconstruite
depuis du Parquet brut — les tests vérifient donc la persistance et
l'upsert, pas un mapping brut -> ligne.
"""

import duckdb

from segment_predictor.storage.users import ensure_users_table, get_user, upsert_user


def test_ensure_users_table_is_idempotent() -> None:
    """Appeler deux fois ne doit pas planter (IF NOT EXISTS, pas CREATE OR
    REPLACE) ni effacer les comptes déjà insérés entre les deux appels."""
    conn = duckdb.connect(":memory:")
    ensure_users_table(conn)
    upsert_user(conn, 1, "Manu", "F.", "acc-1", "ref-1", 1_700_000_000)

    ensure_users_table(conn)  # deuxième appel, ex. au démarrage d'un autre process

    assert get_user(conn, 1) is not None


def test_upsert_user_creates_a_new_account() -> None:
    conn = duckdb.connect(":memory:")
    ensure_users_table(conn)

    upsert_user(conn, 42, "Ada", "L.", "access-token", "refresh-token", 1_700_000_000)
    user = get_user(conn, 42)

    assert user is not None
    assert user.id == 42
    assert user.firstname == "Ada"
    assert user.lastname == "L."
    assert user.access_token == "access-token"
    assert user.refresh_token == "refresh-token"
    assert user.expires_at == 1_700_000_000


def test_upsert_user_updates_tokens_without_duplicating_the_account() -> None:
    """Reconnexion ou rafraîchissement de token (T-45) : même id, nouveaux
    tokens — un seul compte doit exister, pas deux lignes pour le même id."""
    conn = duckdb.connect(":memory:")
    ensure_users_table(conn)
    upsert_user(conn, 1, "Manu", "F.", "old-access", "old-refresh", 1_000)

    upsert_user(conn, 1, "Manu", "F.", "new-access", "new-refresh", 2_000)

    user = get_user(conn, 1)
    assert user.access_token == "new-access"
    assert user.refresh_token == "new-refresh"
    assert user.expires_at == 2_000
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_get_user_returns_none_for_unknown_id() -> None:
    conn = duckdb.connect(":memory:")
    ensure_users_table(conn)

    assert get_user(conn, 999) is None
