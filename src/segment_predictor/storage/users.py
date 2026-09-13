"""Table DuckDB `users` (T-44a) : un compte par athlète Strava connecté.

Contrairement aux autres tables de storage/ (reconstruites en entier
depuis du Parquet brut à chaque ingest, ex. build_activities_table),
`users` est une table VIVANTE : un compte apparaît au moment où quelqu'un
se connecte (T-45, OAuth) et ses tokens changent au fil des
rafraîchissements — pas de notion de "reconstruction depuis du brut" ici,
juste des upserts classiques au fil du temps.

`id` = l'id athlète Strava lui-même, pas un id interne séparé : cohérent
avec le reste du projet (segments.id, activities.id sont déjà des ids
Strava bruts), et Strava en garantit l'unicité — inventer un second id
n'ajouterait qu'une indirection sans bénéfice.
"""

from dataclasses import dataclass

import duckdb


def ensure_users_table(conn: duckdb.DuckDBPyConnection) -> None:
    """(Re)crée le schéma de `users` s'il n'existe pas déjà.

    IF NOT EXISTS, pas CREATE OR REPLACE comme les autres tables de
    storage/ : celles-ci sont rebâties depuis du Parquet à chaque ingest
    (rien à perdre), alors qu'écraser `users` effacerait les comptes déjà
    connectés — cette table n'a pas de "source brute" à relire pour les
    reconstruire.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGINT PRIMARY KEY,
            firstname VARCHAR,
            lastname VARCHAR,
            access_token VARCHAR NOT NULL,
            refresh_token VARCHAR NOT NULL,
            expires_at BIGINT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        )
        """
    )


@dataclass(frozen=True)
class User:
    id: int
    firstname: str | None
    lastname: str | None
    access_token: str
    refresh_token: str
    # timestamp unix (secondes), même convention que TokenState (ingest/strava_auth)
    expires_at: int


def upsert_user(
    conn: duckdb.DuckDBPyConnection,
    id: int,
    firstname: str | None,
    lastname: str | None,
    access_token: str,
    refresh_token: str,
    expires_at: int,
) -> None:
    """Crée le compte s'il n'existe pas, ou met à jour ses tokens/son nom
    sinon (reconnexion ou rafraîchissement de token, T-45) — `ON CONFLICT`
    garantit qu'un même `id` ne produit jamais deux lignes.
    """
    conn.execute(
        """
        INSERT INTO users (id, firstname, lastname, access_token, refresh_token, expires_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            firstname = excluded.firstname,
            lastname = excluded.lastname,
            access_token = excluded.access_token,
            refresh_token = excluded.refresh_token,
            expires_at = excluded.expires_at
        """,
        [id, firstname, lastname, access_token, refresh_token, expires_at],
    )


def get_user(conn: duckdb.DuckDBPyConnection, user_id: int) -> User | None:
    row = conn.execute(
        "SELECT id, firstname, lastname, access_token, refresh_token, expires_at "
        "FROM users WHERE id = ?",
        [user_id],
    ).fetchone()
    if row is None:
        return None
    return User(*row)
