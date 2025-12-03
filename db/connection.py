from typing import Any

import psycopg2
from psycopg2 import extras
import json
import hashlib


def get_connection(dbname: str, user: str, password: str, host: str, port: int) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    return conn


def compute_state_hash(state_dict: dict[str, dict[str, Any]]) -> int:
    raw: str = json.dumps(state_dict, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(raw.encode(), digest_size=8).digest()
    hash = int.from_bytes(digest)

    return hash


def insert_state(conn: psycopg2.extensions.connection, state_dict: dict[str, dict[str, Any]]) -> int:
    state_hash: int = compute_state_hash(state_dict)
    state_jsonb: extras.Json = extras.Json(state_dict)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO battle_state (
                state_hash,
                raw_state
            )
            VALUES (%s, %s)

            RETURNING state_id
            """,
            (
                state_hash,
                state_jsonb,
            )
        )

        (state_id,) = cur.fetchone()
    
    return state_id


def insert_transition(conn: psycopg2.extensions.connection, step_index: int, episode_id: int, state_id: int, new_state_id: int, action: int, reward: float, terminal: bool) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO transition (
                step_index,
                episode_id,
                state_id,
                new_state_id,
                action,
                reward,
                terminal
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)

            RETURNING transition_id
            """,
            (
                step_index,
                episode_id,
                state_id,
                new_state_id,
                action,
                reward,
                terminal
            )
        )

        (transition_id,) = cur.fetchone()
    
    return transition_id