from typing import Any
from datetime import date, datetime

import psycopg2
from psycopg2 import extras
from psycopg2.extensions import connection
from pgvector.psycopg2 import register_vector

import json
import hashlib


EMBEDDING_DIM: int = 1024


def get_connection(dbname: str, user: str, password: str, host: str, port: int) -> connection:
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )

    register_vector(conn)

    return conn


def compute_state_hash(state_dict: dict[str, dict[str, Any]]) -> int:
    raw: str = json.dumps(state_dict, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(raw.encode(), digest_size=8).digest()
    hash = int.from_bytes(digest)

    return hash


def insert_battle(conn: connection, state_dict: dict[str, dict[str, Any]]) -> int:
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


def insert_transition(conn: connection, step_index: int, episode_id: int, state_id: int, new_state_id: int, action: int, reward: float, terminal: bool) -> int:
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


def insert_episode(conn: connection, version_id: int, start_time: datetime, end_time: datetime | None) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO episode (
                version_id,
                start_time,
                end_time
            )
            VALUES (%s, %s, %s)

            RETURNING episode_id
            """,
            (
                version_id,
                start_time,
                end_time
            )
        )

        (episode_id,) = cur.fetchone()
    
    return episode_id


def insert_model(conn: connection, date_created: date, checkpoint_path: str, metrics: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_version (
                date_created,
                checkpoint_path,
                metrics
            )
            VALUES (%s, %s, %s)

            RETURNING version_id
            """,
            (
                date_created,
                checkpoint_path,
                metrics
            )
        )

        (version_id,) = cur.fetchone()
    
    return version_id


def insert_battle_embedding(conn: connection, state_id: int, embedding: list[float]) -> int:
    if len(embedding) != EMBEDDING_DIM:
        return -1

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO battle_state_embedding (
                embedding
            )
            VALUES (%s)

            RETURNING embedding_id
            """,
            (
                state_id,
                embedding
            )
        )

        (embedding_id,) = cur.fetchone()
    
    return embedding_id