CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE model_version (
    version_id BIGSERIAL PRIMARY KEY,
    date_created DATE NOT NULL DEFAULT CURRENT_DATE,
    checkpoint_path VARCHAR(512) NOT NULL,
    metrics TEXT
);

CREATE TABLE episode (
    episode_id BIGSERIAL PRIMARY KEY,
    version_id BIGINT NOT NULL REFERENCES model_version(version_id),
    start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_time TIMESTAMPTZ DEFAULT NULL
);

CREATE TABLE battle_state (
    state_id BIGSERIAL PRIMARY KEY,
    state_hash BIGINT NOT NULL UNIQUE,
    raw_state JSONB NOT NULL,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE battle_state_embedding (
    embedding_id BIGSERIAL PRIMARY KEY,
    state_id BIGINT NOT NULL REFERENCES battle_state(state_id),
    embedding VECTOR(1024) NOT NULL
);


CREATE TABLE transition (
    transition_id BIGSERIAL PRIMARY KEY,
    step_index SMALLINT NOT NULL,
    episode_id BIGINT NOT NULL REFERENCES episode(episode_id),
    state_id BIGINT NOT NULL REFERENCES battle_state(state_id),
    new_state_id BIGINT NOT NULL REFERENCES battle_state(state_id),
    action SMALLINT NOT NULL,
    reward DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    terminal BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    td_error DOUBLE PRECISION,
    UNIQUE (episode_id, step_index)
);

CREATE INDEX ON transition (episode_id);
CREATE INDEX ON transition (state_id);