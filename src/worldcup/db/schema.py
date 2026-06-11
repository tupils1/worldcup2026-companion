"""SQLite schema + connection helpers.

One database file, raw SQL via sqlite3 module — no ORM. Idempotent DDL.

Tables:
    teams                    — 48 WC teams + historical national teams from ingest sources
    team_code_map            — (source, source_code) -> teams.code mapping (ES->ESP etc.)
    team_ratings             — Elo / FIFA-points time series
    matches                  — international matches with score + xG when available
    players                  — player master data
    player_market_values     — Transfermarkt valuation history
    player_match_stats       — per-match player line (xG, xA, minutes, rating)
    injuries                 — current injury / suspension state
    odds                     — bookmaker price snapshots
    ingest_runs              — bookkeeping for which scrapers ran when
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "worldcup.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS teams (
    code              TEXT PRIMARY KEY,        -- FIFA 3-letter (USA, BRA, ENG, ...)
    name              TEXT NOT NULL,
    confederation     TEXT,                    -- AFC / CAF / CONCACAF / CONMEBOL / OFC / UEFA
    in_worldcup_2026  INTEGER NOT NULL DEFAULT 0,
    pot               INTEGER,                 -- 1..4 if in WC 2026
    group_letter      TEXT,                    -- A..L if in WC 2026
    qualified_via     TEXT,
    role              TEXT,                    -- 'host' or NULL
    notes             TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS team_code_map (
    source       TEXT NOT NULL,                -- 'eloratings', 'fbref', 'transfermarkt', 'api_football', ...
    source_code  TEXT NOT NULL,
    fifa_code    TEXT NOT NULL,
    PRIMARY KEY (source, source_code),
    FOREIGN KEY (fifa_code) REFERENCES teams(code)
);

CREATE TABLE IF NOT EXISTS team_ratings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    team_code    TEXT NOT NULL,
    rating_type  TEXT NOT NULL,                -- 'elo', 'fifa_rank', 'fifa_points', 'spi_off', 'spi_def', ...
    value        REAL NOT NULL,
    as_of_date   TEXT NOT NULL,                -- YYYY-MM-DD
    source       TEXT NOT NULL,                -- 'eloratings.net', 'fifa.com', ...
    FOREIGN KEY (team_code) REFERENCES teams(code),
    UNIQUE (team_code, rating_type, as_of_date, source)
);
CREATE INDEX IF NOT EXISTS idx_ratings_team_date ON team_ratings(team_code, as_of_date);

CREATE TABLE IF NOT EXISTS matches (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    fbref_id               TEXT UNIQUE,
    api_football_id        INTEGER UNIQUE,
    home_code              TEXT NOT NULL,
    away_code              TEXT NOT NULL,
    match_date             TEXT NOT NULL,      -- ISO 8601 kickoff (UTC if known)
    competition            TEXT,               -- 'World Cup 2026', 'Nations League', 'Friendly', ...
    stage                  TEXT,               -- 'Group A', 'Round of 32', ...
    venue                  TEXT,
    neutral_venue          INTEGER NOT NULL DEFAULT 0,
    home_score             INTEGER,
    away_score             INTEGER,
    home_xg                REAL,
    away_xg                REAL,
    home_shots             INTEGER,
    away_shots             INTEGER,
    home_shots_on_target   INTEGER,
    away_shots_on_target   INTEGER,
    home_possession_pct    REAL,
    finished               INTEGER NOT NULL DEFAULT 0,
    extra_time             INTEGER NOT NULL DEFAULT 0,
    penalties              INTEGER NOT NULL DEFAULT 0,
    home_pens              INTEGER,
    away_pens              INTEGER,
    source                 TEXT,
    FOREIGN KEY (home_code) REFERENCES teams(code),
    FOREIGN KEY (away_code) REFERENCES teams(code),
    UNIQUE (match_date, home_code, away_code)
);
CREATE INDEX IF NOT EXISTS idx_matches_date    ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_home    ON matches(home_code);
CREATE INDEX IF NOT EXISTS idx_matches_away    ON matches(away_code);

CREATE TABLE IF NOT EXISTS players (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    transfermarkt_id    INTEGER UNIQUE,
    fbref_id            TEXT UNIQUE,
    api_football_id     INTEGER UNIQUE,
    name                TEXT NOT NULL,
    full_name           TEXT,
    date_of_birth       TEXT,                  -- YYYY-MM-DD
    height_cm           INTEGER,
    foot                TEXT,                  -- 'L', 'R', 'B'
    position            TEXT,                  -- 'GK', 'CB', 'CM', 'ST', ...
    nationality_code    TEXT,
    current_club        TEXT,
    FOREIGN KEY (nationality_code) REFERENCES teams(code)
);
CREATE INDEX IF NOT EXISTS idx_players_nat ON players(nationality_code);

CREATE TABLE IF NOT EXISTS player_market_values (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id    INTEGER NOT NULL,
    value_eur    INTEGER NOT NULL,
    as_of_date   TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES players(id),
    UNIQUE (player_id, as_of_date)
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id            INTEGER NOT NULL,
    player_id           INTEGER NOT NULL,
    team_code           TEXT NOT NULL,
    minutes             INTEGER,
    is_starter          INTEGER NOT NULL DEFAULT 0,
    goals               INTEGER NOT NULL DEFAULT 0,
    assists             INTEGER NOT NULL DEFAULT 0,
    shots               INTEGER,
    shots_on_target     INTEGER,
    xg                  REAL,
    xa                  REAL,
    key_passes          INTEGER,
    sofascore_rating    REAL,
    yellow_cards        INTEGER NOT NULL DEFAULT 0,
    red_cards           INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (match_id)  REFERENCES matches(id),
    FOREIGN KEY (player_id) REFERENCES players(id),
    FOREIGN KEY (team_code) REFERENCES teams(code),
    UNIQUE (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS injuries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id         INTEGER NOT NULL,
    team_code         TEXT NOT NULL,
    status            TEXT NOT NULL,           -- 'injured' | 'suspended' | 'doubtful' | 'recovered'
    detail            TEXT,
    expected_return   TEXT,                    -- YYYY-MM-DD or NULL
    source            TEXT NOT NULL,
    captured_at       TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (player_id) REFERENCES players(id),
    FOREIGN KEY (team_code) REFERENCES teams(code)
);
CREATE INDEX IF NOT EXISTS idx_inj_team ON injuries(team_code, captured_at);

CREATE TABLE IF NOT EXISTS odds (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id     INTEGER,
    -- For pre-match outright / futures markets that aren't tied to a match:
    market_scope TEXT NOT NULL,                -- 'match' | 'outright' | 'player'
    subject_code TEXT,                         -- team_code for outrights, player_id-as-text for props
    bookmaker    TEXT NOT NULL,                -- 'pinnacle' | 'bet365' | 'crown' | 'sbobet' | ...
    market       TEXT NOT NULL,                -- '1X2' | 'AH' | 'OU' | 'winner' | 'top_scorer' | 'reaches_qf' ...
    selection    TEXT NOT NULL,                -- 'home'|'away'|'draw'|'over'|'under'|team_code|player_id
    line         REAL,                         -- AH handicap, OU total, NULL otherwise
    price        REAL NOT NULL,                -- decimal odds
    captured_at  TEXT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
CREATE INDEX IF NOT EXISTS idx_odds_match    ON odds(match_id, market, captured_at);
CREATE INDEX IF NOT EXISTS idx_odds_outright ON odds(market_scope, market, subject_code, captured_at);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',   -- 'running'|'success'|'error'
    rows_written    INTEGER,
    error_message   TEXT
);
"""


def get_conn(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with sane defaults (FKs on, WAL)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)  # autocommit; explicit BEGIN per txn
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(path: Path | str = DEFAULT_DB_PATH) -> Path:
    """Create the schema if missing. Idempotent."""
    conn = get_conn(path)
    try:
        conn.executescript(SCHEMA_SQL)
    finally:
        conn.close()
    return Path(path)
