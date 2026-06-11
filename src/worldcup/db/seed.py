"""Seed teams table from configs/teams.yaml."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def seed_teams(
    yaml_path: Path | str = CONFIGS_DIR / "teams.yaml",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Load `teams.yaml` into the `teams` table. Returns rows written."""
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    groups = cfg["groups"]
    pots = cfg["pots"]
    teams = cfg["teams"]

    # Reverse-index: code → (group_letter, pot)
    code_to_group = {
        code: letter for letter, gdata in groups.items() for code in gdata["teams"]
    }
    code_to_pot = {code: pot for pot, codes in pots.items() for code in codes}

    rows = []
    for code, meta in teams.items():
        rows.append(
            (
                code,
                meta["name"],
                meta.get("confederation"),
                1,                              # in_worldcup_2026
                code_to_pot.get(code),
                code_to_group.get(code),
                meta.get("qualified_via"),
                meta.get("role"),
                meta.get("notes"),
            )
        )

    conn = get_conn(db_path)
    try:
        conn.execute("BEGIN")
        conn.executemany(
            """
            INSERT INTO teams
                (code, name, confederation, in_worldcup_2026, pot, group_letter,
                 qualified_via, role, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                name             = excluded.name,
                confederation    = excluded.confederation,
                in_worldcup_2026 = excluded.in_worldcup_2026,
                pot              = excluded.pot,
                group_letter     = excluded.group_letter,
                qualified_via    = excluded.qualified_via,
                role             = excluded.role,
                notes            = excluded.notes
            """,
            rows,
        )
        conn.execute("COMMIT")
    except sqlite3.Error:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    return len(rows)
