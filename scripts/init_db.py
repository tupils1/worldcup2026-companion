#!/usr/bin/env python
"""Initialize the SQLite database and seed the 48-team roster.

Usage:
    python scripts/init_db.py            # use default data/worldcup.db
    python scripts/init_db.py --reset    # drop & recreate (DESTRUCTIVE)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn, init_db
from worldcup.db.seed import seed_teams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite path")
    ap.add_argument("--reset", action="store_true", help="Delete existing DB file first")
    args = ap.parse_args()

    db_path = Path(args.db)
    if args.reset and db_path.exists():
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()
        print(f"🗑  removed {db_path}")

    init_db(db_path)
    print(f"✅ schema initialized at {db_path}")

    n = seed_teams(db_path=db_path)
    print(f"✅ seeded {n} teams from configs/teams.yaml")

    # quick sanity dump
    conn = get_conn(db_path)
    try:
        n_total = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
        per_conf = conn.execute(
            "SELECT confederation, COUNT(*) FROM teams "
            "WHERE in_worldcup_2026=1 GROUP BY confederation ORDER BY 1"
        ).fetchall()
        per_group = conn.execute(
            "SELECT group_letter, COUNT(*) FROM teams "
            "WHERE in_worldcup_2026=1 GROUP BY group_letter ORDER BY 1"
        ).fetchall()
        print(f"\n   teams in DB: {n_total}")
        print("   by confederation:", {r[0]: r[1] for r in per_conf})
        print("   by group:        ", {r[0]: r[1] for r in per_group})
    finally:
        conn.close()


if __name__ == "__main__":
    main()
