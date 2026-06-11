"""eloratings.net (World Football Elo Ratings) ingester.

The site is JS-rendered but exposes a static TSV at
https://www.eloratings.net/World.tsv — 244 rows, 31 columns. We only consume
columns 0 (rank), 2 (2-letter code), 3 (Elo rating).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import yaml

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

ELORATINGS_TSV = "https://www.eloratings.net/World.tsv"
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
CODE_MAP_PATH = CONFIG_DIR / "code_maps" / "eloratings_to_fifa.yaml"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) worldcup-research"


def fetch_world_tsv(client: httpx.Client | None = None) -> str:
    """Return the raw TSV text of current world Elo ratings."""
    if client is None:
        with httpx.Client(headers={"User-Agent": UA}, timeout=20) as c:
            r = c.get(ELORATINGS_TSV)
    else:
        r = client.get(ELORATINGS_TSV)
    r.raise_for_status()
    return r.text


def parse_world_tsv(text: str) -> list[dict[str, Any]]:
    """Parse World.tsv. Columns: 0=rank, 2=eloratings_code, 3=Elo rating."""
    rows = []
    for ln in text.strip().split("\n"):
        f = ln.split("\t")
        if len(f) < 4:
            continue
        try:
            rows.append({"rank": int(f[0]), "src_code": f[2], "elo": float(f[3])})
        except ValueError:
            continue
    return rows


def load_code_map() -> dict[str, str]:
    return yaml.safe_load(CODE_MAP_PATH.read_text())


def ingest_elo(
    db_path: Path | str = DEFAULT_DB_PATH,
    as_of: dt.date | None = None,
) -> dict[str, Any]:
    """Fetch World.tsv and write current Elo into team_ratings.

    Returns a summary dict with fetched/written counts and any unmapped codes.
    """
    as_of = as_of or dt.date.today()
    text = fetch_world_tsv()
    rows = parse_world_tsv(text)
    code_map = load_code_map()

    conn = get_conn(db_path)
    run_id: int | None = None
    try:
        run_id = conn.execute(
            "INSERT INTO ingest_runs (source) VALUES (?)", ("eloratings.net",)
        ).lastrowid

        # Persist the mapping itself for traceability
        conn.executemany(
            "INSERT OR IGNORE INTO team_code_map (source, source_code, fifa_code) VALUES (?, ?, ?)",
            [("eloratings", src, fifa) for src, fifa in code_map.items()],
        )

        batch = []
        unmapped: list[tuple[int, str, float]] = []
        for r in rows:
            fifa = code_map.get(r["src_code"])
            if fifa is None:
                unmapped.append((r["rank"], r["src_code"], r["elo"]))
                continue
            batch.append((fifa, "elo", r["elo"], as_of.isoformat(), "eloratings.net"))

        conn.executemany(
            """
            INSERT INTO team_ratings (team_code, rating_type, value, as_of_date, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (team_code, rating_type, as_of_date, source) DO UPDATE SET
                value = excluded.value
            """,
            batch,
        )

        conn.execute(
            "UPDATE ingest_runs SET finished_at=datetime('now'), status='success', "
            "rows_written=? WHERE id=?",
            (len(batch), run_id),
        )
        return {
            "fetched": len(rows),
            "mapped": len(batch),
            "written": len(batch),
            "skipped_unmapped": len(unmapped),
            "unmapped_codes": [u[1] for u in unmapped],
        }
    except sqlite3.Error:
        if run_id:
            conn.execute(
                "UPDATE ingest_runs SET finished_at=datetime('now'), status='error' WHERE id=?",
                (run_id,),
            )
        raise
    finally:
        conn.close()


def main() -> None:
    import json

    result = ingest_elo()
    # Print a compact summary; collapse the long unmapped list
    summary = {**result}
    summary["unmapped_codes"] = (
        f"{len(result['unmapped_codes'])} codes (non-WC teams skipped)"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
