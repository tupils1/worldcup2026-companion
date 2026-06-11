"""Historical international match ingester.

Source: https://github.com/martj42/international_results
  - results.csv: 1872-present, ~49k rows. Date, home/away team (full names),
    scores (NA for unplayed future fixtures), tournament, city, country, neutral.
  - Updated monthly via PR.

We input only the matches involving at least one mapped team (initially the 48
WC teams, extensible). Future fixtures with NA scores are kept as
unfinished rows — useful because martj42 already lists the 2026 WC fixtures.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import yaml

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

RESULTS_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
ALIAS_PATH = CONFIG_DIR / "code_maps" / "team_name_aliases.yaml"

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) worldcup-research"

DEFAULT_SINCE = "2014-01-01"  # cover last 3 WC cycles for Dixon-Coles


def fetch_results_csv() -> bytes:
    r = httpx.get(RESULTS_CSV_URL, headers={"User-Agent": UA}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.content


def load_name_to_code(conn: sqlite3.Connection) -> dict[str, str]:
    """teams.name → teams.code, plus configured aliases.

    Returns a unified lookup. None values in aliases mean "explicit skip".
    """
    name_map: dict[str, str | None] = {}
    for row in conn.execute("SELECT name, code FROM teams"):
        name_map[row["name"]] = row["code"]

    aliases = yaml.safe_load(ALIAS_PATH.read_text()) or {}
    for k, v in aliases.items():
        name_map[k] = v  # may be None == explicit skip

    return name_map


def ingest_historical_matches(
    db_path: Path | str = DEFAULT_DB_PATH,
    since: str = DEFAULT_SINCE,
    csv_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Pull martj42 results.csv, filter to WC-relevant matches, persist.

    A match is included if BOTH teams resolve to a known FIFA code.
    """
    raw = csv_bytes or fetch_results_csv()
    df = pl.read_csv(io.BytesIO(raw), null_values=["NA"])
    df = df.filter(pl.col("date") >= since)

    conn = get_conn(db_path)
    try:
        name_map = load_name_to_code(conn)
        run_id = conn.execute(
            "INSERT INTO ingest_runs (source) VALUES (?)", ("martj42/international_results",)
        ).lastrowid

        rows: list[tuple] = []
        unmapped_names: dict[str, int] = {}
        for r in df.iter_rows(named=True):
            home = name_map.get(r["home_team"], "MISS")
            away = name_map.get(r["away_team"], "MISS")
            if home is None or away is None:
                continue  # explicit skip
            if home == "MISS":
                unmapped_names[r["home_team"]] = unmapped_names.get(r["home_team"], 0) + 1
            if away == "MISS":
                unmapped_names[r["away_team"]] = unmapped_names.get(r["away_team"], 0) + 1
            if home == "MISS" or away == "MISS":
                continue
            finished = 1 if (r["home_score"] is not None and r["away_score"] is not None) else 0
            rows.append(
                (
                    home, away, r["date"],
                    r["tournament"],
                    r["city"],
                    1 if r["neutral"] else 0,
                    r["home_score"], r["away_score"],
                    finished,
                    "martj42",
                )
            )

        conn.execute("BEGIN")
        conn.executemany(
            """
            INSERT INTO matches
                (home_code, away_code, match_date, competition, venue, neutral_venue,
                 home_score, away_score, finished, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (match_date, home_code, away_code) DO UPDATE SET
                competition   = COALESCE(excluded.competition,   competition),
                venue         = COALESCE(excluded.venue,         venue),
                neutral_venue = excluded.neutral_venue,
                home_score    = excluded.home_score,
                away_score    = excluded.away_score,
                finished      = excluded.finished
            """,
            rows,
        )
        conn.execute("COMMIT")

        conn.execute(
            "UPDATE ingest_runs SET finished_at=datetime('now'), status='success', "
            "rows_written=? WHERE id=?",
            (len(rows), run_id),
        )

        return {
            "csv_rows_since": df.height,
            "ingested": len(rows),
            "unmapped_distinct": len(unmapped_names),
            "unmapped_top10": sorted(unmapped_names.items(), key=lambda x: -x[1])[:10],
        }
    except Exception:
        conn.execute("ROLLBACK")
        conn.execute(
            "UPDATE ingest_runs SET finished_at=datetime('now'), status='error' WHERE id=?",
            (run_id,),
        )
        raise
    finally:
        conn.close()


def main() -> None:
    import json

    print(json.dumps(ingest_historical_matches(), indent=2, default=str))


if __name__ == "__main__":
    main()
