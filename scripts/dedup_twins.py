"""De-duplicate twin match rows for the 2026 World Cup group stage.

Two ingest feeds (martj42 + api_football) sometimes create two rows for the same
fixture, dated up to a day apart, because the date conventions differ. In a round
robin each pair plays exactly once, so any two rows with the same unordered
(home, away) inside the WC group stage are the same match. The martj42 bare rows
(no stage / no kickoff / no xG / no api_football_id) are a strict subset of the
api_football rows, so they're the ones to drop.

Duplicates corrupt everything downstream once results are in: the DC fit counts a
match twice, the 对答案 scoreboard double-counts, and group standings would award
double points. This is a one-shot cleanup the USER runs (the app treats the DB as
read-only) — dry-run by default, --apply to execute (auto-backs up first).

    PYTHONPATH=src python scripts/dedup_twins.py            # preview
    PYTHONPATH=src python scripts/dedup_twins.py --apply    # execute (backs up first)
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sqlite3
from pathlib import Path

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

WC_LO, WC_HI = "2026-06-11", "2026-07-31"


def _completeness(r: sqlite3.Row) -> tuple:
    """Higher = richer row to KEEP. Prefer api_football_id, then a stage label,
    then xG, then having scores, then the lower id (older/canonical)."""
    return (
        1 if r["api_football_id"] is not None else 0,
        1 if (r["stage"] or "") else 0,
        1 if r["home_xg"] is not None else 0,
        1 if r["home_score"] is not None else 0,
        -r["id"],  # tie-break: keep the lower id
    )


def find_dupes(conn: sqlite3.Connection) -> list[tuple[sqlite3.Row, list[sqlite3.Row]]]:
    """Group 2026 WC group-stage rows by unordered pair; return (keep, [drop...])."""
    rows = conn.execute(
        """SELECT id, home_code, away_code, match_date, kickoff_utc, stage,
                  home_score, away_score, home_xg, api_football_id, source
           FROM matches
           WHERE match_date BETWEEN ? AND ?
             AND competition LIKE '%World Cup%'
             AND (stage LIKE 'Group%' OR stage IS NULL OR stage = '')
           ORDER BY id""",
        (WC_LO, WC_HI),
    ).fetchall()
    groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for r in rows:
        key = tuple(sorted((r["home_code"], r["away_code"])))
        groups.setdefault(key, []).append(r)
    out = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        members.sort(key=_completeness, reverse=True)
        keep, drop = members[0], members[1:]
        # Safety: only drop rows that are genuinely less complete (never drop a row
        # carrying a stage/afid the keeper lacks — that would be a real merge, not a dup).
        safe_drop = [d for d in drop
                     if not (d["api_football_id"] is not None and keep["api_football_id"] is None)
                     and not ((d["stage"] or "") and not (keep["stage"] or ""))]
        if safe_drop:
            out.append((keep, safe_drop))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="execute the deletes (default: dry-run)")
    args = ap.parse_args()

    conn = get_conn(DEFAULT_DB_PATH)
    pairs = find_dupes(conn)
    drop_ids = [d["id"] for _, ds in pairs for d in ds]

    print(f"{'APPLY' if args.apply else 'DRY-RUN'} — {len(pairs)} fixture(s) with twins, "
          f"{len(drop_ids)} row(s) to drop:\n")
    for keep, ds in pairs:
        print(f"  KEEP {keep['id']:>6} {keep['home_code']}-{keep['away_code']} "
              f"{keep['match_date']} [{keep['stage'] or '无stage'}] src={keep['source']}")
        for d in ds:
            print(f"  DROP {d['id']:>6} {d['home_code']}-{d['away_code']} "
                  f"{d['match_date']} [{d['stage'] or '无stage'}] src={d['source']}")

    if not args.apply:
        print("\n(dry-run; re-run with --apply to delete. The DB is backed up automatically.)")
        conn.close()
        return
    if not drop_ids:
        print("\nNothing to drop.")
        conn.close()
        return

    src = Path(DEFAULT_DB_PATH)
    bak = src.with_name(src.name + f".bak-{dt.datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy2(src, bak)
    print(f"\nbackup → {bak}")
    conn.execute(f"DELETE FROM matches WHERE id IN ({','.join('?' * len(drop_ids))})", drop_ids)
    conn.commit()
    print(f"deleted {len(drop_ids)} row(s).")

    # B21: unify the split 2026-WC competition label ('FIFA World Cup' from martj42
    # vs 'FIFA World Cup 2026' from api_football) so exact-match queries are reliable.
    n_lbl = conn.execute(
        "UPDATE matches SET competition='FIFA World Cup 2026' "
        "WHERE competition='FIFA World Cup' AND match_date>=?", (WC_LO,)).rowcount
    conn.commit()
    print(f"normalized {n_lbl} competition label(s) → 'FIFA World Cup 2026'.")
    remaining = conn.execute(
        """SELECT COUNT(*) FROM (
             SELECT 1 FROM matches WHERE match_date BETWEEN ? AND ?
               AND competition LIKE '%World Cup%'
               AND (stage LIKE 'Group%' OR stage IS NULL OR stage='')
             GROUP BY (CASE WHEN home_code<away_code THEN home_code||away_code
                            ELSE away_code||home_code END)
             HAVING COUNT(*) > 1)""", (WC_LO, WC_HI)).fetchone()[0]
    print(f"remaining duplicate pairs: {remaining} (should be 0)")
    conn.close()


if __name__ == "__main__":
    main()
