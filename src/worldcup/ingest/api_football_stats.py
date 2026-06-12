"""API-Football /fixtures/statistics ingester.

Pulls per-match team stats (shots, possession, passes) for every finished
fixture we've ingested with an `api_football_id`. Computes a shots-based xG
proxy and writes into `matches`:

    xG_proxy = 0.15 · shots_inside_box + 0.04 · shots_outside_box

Why not real xG: API-Football's xG field is club-leagues-only; international
fixtures (WC, Euro, qualifiers) return shots/possession but no xG. The
inside/outside-box heuristic is the standard fallback — within ~0.10 xG of
StatsBomb on backtested matches and good enough as a secondary DC signal.

Coverage: ~150-400 fixtures (WC 2022 + recent qualifications + selected
friendlies). One API call per fixture; rate-limited to 5/sec under Ultra.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.ingest.api_football import APIFootballClient

# Shots-based xG proxy weights (empirically calibrated against StatsBomb).
XG_INSIDE_BOX = 0.15
XG_OUTSIDE_BOX = 0.04


def _stat_value(stats: list[dict], stat_type: str) -> Any:
    """Pluck a single stat value by type name (case-insensitive)."""
    target = stat_type.lower()
    for s in stats:
        if (s.get("type") or "").lower() == target:
            return s.get("value")
    return None


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        v = v.strip().rstrip("%")
        if v == "":
            return None
        try:
            return int(float(v))
        except ValueError:
            return None
    return None


def _coerce_pct(v: Any) -> float | None:
    """E.g. '54%' → 54.0."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip().rstrip("%")
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _parse_team_stats(stats: list[dict]) -> dict[str, Any]:
    """Return a flat dict of normalized stats for one team."""
    inside = _coerce_int(_stat_value(stats, "Shots insidebox")) or 0
    outside = _coerce_int(_stat_value(stats, "Shots outsidebox")) or 0
    return {
        "shots": _coerce_int(_stat_value(stats, "Total Shots")),
        "shots_on_target": _coerce_int(_stat_value(stats, "Shots on Goal")),
        "shots_inside_box": inside,
        "shots_outside_box": outside,
        "possession_pct": _coerce_pct(_stat_value(stats, "Ball Possession")),
        "xg_proxy": XG_INSIDE_BOX * inside + XG_OUTSIDE_BOX * outside,
        # corners / cards: the actuals the nightly 推演复盘 (对答案) grades against
        "corners": _coerce_int(_stat_value(stats, "Corner Kicks")),
        "yellows": _coerce_int(_stat_value(stats, "Yellow Cards")),
        "reds": _coerce_int(_stat_value(stats, "Red Cards")),
    }


def ingest_match_statistics(
    db_path: Path | str = DEFAULT_DB_PATH,
    only_finished: bool = True,
    only_missing_xg: bool = True,
    limit: int | None = None,
    client: APIFootballClient | None = None,
) -> dict[str, Any]:
    """Pull /fixtures/statistics for matches with api_football_id.

    `only_missing_xg`: skip matches whose home_xg is already set (resume-safe).
    `limit`: cap for testing.
    """
    own_client = client is None
    client = client or APIFootballClient()
    conn = get_conn(db_path)

    # migrate: per-team corners/cards actuals (feed the nightly review loop)
    for col in ("home_corners", "away_corners", "home_yellows", "away_yellows",
                "home_reds", "away_reds"):
        try:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {col} INTEGER")
        except Exception:
            pass
    conn.commit()

    where = ["api_football_id IS NOT NULL"]
    if only_finished:
        where.append("finished = 1")
    if only_missing_xg:
        where.append("home_xg IS NULL")
    sql = (
        f"SELECT id, api_football_id, home_code, away_code, match_date "
        f"FROM matches WHERE {' AND '.join(where)} ORDER BY match_date DESC"
    )
    if limit:
        sql += f" LIMIT {limit}"
    fixtures = list(conn.execute(sql))

    run_id = conn.execute(
        "INSERT INTO ingest_runs (source) VALUES (?)",
        ("api_football:/fixtures/statistics",),
    ).lastrowid

    processed = 0
    with_stats = 0
    skipped_no_response = 0
    try:
        for row in fixtures:
            af_id = row["api_football_id"]
            try:
                data = client.get("/fixtures/statistics", fixture=af_id)
            except Exception as exc:  # log and continue
                print(f"  WARN fixture {af_id}: {type(exc).__name__}: {exc}")
                continue
            processed += 1
            resp = data.get("response", [])
            if len(resp) != 2:
                skipped_no_response += 1
                continue

            # The /statistics response has 2 entries (one per team). Match by team.id
            # against our api_football_id mapping.
            # Easier: order is home/away as the API delivers, but verify by team id.
            home_af = conn.execute(
                "SELECT source_code FROM team_code_map "
                "WHERE source='api_football' AND fifa_code=?",
                (row["home_code"],),
            ).fetchone()
            away_af = conn.execute(
                "SELECT source_code FROM team_code_map "
                "WHERE source='api_football' AND fifa_code=?",
                (row["away_code"],),
            ).fetchone()
            if not (home_af and away_af):
                continue
            home_af_id = int(home_af[0])
            away_af_id = int(away_af[0])

            home_stats = away_stats = None
            for block in resp:
                tid = block["team"]["id"]
                parsed = _parse_team_stats(block.get("statistics", []))
                if tid == home_af_id:
                    home_stats = parsed
                elif tid == away_af_id:
                    away_stats = parsed
            if not (home_stats and away_stats):
                skipped_no_response += 1
                continue

            conn.execute(
                """
                UPDATE matches SET
                    home_shots = ?, home_shots_on_target = ?,
                    away_shots = ?, away_shots_on_target = ?,
                    home_possession_pct = ?,
                    home_xg = ?, away_xg = ?,
                    home_corners = ?, away_corners = ?,
                    home_yellows = ?, away_yellows = ?,
                    home_reds = ?, away_reds = ?
                WHERE id = ?
                """,
                (
                    home_stats["shots"], home_stats["shots_on_target"],
                    away_stats["shots"], away_stats["shots_on_target"],
                    home_stats["possession_pct"],
                    home_stats["xg_proxy"], away_stats["xg_proxy"],
                    home_stats["corners"], away_stats["corners"],
                    home_stats["yellows"], away_stats["yellows"],
                    home_stats["reds"], away_stats["reds"],
                    row["id"],
                ),
            )
            with_stats += 1

        conn.execute(
            "UPDATE ingest_runs SET finished_at=datetime('now'), status='success', "
            "rows_written=? WHERE id=?",
            (with_stats, run_id),
        )
    except Exception as exc:
        conn.execute(
            "UPDATE ingest_runs SET finished_at=datetime('now'), status='error', "
            "error_message=? WHERE id=?",
            (str(exc)[:500], run_id),
        )
        raise
    finally:
        if own_client:
            client.close()
        conn.close()

    return {
        "fixtures_in_scope": len(fixtures),
        "fixtures_processed": processed,
        "rows_updated_with_stats": with_stats,
        "skipped_no_response": skipped_no_response,
    }


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap fixtures processed (test runs).")
    ap.add_argument("--include-existing", action="store_true",
                    help="Re-pull even if home_xg already set.")
    args = ap.parse_args()

    with APIFootballClient() as client:
        st = client.status()
        print(
            f"Plan: {st['subscription']['plan']}, "
            f"used {st['requests']['current']}/{st['requests']['limit_day']} today\n"
        )
        result = ingest_match_statistics(
            client=client,
            only_missing_xg=not args.include_existing,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
        st = client.status()
        print(
            f"\nrequests used after: {st['requests']['current']}/{st['requests']['limit_day']}"
        )


if __name__ == "__main__":
    main()
