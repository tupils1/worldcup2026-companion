"""API-Football (api-sports.io v3) ingester.

Pulls:
  - team mapping (league=1 WC, season=2026) → team_code_map
  - fixtures for any (league, season) → matches table
  - odds when available → odds table

Notes:
  - API-Football team `code` is unreliable (duplicates: IRA = both Iran & Iraq,
    AUS = both Australia & Austria). Always use `team.id`.
  - `match_date` is normalized to YYYY-MM-DD (date only) so it merges with
    the martj42 ingest on the existing UNIQUE (match_date, home, away).
  - Status `FT` = full time; `NS` = not started; treat all others as in-progress.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator

import httpx

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

BASE_URL = "https://v3.football.api-sports.io"
SECRETS_PATH = Path(__file__).resolve().parents[3] / "configs" / "secrets.env"

# API-Football name → our FIFA code (where name differs from teams.yaml.name)
NAME_ALIASES: dict[str, str] = {
    "Czech Republic": "CZE",
    "Bosnia & Herzegovina": "BIH",
    "Congo DR": "COD",
    "Cape Verde Islands": "CPV",
    "USA": "USA",
    "Saudi Arabia": "KSA",
    "South Korea": "KOR",
    "South Africa": "RSA",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Switzerland": "SUI",
    "Netherlands": "NED",
    "Ivory Coast": "CIV",
    "Türkiye": "TUR",
    "Curaçao": "CUW",
}


def load_api_key(env_path: Path = SECRETS_PATH) -> str:
    """Read API_FOOTBALL_KEY from configs/secrets.env."""
    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} missing — copy secrets.example.env first")
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("API_FOOTBALL_KEY="):
            key = line.split("=", 1)[1].strip()
            if not key:
                raise ValueError("API_FOOTBALL_KEY is empty in secrets.env")
            return key
    raise KeyError("API_FOOTBALL_KEY not found in secrets.env")


class APIFootballClient:
    """Thin client with built-in pagination and a soft rate limit."""

    def __init__(
        self,
        key: str | None = None,
        min_interval_sec: float = 0.15,  # ~6 req/sec, well under Ultra's caps
    ):
        self.key = key or load_api_key()
        self.headers = {"x-apisports-key": self.key}
        self.min_interval = min_interval_sec
        self._last_call = 0.0
        self._client = httpx.Client(headers=self.headers, timeout=30)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get(self, path: str, **params: Any) -> dict:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        r = self._client.get(f"{BASE_URL}{path}", params=params)
        self._last_call = time.monotonic()
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            errs = data["errors"]
            if errs:  # non-empty dict/list means real errors
                raise RuntimeError(f"API-Football errors for {path}: {errs}")
        return data

    def paginated(self, path: str, **params: Any) -> Iterator[dict]:
        """Yield response items. Avoids sending `page=1` since some endpoints
        (e.g. /fixtures) reject the `page` field entirely."""
        page = 1
        while True:
            call_params = dict(params)
            if page > 1:
                call_params["page"] = page
            data = self.get(path, **call_params)
            for item in data.get("response", []):
                yield item
            paging = data.get("paging") or {"current": 1, "total": 1}
            if page >= paging.get("total", 1):
                return
            page += 1

    def status(self) -> dict:
        return self.get("/status")["response"]


def ingest_team_mapping(
    league: int = 1,
    season: int = 2026,
    db_path: Path | str = DEFAULT_DB_PATH,
    client: APIFootballClient | None = None,
) -> dict[str, Any]:
    """Pull (league, season) team list and write team_code_map rows.

    Returns {written, missing_codes (teams API gave us that didn't match anything)}.
    """
    own_client = client is None
    client = client or APIFootballClient()
    try:
        data = client.get("/teams", league=league, season=season)
    finally:
        if own_client:
            client.close()

    conn = get_conn(db_path)
    try:
        fifa_by_name = {
            r["name"]: r["code"]
            for r in conn.execute("SELECT code, name FROM teams")
        }

        rows: list[tuple[str, str, str]] = []
        unmapped: list[dict] = []
        for r in data["response"]:
            t = r["team"]
            fifa_code = NAME_ALIASES.get(t["name"]) or fifa_by_name.get(t["name"])
            if fifa_code is None:
                unmapped.append({"af_id": t["id"], "name": t["name"]})
                continue
            rows.append(("api_football", str(t["id"]), fifa_code))

        conn.executemany(
            "INSERT INTO team_code_map (source, source_code, fifa_code) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(source, source_code) DO UPDATE SET fifa_code=excluded.fifa_code",
            rows,
        )

        run = conn.execute(
            "INSERT INTO ingest_runs (source, finished_at, status, rows_written) "
            "VALUES (?, datetime('now'), 'success', ?)",
            (f"api_football:/teams?league={league}&season={season}", len(rows)),
        )
    finally:
        conn.close()

    return {
        "league": league,
        "season": season,
        "total_teams": data["results"],
        "mapped": len(rows),
        "unmapped": unmapped,
    }


def ingest_fixtures(
    league: int,
    season: int,
    db_path: Path | str = DEFAULT_DB_PATH,
    client: APIFootballClient | None = None,
) -> dict[str, Any]:
    """Pull all fixtures for (league, season) and upsert into `matches`.

    Merges with martj42 rows via UNIQUE (match_date, home_code, away_code) —
    adds api_football_id + venue + stage, fills score when finished.
    """
    own_client = client is None
    client = client or APIFootballClient()

    conn = get_conn(db_path)
    try:
        try:  # migrate: full ISO kickoff (match_date stays date-only as the merge key)
            conn.execute("ALTER TABLE matches ADD COLUMN kickoff_utc TEXT")
        except Exception:
            pass
        af_to_fifa = {
            int(r["source_code"]): r["fifa_code"]
            for r in conn.execute(
                "SELECT source_code, fifa_code FROM team_code_map "
                "WHERE source='api_football'"
            )
        }
        if not af_to_fifa:
            raise RuntimeError(
                "No api_football team mapping in team_code_map. "
                "Run ingest_team_mapping() first."
            )

        run_id = conn.execute(
            "INSERT INTO ingest_runs (source) VALUES (?)",
            (f"api_football:/fixtures?league={league}&season={season}",),
        ).lastrowid

        # Count rows before to compute (new vs merged-into-existing) accurately.
        # SQLite's UPSERT rowcount is always 1, so we can't tell them apart per-call.
        n_before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        seen = 0
        skipped_unmapped = 0
        try:
            for f in client.paginated("/fixtures", league=league, season=season):
                fx = f["fixture"]
                teams = f["teams"]
                score = f["score"]
                home_af = teams["home"]["id"]
                away_af = teams["away"]["id"]
                home_code = af_to_fifa.get(home_af)
                away_code = af_to_fifa.get(away_af)
                if not home_code or not away_code:
                    skipped_unmapped += 1
                    continue
                date_only = fx["date"][:10]  # YYYY-MM-DD
                kickoff = fx["date"]         # full ISO incl. timezone — the digest's 开球时间
                status = fx["status"]["short"]
                finished = 1 if status == "FT" else 0
                h_score = score["fulltime"]["home"] if finished else None
                a_score = score["fulltime"]["away"] if finished else None
                stage = f.get("league", {}).get("round")
                venue_name = (fx.get("venue") or {}).get("name")

                seen += 1
                comp = _competition_name(league, season)
                # Merge into an existing fixture for the same pairing even when the
                # date drifted by a day. The martj42 and api_football feeds disagree
                # on the calendar date (timezone), so a strict ON CONFLICT(match_date,
                # home, away) used to MISS the existing row and insert a duplicate.
                # Match on teams within ±1 day instead (either orientation).
                existing = conn.execute(
                    """
                    SELECT id, home_code, away_code FROM matches
                    WHERE ((home_code=? AND away_code=?) OR (home_code=? AND away_code=?))
                      AND ABS(julianday(match_date) - julianday(?)) <= 1
                    ORDER BY ABS(julianday(match_date) - julianday(?)) LIMIT 1
                    """,
                    (home_code, away_code, away_code, home_code, date_only, date_only),
                ).fetchone()
                if existing is not None:
                    if existing["home_code"] == home_code and existing["away_code"] == away_code:
                        # Same orientation → safe to backfill id/stage/venue + scores.
                        conn.execute(
                            """
                            UPDATE matches SET
                                api_football_id = ?,
                                kickoff_utc = ?,
                                competition = COALESCE(competition, ?),
                                stage       = COALESCE(stage, ?),
                                venue       = COALESCE(venue, ?),
                                home_score  = COALESCE(?, home_score),
                                away_score  = COALESCE(?, away_score),
                                finished    = MAX(finished, ?)
                            WHERE id = ?
                            """,
                            (fx["id"], kickoff, comp, stage, venue_name,
                             h_score, a_score, finished, existing["id"]),
                        )
                    else:
                        # Reversed orientation (sources disagree on nominal home team)
                        # → backfill stage + kickoff only (both orientation-independent);
                        # copying the fixture id or scores would flip home/away (and
                        # break live-score attribution) on this row.
                        conn.execute(
                            "UPDATE matches SET stage = COALESCE(stage, ?), "
                            "kickoff_utc = COALESCE(kickoff_utc, ?) WHERE id = ?",
                            (stage, kickoff, existing["id"]),
                        )
                else:
                    conn.execute(
                        """
                        INSERT INTO matches (
                            api_football_id, home_code, away_code, match_date, kickoff_utc,
                            competition, stage, venue, neutral_venue,
                            home_score, away_score, finished, source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(match_date, home_code, away_code) DO UPDATE SET
                            api_football_id = excluded.api_football_id,
                            kickoff_utc     = excluded.kickoff_utc,
                            competition     = COALESCE(competition,     excluded.competition),
                            stage           = COALESCE(stage,           excluded.stage),
                            venue           = COALESCE(venue,           excluded.venue),
                            home_score      = COALESCE(excluded.home_score, home_score),
                            away_score      = COALESCE(excluded.away_score, away_score),
                            finished        = MAX(finished, excluded.finished)
                        """,
                        (fx["id"], home_code, away_code, date_only, kickoff, comp, stage,
                         venue_name, 0, h_score, a_score, finished, "api_football"),
                    )

            n_after = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            inserted = n_after - n_before
            merged = seen - inserted  # existing rows updated with API-Football info
            conn.execute(
                "UPDATE ingest_runs SET finished_at=datetime('now'), status='success', "
                "rows_written=? WHERE id=?",
                (seen, run_id),
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
        "league": league,
        "season": season,
        "seen": seen,
        "inserted_new": inserted,
        "merged_into_existing": merged,
        "skipped_unmapped": skipped_unmapped,
    }


def _competition_name(league_id: int, season: int) -> str:
    return {
        1: f"FIFA World Cup {season}",
        29: f"World Cup {season} qualif. — CAF",
        31: f"World Cup {season} qualif. — CONCACAF",
        32: f"World Cup {season} qualif. — UEFA",
        33: f"World Cup {season} qualif. — OFC",
        34: f"World Cup {season} qualif. — CONMEBOL",
        37: f"World Cup {season} qualif. — Intercontinental",
    }.get(league_id, f"League {league_id} season {season}")


def dedup_future_fixture_twins(db_path=None) -> int:
    """Remove martj42 future-fixture twins when an api_football row exists for the same
    (home, away). martj42 carries the 2026 WC schedule as 'history' with dates offset ~1 day
    (timezone artifact) and no odds → duplicate fixtures; the api_football row is canonical
    (carries odds + fixture id). Idempotent and safe — deleted rows have no odds / FK refs.
    Returns the number deleted. Called at the end of main() so the daily refresh self-heals."""
    from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
    conn = get_conn(db_path or DEFAULT_DB_PATH)
    try:
        cur = conn.execute("""
            DELETE FROM matches
            WHERE finished=0 AND source='martj42' AND match_date >= '2026-06-01'
              AND EXISTS (SELECT 1 FROM matches m2
                          WHERE m2.home_code = matches.home_code
                            AND m2.away_code = matches.away_code
                            AND m2.source = 'api_football' AND m2.finished = 0
                            AND m2.match_date >= '2026-06-01')""")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def main() -> None:
    """CLI: bootstrap WC 2026 team mapping + fixtures, also pull WC 2022 history."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--leagues",
        default="1:2026,1:2022,32:2024,32:2026,34:2026,29:2026,31:2026,33:2026,37:2026",
        help="Comma-sep league:season pairs to pull fixtures for",
    )
    args = ap.parse_args()

    with APIFootballClient() as client:
        st = client.status()
        print(
            f"API-Football {st['subscription']['plan']} plan, "
            f"used {st['requests']['current']}/{st['requests']['limit_day']} today\n"
        )

        # Team mapping comes from WC 2026 first (gives us 48/48)
        print("=== /teams league=1 season=2026 ===")
        r = ingest_team_mapping(league=1, season=2026, client=client)
        print(json.dumps(r, indent=2))

        # Fixtures for each (league, season)
        for pair in args.leagues.split(","):
            lg, sea = pair.split(":")
            print(f"\n=== /fixtures league={lg} season={sea} ===")
            r = ingest_fixtures(league=int(lg), season=int(sea), client=client)
            print(json.dumps(r, indent=2))

        n_dedup = dedup_future_fixture_twins()
        if n_dedup:
            print(f"\ndeduped {n_dedup} martj42 future-fixture twins (kept api_football canonical)")

        st = client.status()
        print(
            f"\nrequests used: {st['requests']['current']}/{st['requests']['limit_day']}"
        )


if __name__ == "__main__":
    main()
