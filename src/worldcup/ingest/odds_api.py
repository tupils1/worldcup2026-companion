"""The Odds API (the-odds-api.com v4) ingester.

Pulls FIFA World Cup 2026 odds across configured bookmakers, persists snapshots
into the `odds` table. Tracks quota in response headers.

Markets pulled:
    - h2h     → 1X2 (home / draw / away)
    - spreads → Asian-like handicap (point from each side's perspective)
    - totals  → over/under with line
    - outrights → tournament winner futures

Key in `configs/secrets.env` as ODDS_API_KEY.

Quota model: each "request" is one API call (regardless of how many bookmakers/markets).
Match-odds + outrights for WC ≈ 2-4 requests per refresh. Hourly refresh during
the tournament burns ~100 req/day — well under the $59/mo plan's 20k/mo.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

BASE_URL = "https://api.the-odds-api.com/v4"
# Two distinct sport keys — match odds vs tournament-winner futures are split.
SPORT_KEY_MATCH = "soccer_fifa_world_cup"
SPORT_KEY_OUTRIGHT = "soccer_fifa_world_cup_winner"
SECRETS_PATH = Path(__file__).resolve().parents[3] / "configs" / "secrets.env"

DEFAULT_REGIONS = "us,uk,eu,au"
DEFAULT_MARKETS = "h2h,spreads,totals"
# Sharp + popular books. Pinnacle is the no-vig baseline; market-leader books
# (Bet365, Williamhill, Marathon) provide depth.
DEFAULT_BOOKMAKERS: str | None = None  # None → all books in the regions

# Map Odds API team names → our FIFA 3-letter codes
NAME_ALIASES: dict[str, str] = {
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Bosnia & Herzegovina": "BIH",
    "Bosnia and Herzegovina": "BIH",
    "Congo DR": "COD",
    "DR Congo": "COD",
    "Cape Verde Islands": "CPV",
    "Cape Verde": "CPV",
    "USA": "USA",
    "United States": "USA",
    "Saudi Arabia": "KSA",
    "South Korea": "KOR",
    "Korea Republic": "KOR",
    "South Africa": "RSA",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Switzerland": "SUI",
    "Netherlands": "NED",
    "Holland": "NED",
    "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV",
    "Türkiye": "TUR",
    "Turkey": "TUR",
    "Curaçao": "CUW",
    "Curacao": "CUW",
}


def load_api_key(env_path: Path = SECRETS_PATH) -> str:
    if not env_path.exists():
        raise FileNotFoundError(env_path)
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("ODDS_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if not key:
                raise ValueError(
                    "ODDS_API_KEY is empty in configs/secrets.env — set it first"
                )
            return key
    raise KeyError("ODDS_API_KEY not found in configs/secrets.env")


def _build_name_map(conn: sqlite3.Connection) -> dict[str, str]:
    """All known name → fifa_code mappings (teams.name + manual aliases)."""
    m = {}
    for r in conn.execute("SELECT name, code FROM teams"):
        m[r["name"]] = r["code"]
    m.update(NAME_ALIASES)
    return m


class OddsAPIClient:
    """Thin client. Auth via `apiKey` query param. Tracks quota from headers."""

    def __init__(self, key: str | None = None, min_interval_sec: float = 0.5):
        self.key = key or load_api_key()
        self.min_interval = min_interval_sec
        self._last_call = 0.0
        self._client = httpx.Client(timeout=30)
        self.requests_remaining: int | None = None
        self.requests_used: int | None = None
        self.last_cost: int | None = None  # per-call cost from x-requests-last header

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get(self, path: str, **params: Any) -> Any:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        params["apiKey"] = self.key
        r = self._client.get(f"{BASE_URL}{path}", params=params)
        self._last_call = time.monotonic()
        # Quota headers
        rh = r.headers
        try:
            self.requests_remaining = int(rh.get("x-requests-remaining", ""))
        except ValueError:
            pass
        try:
            self.requests_used = int(rh.get("x-requests-used", ""))
        except ValueError:
            pass
        try:
            self.last_cost = int(rh.get("x-requests-last", ""))
        except ValueError:
            pass
        if r.status_code == 401:
            raise PermissionError(
                "Odds API rejected the key (401). Check ODDS_API_KEY in secrets.env."
            )
        if r.status_code == 429:
            raise RuntimeError("Odds API rate-limited (429).")
        r.raise_for_status()
        return r.json()

    def list_sports(self) -> list[dict]:
        return self.get("/sports/", all="true")

    def get_match_odds(
        self,
        sport: str = SPORT_KEY_MATCH,
        regions: str = DEFAULT_REGIONS,
        markets: str = DEFAULT_MARKETS,
        bookmakers: str | None = DEFAULT_BOOKMAKERS,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self.get(f"/sports/{sport}/odds/", **params)

    def get_outright_odds(
        self,
        sport: str = SPORT_KEY_OUTRIGHT,
        regions: str = DEFAULT_REGIONS,
        bookmakers: str | None = DEFAULT_BOOKMAKERS,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "regions": regions,
            "markets": "outrights",
            "oddsFormat": "decimal",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        return self.get(f"/sports/{sport}/odds/", **params)


def _normalize_market_outcome(
    market_key: str,
    outcome: dict,
    home_name: str,
    away_name: str,
) -> tuple[str, str, float | None] | None:
    """Return (market, selection, line) for our `odds` table, or None to skip.

    market: '1X2' | 'AH' | 'OU'
    selection: 'home' | 'away' | 'draw' | 'over' | 'under'
    line: handicap (from selection's POV) for AH, total goals for OU, None for 1X2
    """
    name = outcome.get("name", "")
    price = outcome.get("price")
    if price is None or price <= 1.0:
        return None
    if market_key == "h2h":
        if name == home_name:
            return ("1X2", "home", None)
        if name == away_name:
            return ("1X2", "away", None)
        return ("1X2", "draw", None)
    if market_key == "spreads":
        point = outcome.get("point")
        if name == home_name:
            return ("AH", "home", point)
        if name == away_name:
            return ("AH", "away", point)
        return None
    if market_key == "totals":
        point = outcome.get("point")
        if name.lower() == "over":
            return ("OU", "over", point)
        if name.lower() == "under":
            return ("OU", "under", point)
        return None
    return None


def ingest_match_odds(
    db_path: Path | str = DEFAULT_DB_PATH,
    client: OddsAPIClient | None = None,
    markets: str = DEFAULT_MARKETS,
    regions: str = DEFAULT_REGIONS,
    bookmakers: str | None = DEFAULT_BOOKMAKERS,
) -> dict[str, Any]:
    """Pull current match odds (1X2 + spreads + totals) and persist."""
    own_client = client is None
    client = client or OddsAPIClient()
    captured_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "+00:00"

    conn = get_conn(db_path)
    run_id = conn.execute(
        "INSERT INTO ingest_runs (source) VALUES (?)",
        ("the-odds-api:/odds (match)",),
    ).lastrowid
    try:
        try:
            data = client.get_match_odds(
                regions=regions, markets=markets, bookmakers=bookmakers
            )
        except Exception as exc:
            conn.execute(
                "UPDATE ingest_runs SET finished_at=datetime('now'), status='error', "
                "error_message=? WHERE id=?",
                (str(exc)[:500], run_id),
            )
            raise

        name_map = _build_name_map(conn)
        rows: list[tuple] = []
        matched = 0
        unmatched_events: list[dict] = []
        for event in data:
            home_name = event.get("home_team") or ""
            away_name = event.get("away_team") or ""
            home_code = name_map.get(home_name)
            away_code = name_map.get(away_name)
            if not (home_code and away_code):
                unmatched_events.append(
                    {"id": event.get("id"), "home": home_name, "away": away_name}
                )
                continue
            event_date = (event.get("commence_time") or "")[:10]
            match_row = conn.execute(
                "SELECT id FROM matches WHERE match_date=? AND home_code=? AND away_code=?",
                (event_date, home_code, away_code),
            ).fetchone()
            match_id = match_row["id"] if match_row else None

            matched += 1
            for book in event.get("bookmakers", []):
                bk = book.get("key") or "unknown"
                for m in book.get("markets", []):
                    m_key = m.get("key")
                    if m_key not in ("h2h", "spreads", "totals"):
                        continue
                    for outcome in m.get("outcomes", []):
                        norm = _normalize_market_outcome(
                            m_key, outcome, home_name, away_name
                        )
                        if not norm:
                            continue
                        market, sel, line = norm
                        rows.append(
                            (
                                match_id,
                                "match",
                                None,
                                bk,
                                market,
                                sel,
                                line,
                                float(outcome["price"]),
                                captured_at,
                            )
                        )

        conn.executemany(
            """
            INSERT INTO odds (match_id, market_scope, subject_code, bookmaker,
                              market, selection, line, price, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.execute(
            "UPDATE ingest_runs SET finished_at=datetime('now'), status='success', "
            "rows_written=? WHERE id=?",
            (len(rows), run_id),
        )
    finally:
        if own_client:
            client.close()
        conn.close()

    return {
        "events_received": len(data),
        "events_matched": matched,
        "events_unmatched": unmatched_events,
        "odds_rows_inserted": len(rows),
        "quota_used": client.requests_used,
        "quota_remaining": client.requests_remaining,
        "captured_at": captured_at,
    }


def ingest_outright_odds(
    db_path: Path | str = DEFAULT_DB_PATH,
    client: OddsAPIClient | None = None,
    regions: str = DEFAULT_REGIONS,
    bookmakers: str | None = DEFAULT_BOOKMAKERS,
) -> dict[str, Any]:
    """Pull outright (tournament winner) odds and persist as market_scope='outright'."""
    own_client = client is None
    client = client or OddsAPIClient()
    captured_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "+00:00"

    conn = get_conn(db_path)
    run_id = conn.execute(
        "INSERT INTO ingest_runs (source) VALUES (?)",
        ("the-odds-api:/odds (outright)",),
    ).lastrowid
    try:
        try:
            data = client.get_outright_odds(regions=regions, bookmakers=bookmakers)
        except Exception as exc:
            conn.execute(
                "UPDATE ingest_runs SET finished_at=datetime('now'), status='error', "
                "error_message=? WHERE id=?",
                (str(exc)[:500], run_id),
            )
            raise

        name_map = _build_name_map(conn)
        rows: list[tuple] = []
        unmatched: set[str] = set()
        for event in data:
            for book in event.get("bookmakers", []):
                bk = book.get("key") or "unknown"
                for m in book.get("markets", []):
                    if m.get("key") != "outrights":
                        continue
                    for outcome in m.get("outcomes", []):
                        team_name = outcome.get("name") or ""
                        fifa = name_map.get(team_name)
                        price = outcome.get("price")
                        if not fifa:
                            unmatched.add(team_name)
                            continue
                        if not price or price <= 1.0:
                            continue
                        rows.append(
                            (
                                None,
                                "outright",
                                fifa,
                                bk,
                                "winner",
                                fifa,
                                None,
                                float(price),
                                captured_at,
                            )
                        )

        conn.executemany(
            """
            INSERT INTO odds (match_id, market_scope, subject_code, bookmaker,
                              market, selection, line, price, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.execute(
            "UPDATE ingest_runs SET finished_at=datetime('now'), status='success', "
            "rows_written=? WHERE id=?",
            (len(rows), run_id),
        )
    finally:
        if own_client:
            client.close()
        conn.close()

    return {
        "outright_rows_inserted": len(rows),
        "unmatched_team_names": sorted(unmatched),
        "quota_used": client.requests_used,
        "quota_remaining": client.requests_remaining,
        "captured_at": captured_at,
    }


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["match", "outright", "both"], default="both")
    ap.add_argument("--regions", default=DEFAULT_REGIONS)
    ap.add_argument("--markets", default=DEFAULT_MARKETS)
    ap.add_argument("--bookmakers", default=None, help="optional comma-sep list")
    args = ap.parse_args()

    with OddsAPIClient() as client:
        if args.mode in ("match", "both"):
            print("=== Match odds ingest ===")
            r = ingest_match_odds(
                client=client,
                regions=args.regions,
                markets=args.markets,
                bookmakers=args.bookmakers,
            )
            # Trim unmatched list for printing
            r["events_unmatched"] = (
                r["events_unmatched"][:5] if r["events_unmatched"] else []
            )
            print(json.dumps(r, indent=2))
        if args.mode in ("outright", "both"):
            print("\n=== Outright odds ingest ===")
            r = ingest_outright_odds(
                client=client, regions=args.regions, bookmakers=args.bookmakers
            )
            print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
