"""Polymarket (decentralized prediction market) ingester.

Pulls outright probabilities from polymarket.com via their public Gamma API.

Why this source matters:
    - Polymarket is real-money USDC trading on Polygon. $1.24B volume + $281M
      liquidity on the WC 2026 Winner market alone (as of 2026-05-27).
    - No bookmaker margin/vig — `lastTradePrice` is the no-vig probability.
    - Bid/ask spread on top markets is typically <1¢ (vs 2-5% vig on books).
    - Aggregates global pro + amateur capital; usually as sharp as Pinnacle
      and sometimes ahead on slower-moving info (e.g. political/health events;
      sports less of an edge but still useful as triangulation).

What we ingest:
    - WC 2026 Winner: 60 markets ("Will X win?") → implied probability per team.
    - UCL Winner: same structure.
    - Future: top-scorer event ($2M volume) when relevant.

Storage:
    - odds table, bookmaker='polymarket', market_scope='outright',
      market='winner', selection=<fifa_code>.
    - decimal_odds = 1 / yes_price (no-vig fair odds).
    - We also store `liquidity` in odds.line as a sentinel "depth" signal
      so consumers can filter for liquid markets (>$10k recommended).
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.ingest.odds_api import NAME_ALIASES as ODDS_API_ALIASES

BASE_URL = "https://gamma-api.polymarket.com"

# Polymarket uses common English names like "Spain"/"South Korea"/"Türkiye".
# Most map cleanly to teams.name; aliases here cover the remainder.
NAME_ALIASES: dict[str, str] = {
    **ODDS_API_ALIASES,  # inherit existing aliases for shared name variants
    "South Korea": "KOR",
    "USA": "USA",
    "Switzerland": "SUI",
    "Netherlands": "NED",
    "Saudi Arabia": "KSA",
    "Türkiye": "TUR",
    "Turkey": "TUR",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Ivory Coast": "CIV",
    "Bosnia and Herzegovina": "BIH",
    "Cape Verde": "CPV",
    "DR Congo": "COD",
    "Curaçao": "CUW",
    "Curacao": "CUW",
    "Bosnia-Herzegovina": "BIH",
    "Turkiye": "TUR",       # Polymarket spelling without umlaut/accent
    "Korea Republic": "KOR",
    "Cote d'Ivoire": "CIV",
}

# Regex to pull the team name out of a Polymarket "Will <Team> win the ..." question.
TEAM_FROM_QUESTION = re.compile(r"^Will\s+(.+?)\s+win\s+", re.IGNORECASE)


def _name_to_fifa_code(team_name: str, conn: sqlite3.Connection) -> str | None:
    """Resolve a Polymarket team name to our FIFA 3-letter code.

    Tries: alias map → teams.name exact → teams.name with normalized whitespace.
    """
    if team_name in NAME_ALIASES:
        v = NAME_ALIASES[team_name]
        return v if v else None  # None means explicit skip
    row = conn.execute(
        "SELECT code FROM teams WHERE name = ? LIMIT 1", (team_name,)
    ).fetchone()
    return row[0] if row else None


def fetch_event_by_slug(slug: str, client: httpx.Client) -> dict | None:
    """Fetch full event detail (includes all markets) via Polymarket Gamma API."""
    # The /events endpoint with slug match
    r = client.get(f"{BASE_URL}/events", params={"slug": slug, "limit": 1})
    r.raise_for_status()
    data = r.json()
    return data[0] if data else None


def find_event(title_contains: str, client: httpx.Client) -> dict | None:
    """Find an active sports event by partial title match."""
    r = client.get(
        f"{BASE_URL}/events",
        params={"tag": "Sports", "active": "true", "closed": "false", "limit": 100},
    )
    r.raise_for_status()
    for e in r.json():
        if title_contains.lower() in e.get("title", "").lower():
            return e
    return None


def parse_team_from_question(question: str) -> str | None:
    m = TEAM_FROM_QUESTION.match(question.strip())
    return m.group(1).strip() if m else None


def collect_outright_quotes(event: dict, conn: sqlite3.Connection) -> list[dict]:
    """For each market in an outright event, return team_code + yes_price + meta."""
    out: list[dict] = []
    unmapped = []
    for m in event.get("markets", []):
        if not m.get("active") or m.get("closed"):
            continue
        team_name = parse_team_from_question(m.get("question", ""))
        if team_name is None:
            continue
        fifa = _name_to_fifa_code(team_name, conn)
        if fifa is None:
            unmapped.append(team_name)
            continue

        # Prefer lastTradePrice; fall back to outcomePrices[0] (YES) or midpoint of bid/ask
        yes_price = None
        if m.get("lastTradePrice") is not None:
            yes_price = float(m["lastTradePrice"])
        elif m.get("outcomePrices"):
            try:
                yes_price = float(eval(m["outcomePrices"])[0])  # JSON list as string
            except Exception:
                pass
        if yes_price is None or yes_price <= 0 or yes_price >= 1:
            # Fall back to bid/ask midpoint
            bb = m.get("bestBid")
            ba = m.get("bestAsk")
            if bb is not None and ba is not None:
                yes_price = (float(bb) + float(ba)) / 2.0
        if yes_price is None or yes_price <= 0 or yes_price >= 1:
            continue

        out.append({
            "team_code": fifa,
            "team_name": team_name,
            "yes_price": yes_price,
            "decimal_odds": 1.0 / yes_price,
            "volume": float(m.get("volume") or 0),
            "liquidity": float(m.get("liquidity") or 0),
            "best_bid": float(m.get("bestBid") or 0) or None,
            "best_ask": float(m.get("bestAsk") or 0) or None,
            "market_slug": m.get("slug"),
        })
    return out, unmapped


def persist_outright(
    quotes: list[dict],
    db_path: Path | str,
    market_label: str = "winner",
    bookmaker: str = "polymarket",
) -> int:
    """Insert outright snapshots into odds table."""
    if not quotes:
        return 0
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    conn = get_conn(db_path)
    try:
        rows = [
            (
                None,                       # match_id
                "outright",                 # market_scope
                q["team_code"],             # subject_code
                bookmaker,                  # bookmaker
                market_label,               # market
                q["team_code"],             # selection (team)
                q["liquidity"],             # line (stash liquidity here as depth signal)
                q["decimal_odds"],          # price
                captured_at,
            )
            for q in quotes
        ]
        conn.executemany(
            """
            INSERT INTO odds (match_id, market_scope, subject_code, bookmaker,
                              market, selection, line, price, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)
    finally:
        conn.close()


def ingest_polymarket(
    db_path: Path | str = DEFAULT_DB_PATH,
    include_ucl: bool = True,
) -> dict[str, Any]:
    """Pull WC 2026 (and optionally UCL) outright prices from Polymarket.

    Returns a summary report.
    """
    out: dict[str, Any] = {}
    conn = get_conn(db_path)
    try:
        with httpx.Client(timeout=30) as client:
            # WC 2026 Winner. Discover by STABLE SLUG first: the event title has
            # drifted over time ("2026 FIFA World Cup Winner" → "World Cup Winner")
            # and /events hard-caps at 100 results, so a title scan silently misses
            # it once the event falls out of the first 100. Slug lookup is exact.
            wc = fetch_event_by_slug("world-cup-winner", client) or find_event("World Cup Winner", client)
            if wc:
                quotes, unmapped = collect_outright_quotes(wc, conn)
                n = persist_outright(quotes, db_path, market_label="winner")
                out["wc_2026"] = {
                    "event_title": wc["title"],
                    "event_volume_usd": wc.get("volume"),
                    "event_liquidity_usd": wc.get("liquidity"),
                    "markets_total": len(wc.get("markets", [])),
                    "quotes_persisted": n,
                    "unmapped_names": unmapped,
                }

            if include_ucl:
                ucl = find_event("UEFA Champions League Winner", client)
                if ucl:
                    quotes, unmapped = collect_outright_quotes(ucl, conn)
                    # For UCL, "team_code" won't match WC 48 codes (it's club teams).
                    # We persist with subject_code = team name (truncated/uppercased).
                    # But our current `teams` table only has nat'l teams → FK fails on these.
                    # So skip UCL persistence here; just report counts in-memory.
                    out["ucl"] = {
                        "event_title": ucl["title"],
                        "event_volume_usd": ucl.get("volume"),
                        "event_liquidity_usd": ucl.get("liquidity"),
                        "markets_total": len(ucl.get("markets", [])),
                        "quotes_found": len(quotes),
                        "note": "UCL clubs not in WC teams table; not persisted to odds (ad-hoc query only).",
                    }
    finally:
        conn.close()
    return out


def main() -> None:
    import json

    print("=== Polymarket ingest ===")
    result = ingest_polymarket()
    # Trim noisy fields in print
    if "wc_2026" in result:
        result["wc_2026"]["unmapped_names"] = result["wc_2026"]["unmapped_names"][:10]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
