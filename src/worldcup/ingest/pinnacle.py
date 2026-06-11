"""Pinnacle guest API ingester — sharp book sub-markets.

Pinnacle's website frontend uses a public guest API (no login required) that
exposes EVERY market they offer for the WC, including the sub-markets that
The Odds API and Polymarket don't carry:

    - Group Winner (all 12 groups)
    - To Qualify From Group (advancement)
    - To Finish Bottom of Group
    - Group Stage team total points
    - Team Props: BTTS, Half-Time/Full-Time, Winner+Total combos
      (these ARE the correlated-parlay markets!)
    - Player props, Futures (champion)

Why this matters: Pinnacle is the sharpest book in the world (lowest margin,
highest limits, no player-banning). Their group-winner/qualify prices are the
best available "true" probability benchmark for sub-markets. Free + no auth.

Endpoint: https://guest.api.arcadia.pinnacle.com/0.1
Key (public, used by their own frontend): CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R
WC 2026 league id: 2686
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

GUEST_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"
ARCADIA = "https://guest.api.arcadia.pinnacle.com/0.1"
WC_LEAGUE_ID = 2686

HEADERS = {
    "X-API-Key": GUEST_KEY,
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Accept": "application/json",
    "Referer": "https://www.pinnacle.com/",
    "Origin": "https://www.pinnacle.com",
}

# Pinnacle team name → FIFA code
NAME_TO_FIFA: dict[str, str] = {
    "Mexico": "MEX", "South Africa": "RSA", "South Korea": "KOR", "Czechia": "CZE",
    "Canada": "CAN", "Switzerland": "SUI", "Qatar": "QAT", "Bosnia and Herzegovina": "BIH",
    "Bosnia & Herzegovina": "BIH",
    "Brazil": "BRA", "Morocco": "MAR", "Scotland": "SCO", "Haiti": "HAI",
    "United States": "USA", "USA": "USA", "Australia": "AUS", "Paraguay": "PAR", "Turkey": "TUR",
    "Turkiye": "TUR", "Türkiye": "TUR",
    "Germany": "GER", "Ecuador": "ECU", "Ivory Coast": "CIV", "Cote d'Ivoire": "CIV", "Curacao": "CUW",
    "Curaçao": "CUW",
    "Netherlands": "NED", "Japan": "JPN", "Tunisia": "TUN", "Sweden": "SWE",
    "Belgium": "BEL", "Egypt": "EGY", "Iran": "IRN", "New Zealand": "NZL",
    "Spain": "ESP", "Uruguay": "URU", "Cape Verde": "CPV", "Cabo Verde": "CPV", "Saudi Arabia": "KSA",
    "France": "FRA", "Senegal": "SEN", "Norway": "NOR", "Iraq": "IRQ",
    "Argentina": "ARG", "Austria": "AUT", "Algeria": "ALG", "Jordan": "JOR",
    "Portugal": "POR", "Colombia": "COL", "Uzbekistan": "UZB", "DR Congo": "COD", "Congo DR": "COD",
    "England": "ENG", "Croatia": "CRO", "Panama": "PAN", "Ghana": "GHA",
}


def american_to_decimal(american: float) -> float:
    if american > 0:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


class PinnacleClient:
    def __init__(self, retries: int = 4, pause: float = 1.2):
        self.retries = retries
        self.pause = pause

    def get(self, path: str) -> Any:
        url = f"{ARCADIA}{path}"
        for i in range(self.retries):
            try:
                with httpx.Client(headers=HEADERS, timeout=20) as c:
                    r = c.get(url)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if i == self.retries - 1:
                    raise
                time.sleep(self.pause)

    def matchups(self, league: int = WC_LEAGUE_ID) -> list[dict]:
        return self.get(f"/leagues/{league}/matchups")

    def markets(self, league: int = WC_LEAGUE_ID) -> list[dict]:
        return self.get(f"/leagues/{league}/markets/straight")


def _classify_special(desc: str) -> tuple[str, str | None] | None:
    """Map a Pinnacle special description to (market_type, group_letter)."""
    d = desc.lower()
    if "winner" in d and "group" in d:
        # "Group A Winner"
        for letter in "ABCDEFGHIJKL":
            if f"group {letter.lower()}" in d:
                return ("group_winner", letter)
        return ("group_winner", None)
    if "to qualify" in d:
        return ("to_qualify", None)
    if "finish bottom" in d:
        return ("group_bottom", None)
    return None


def ingest_pinnacle(db_path: Path | str = DEFAULT_DB_PATH) -> dict:
    """Pull Pinnacle group-winner + to-qualify markets, persist to odds table."""
    client = PinnacleClient()
    matchups = client.matchups()
    markets = client.markets()

    # Index prices by matchupId
    market_by_matchup: dict[int, list] = defaultdict(list)
    for m in markets:
        market_by_matchup[m.get("matchupId")].append(m)

    captured_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows: list[tuple] = []
    unmapped_names: set[str] = set()
    stats = defaultdict(int)

    for mu in matchups:
        special = mu.get("special")
        if not isinstance(special, dict):
            continue
        desc = special.get("description", "")
        cls = _classify_special(desc)
        if cls is None:
            continue
        market_type, group_letter = cls

        pid_to_name = {p["id"]: p.get("name", "") for p in mu.get("participants", [])}
        mks = [mk for mk in market_by_matchup.get(mu["id"], []) if mk.get("type") == "moneyline"]
        if not mks:
            continue

        if market_type == "to_qualify":
            # Team is in the description; participants are Yes/No.
            team_name = desc.split(" To Qualify")[0].strip()
            fifa = NAME_TO_FIFA.get(team_name)
            if fifa is None:
                if team_name:
                    unmapped_names.add(team_name)
                continue
            for mk in mks:
                for price in mk.get("prices", []):
                    pid = price.get("participantId")
                    american = price.get("price")
                    if pid is None or american is None:
                        continue
                    pname = pid_to_name.get(pid, "").strip().lower()
                    if pname == "yes":  # P(team qualifies)
                        rows.append((None, "to_qualify", fifa, "pinnacle", "to_qualify",
                                     fifa, None, american_to_decimal(american), captured_at))
                        stats["to_qualify"] += 1
        else:
            # group_winner / group_bottom: each participant is a team
            for mk in mks:
                for price in mk.get("prices", []):
                    pid = price.get("participantId")
                    american = price.get("price")
                    if pid is None or american is None:
                        continue
                    name = pid_to_name.get(pid, "")
                    fifa = NAME_TO_FIFA.get(name)
                    if fifa is None:
                        if name and "winner" not in name.lower() and "path" not in name.lower():
                            unmapped_names.add(name)
                        continue
                    rows.append((None, market_type, fifa, "pinnacle", market_type,
                                 fifa, None, american_to_decimal(american), captured_at))
                    stats[market_type] += 1

    conn = get_conn(db_path)
    try:
        run_id = conn.execute(
            "INSERT INTO ingest_runs (source) VALUES (?)",
            ("pinnacle:guest-api/leagues/2686",),
        ).lastrowid
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
        conn.close()

    return {
        "matchups_total": len(matchups),
        "markets_total": len(markets),
        "rows_persisted": len(rows),
        "by_market_type": dict(stats),
        "unmapped_names": sorted(unmapped_names),
        "captured_at": captured_at,
    }


def main() -> None:
    import json
    print("=== Pinnacle guest API ingest ===")
    result = ingest_pinnacle()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
