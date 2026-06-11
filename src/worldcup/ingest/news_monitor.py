"""Free RSS-based news monitor for injury / team-news alerts.

Why not X API: Twitter (X) API is $100-200/mo for read-only access. RSS from
mainstream sources is free + faster (news sites publish ahead of social).

Sources (all free, no auth):
    - BBC Sport football
    - Sky Sports football
    - Goal.com international football
    - Google News searches (filtered queries)
    - ESPN FC

Pipeline:
    1. Pull all configured RSS feeds (cached for 5 min to be polite).
    2. Filter to last 48 hours.
    3. Score each item: matches WC 48 team? injury keyword? player name?
    4. Emit prioritized alerts with suggested conditional-MC commands.
    5. Persist to news_alerts table for trend / dedupe across runs.

Output is meant to surface candidates for Polymarket trades:
    "Mbappé hamstring tear — out for tournament"
       → suggest:   --condition '{"team_attack_multiplier":{"FRA":0.6}}'
       → run conditional_mc → compare new FRA prob vs Polymarket → trade
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import feedparser
import httpx

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

UA = "Mozilla/5.0 (compatible; worldcup-research/1.0)"

FEEDS: dict[str, str] = {
    "BBC Sport Football":   "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "Sky Sports Football":  "https://www.skysports.com/rss/12040",
    "Goal.com":             "https://www.goal.com/feeds/en/news",
    "Google News WC injury":
        "https://news.google.com/rss/search?q=%22World+Cup+2026%22+(injury+OR+injured+OR+%22ruled+out%22)&hl=en",
    "Google News squad announcement":
        "https://news.google.com/rss/search?q=%22final+squad%22+(World+Cup+OR+international)&hl=en",
    "Google News national team injury":
        "https://news.google.com/rss/search?q=national+team+(injury+OR+%22ruled+out%22)+(World+Cup+OR+%22international+duty%22)&hl=en",
    # Off-field / team-disruption (feuds, sent-home, dropped, bust-ups) — NOT injuries.
    "Google News WC disruption":
        "https://news.google.com/rss/search?q=%22World+Cup+2026%22+(rift+OR+feud+OR+%22bust-up%22+OR+%22sent+home%22+OR+%22dropped+from%22+OR+axed+OR+%22kicked+out%22+OR+row+OR+unrest+OR+dispute)&hl=en",
    "Google News camp tension":
        "https://news.google.com/rss/search?q=national+team+(%22dressing+room%22+OR+%22training+ground%22+OR+camp)+(bust-up+OR+rift+OR+row+OR+feud+OR+dispute+OR+unrest+OR+%22sent+home%22)&hl=en",
    "Reddit r/soccer":
        "https://www.reddit.com/r/soccer/.rss",
}

# Severity-weighted keywords
KEYWORDS = {
    # Strong signals (player definitely out)
    5: ["ruled out", "season-ending", "tournament-ending", "out for season",
        "out for tournament", "miss world cup", "miss tournament", "torn acl",
        "torn meniscus", "withdrawn from", "drops out", "drop out"],
    # Medium signals (likely missing key matches)
    3: ["serious injury", "muscle tear", "hamstring tear", "needs surgery",
        "scheduled surgery", "anterior cruciate", "metatarsal", "fractured",
        "broken", "tear", "torn", "out for"],
    # Weak signals (concern but uncertain)
    1: ["injury", "injured", "fitness concern", "fitness test", "doubt",
        "doubtful", "hamstring", "knee", "ankle", "calf", "thigh", "groin",
        "suspended", "suspension", "card accumulation"],
}

# ── Off-field / team-disruption keywords (the non-injury blind spot) ─────────
# HARD = a player will actually be ABSENT for non-injury reasons (sent home,
# dropped, banned, refuses to play). These are REAL squad changes — treated like
# a moderate injury (att haircut applies, genuine lag-edge candidate).
DISRUPTION_HARD = [
    "sent home", "expelled from squad", "expelled from the squad", "axed from",
    "kicked out of squad", "dropped from squad", "dropped from the squad",
    "left out of squad", "omitted from squad", "withdrawn from squad",
    "banned for", "suspended for the tournament", "suspended for world cup",
    "refuses to play", "refused to play", "on strike", "quits national team",
    "retires from international", "frozen out of squad", "removed from squad",
]
# SOFT = morale / feud / rift narrative. NO automatic model impact — these are
# WATCH-LIST signals only (att stays 1.0). Discipline: dressing-room rumours are
# noisy, often media-炒作, and usually ALREADY priced (cf. the Mbappé thesis).
DISRUPTION_SOFT = [
    "bust-up", "training ground bust-up", "dressing room rift", "rift with",
    "feud", "fallout", "fall-out", "unrest", "row with", "training ground row",
    "unhappy", "tension in", "strained relationship", "frozen out", "exiled",
    "fell out with", "clash with", "in-fighting", "infighting", "mutiny",
    "player power", "revolt", "dressing room split", "training ground clash",
]

# Stop-words that mean "false alarm" (cures, returns)
NEG_KEYWORDS = [
    "returns from injury", "back from injury", "recovered", "fit again",
    "passed fitness test", "available for selection", "cleared to play",
    "training again",
]


def fetch_all_feeds(feeds: dict[str, str] | None = None) -> list[dict]:
    feeds = feeds or FEEDS
    items = []
    for src, url in feeds.items():
        try:
            r = httpx.get(url, headers={"User-Agent": UA}, timeout=20, follow_redirects=True)
            r.raise_for_status()
            parsed = feedparser.parse(r.content)
            for entry in parsed.entries[:40]:
                try:
                    pub = dt.datetime(*entry.published_parsed[:6], tzinfo=dt.timezone.utc)
                except Exception:
                    pub = dt.datetime.now(dt.timezone.utc)
                items.append({
                    "source": src,
                    "title": (entry.get("title") or "").strip(),
                    "url": entry.get("link") or "",
                    "summary": (entry.get("summary") or "").strip()[:400],
                    "published": pub,
                })
        except Exception as e:
            print(f"  [warn] {src}: {type(e).__name__}: {e}")
    return items


def load_team_aliases(conn: sqlite3.Connection) -> dict[str, str]:
    """Build name → fifa_code map for WC 48 teams (with common alternate names)."""
    teams = {}
    for r in conn.execute("SELECT code, name FROM teams WHERE in_worldcup_2026=1"):
        teams[r[1]] = r[0]
    # Add common alternates
    extras = {
        "USA": "USA", "United States": "USA", "U.S.": "USA",
        "Türkiye": "TUR", "Turkey": "TUR",
        "Czechia": "CZE", "Czech Republic": "CZE",
        "South Korea": "KOR", "Korea Republic": "KOR",
        "Saudi Arabia": "KSA",
        "Côte d'Ivoire": "CIV", "Ivory Coast": "CIV",
        "Bosnia": "BIH",
        "DR Congo": "COD", "Democratic Republic of the Congo": "COD",
        "Cape Verde": "CPV", "Cabo Verde": "CPV",
        "Curaçao": "CUW", "Curacao": "CUW",
        "Netherlands": "NED", "Holland": "NED", "Dutch": "NED",
        "Switzerland": "SUI", "Swiss": "SUI",
        "Germany": "GER", "German": "GER",
        "France": "FRA", "French": "FRA", "Les Bleus": "FRA",
        "England": "ENG", "English": "ENG", "Three Lions": "ENG",
        "Spain": "ESP", "Spanish": "ESP", "La Roja": "ESP",
        "Argentina": "ARG", "Argentinian": "ARG", "Albiceleste": "ARG",
        "Brazil": "BRA", "Brazilian": "BRA", "Seleção": "BRA",
        "Portugal": "POR", "Portuguese": "POR",
        "Italy": "ITA",  # not in WC but mentioned often
    }
    teams.update(extras)
    return teams


def load_known_players(conn: sqlite3.Connection) -> dict[str, str]:
    """player surname → team_code mapping for surname-based matching."""
    rows = list(conn.execute("""
        SELECT p.name, p.nationality_code FROM players p
        WHERE p.nationality_code IS NOT NULL
          AND p.name IS NOT NULL
    """))
    out = {}
    for name, code in rows:
        # Try surname (last word)
        parts = name.split()
        if len(parts) >= 2 and len(parts[-1]) > 3:
            out[parts[-1]] = code
        # Also full name
        if len(name) > 4:
            out[name] = code
    return out


def score_item(item: dict, team_aliases: dict, players: dict) -> dict | None:
    text = (item["title"] + " " + item["summary"]).lower()

    # Reject if it's a recovery/return story
    for ng in NEG_KEYWORDS:
        if ng in text:
            return None

    # Find injury keyword severity
    severity = 0
    matched_kws = []
    for sev, kws in KEYWORDS.items():
        for k in kws:
            if k in text:
                severity = max(severity, sev)
                matched_kws.append(k)
                break  # one per severity bucket

    # Off-field disruption (non-injury) — the blind spot. HARD events = a player
    # genuinely ABSENT (sent home/dropped/banned) → treat like a moderate injury.
    # SOFT events = morale/feud narrative → WATCH-LIST only, no auto model impact.
    impact_category = "injury"
    hard_hit = [k for k in DISRUPTION_HARD if k in text]
    soft_hit = [k for k in DISRUPTION_SOFT if k in text]
    if hard_hit:
        severity = max(severity, 4)          # confirmed absence ≈ moderate injury
        matched_kws += hard_hit
        impact_category = "squad_change"
    elif soft_hit and severity == 0:
        severity = 1                         # morale-only: surface to LLM/human...
        matched_kws += soft_hit
        impact_category = "morale_watch"     # ...but NEVER auto-applied to the model
    elif soft_hit:
        matched_kws += soft_hit              # soft corroborates an injury story

    if severity == 0:
        return None

    # Find team (look in title preferentially)
    team_code = None
    team_name = None
    for name, code in team_aliases.items():
        if name.lower() in text:
            team_code = code
            team_name = name
            break

    # Find player (surname match)
    player_team_code = None
    player_name = None
    for pname, pcode in players.items():
        if pname.lower() in text and len(pname) > 3:
            player_team_code = pcode
            player_name = pname
            break

    final_team = team_code or player_team_code
    if not final_team:
        return None

    # Composite score: severity × (player_match boost) × (recency)
    age_hours = (dt.datetime.now(dt.timezone.utc) - item["published"]).total_seconds() / 3600
    recency_factor = max(0.1, 1.0 - age_hours / 48)  # decays to 0.1 at 48h
    player_boost = 2.0 if player_name else 1.0
    composite = severity * player_boost * recency_factor

    # Suggested conditional MC
    if impact_category == "morale_watch":
        attack_mult = 1.0  # watch-list only — feud/morale never auto-haircuts λ
    elif severity >= 5 and final_team:
        attack_mult = 0.7  # major loss = -30% attack
    elif severity >= 3:
        attack_mult = 0.85  # moderate
    else:
        attack_mult = 0.95  # marginal
    cond_str = f'{{"team_attack_multiplier":{{"{final_team}":{attack_mult}}}}}'

    return {
        **item,
        "severity": severity,
        "impact_category": impact_category,
        "keywords": matched_kws,
        "team_code": final_team,
        "team_name": team_name,
        "player_name": player_name,
        "age_hours": round(age_hours, 1),
        "composite_score": round(composite, 2),
        "suggested_condition": cond_str,
    }


def ensure_table(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS news_alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            url           TEXT UNIQUE,
            source        TEXT,
            title         TEXT,
            published     TEXT,
            team_code     TEXT,
            player_name   TEXT,
            severity      INTEGER,
            composite     REAL,
            keywords      TEXT,
            impact_category TEXT,
            captured_at   TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_news_team ON news_alerts(team_code);
        CREATE INDEX IF NOT EXISTS idx_news_published ON news_alerts(published);
    """)
    # Migration for pre-existing DBs (table created before disruption support).
    try:
        conn.execute("ALTER TABLE news_alerts ADD COLUMN impact_category TEXT")
    except Exception:
        pass  # column already exists


def persist_alerts(alerts: list[dict], db_path) -> int:
    if not alerts: return 0
    conn = get_conn(db_path)
    ensure_table(conn)
    written = 0
    for a in alerts:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO news_alerts
                    (url, source, title, published, team_code, player_name,
                     severity, composite, keywords, impact_category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (a["url"], a["source"], a["title"], a["published"].isoformat(),
                 a["team_code"], a.get("player_name"),
                 a["severity"], a["composite_score"], ",".join(a["keywords"]),
                 a.get("impact_category", "injury")),
            )
            if conn.total_changes > written:
                written = conn.total_changes
        except Exception as e:
            print(f"  [warn] insert error: {e}")
    conn.close()
    return written


def run_monitor(db_path=DEFAULT_DB_PATH) -> dict:
    print("Fetching RSS feeds ...")
    items = fetch_all_feeds()
    print(f"  Total items: {len(items)} from {len(FEEDS)} feeds\n")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    aliases = load_team_aliases(conn)
    players = load_known_players(conn)
    conn.close()

    alerts = []
    for item in items:
        scored = score_item(item, aliases, players)
        if scored:
            alerts.append(scored)
    # Dedupe by URL
    seen = set()
    deduped = []
    for a in alerts:
        if a["url"] in seen: continue
        seen.add(a["url"])
        deduped.append(a)
    deduped.sort(key=lambda x: -x["composite_score"])

    n_persisted = persist_alerts(deduped, db_path)
    return {
        "total_items": len(items),
        "alerts_total": len(deduped),
        "persisted_new": n_persisted,
        "alerts": deduped,
    }


def main() -> None:
    import json
    result = run_monitor()
    print(f"=== News Monitor (last 48h) ===")
    print(f"  Total feed items: {result['total_items']}")
    print(f"  Alerts matched:   {result['alerts_total']}")
    print(f"  New persisted:    {result['persisted_new']}")

    alerts = result["alerts"]
    if not alerts:
        print("\n  No injury / squad alerts in last 48h. (Or feeds filtered too aggressively.)")
        return

    print(f"\n=== TOP ALERTS (sorted by severity × recency) ===\n")
    for i, a in enumerate(alerts[:10], 1):
        title = a["title"][:90]
        kw_str = ",".join(a["keywords"][:3])
        player_str = f" [{a['player_name']}]" if a.get("player_name") else ""
        cat = a.get("impact_category", "injury")
        tag = {"squad_change": "  ⚔️ SQUAD-CHANGE (off-field absence)",
               "morale_watch": "  👀 WATCH-LIST (morale/feud — NOT auto-applied)"}.get(cat, "")
        print(f"#{i}  sev={a['severity']}  score={a['composite_score']}  "
              f"{a['age_hours']}h ago  {a['team_code']}{player_str}{tag}")
        print(f"    {title}")
        print(f"    src={a['source']}  kw={kw_str}")
        print(f"    url={a['url'][:80]}")
        if a["severity"] >= 3 and cat != "morale_watch":
            print(f"    💡 SUGGESTED: python -m worldcup.simulator.conditional_mc \\\n"
                  f"          --condition '{a['suggested_condition']}' --diff")
        print()


if __name__ == "__main__":
    main()
