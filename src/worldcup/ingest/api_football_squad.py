"""API-Football structured injuries + lineups — feeds the injury-lag edge (better than RSS).

Squeezes the (paid, Ultra) API-Football endpoints we never used:
  /injuries        — structured {player, team, reason, fixture}. No headline traps (vs RSS,
                     which misread "Kudus boost for England" + a recovery story as injuries).
  /fixtures/lineups — confirmed starting XI ~40 min pre-match (the biggest pre-match info jump).

INTEGRATION: structured injuries are stored to `injuries` AND written as clean news_alerts
rows (source='api_football') so the EXISTING llm_news_scorer + elite-squad clamp + injury_lag
chain picks them up automatically — same pipeline, cleaner source.

Lineups: `fetch_lineup` returns the XI; `lineup_absences` flags key players NOT starting
(needs a key-player list per team — pass via key_players, or build one; player_market_values
is empty so no auto baseline yet).

Run:
    PYTHONPATH=src python -m worldcup.ingest.api_football_squad            # ingest WC injuries
    PYTHONPATH=src python -m worldcup.ingest.api_football_squad --fixture 1489370   # lineup
"""
from __future__ import annotations
import argparse, datetime as dt, time
from pathlib import Path
import httpx
from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

BASE = "https://v3.football.api-sports.io"
SECRETS = Path(__file__).resolve().parents[3] / "configs" / "secrets.env"


def _key():
    for ln in SECRETS.read_text().splitlines():
        if ln.startswith("API_FOOTBALL_KEY="):
            return ln.split("=", 1)[1].split("#")[0].strip()


def _get(path, **params):
    for _ in range(3):
        try:
            return httpx.get(BASE + path, params=params, headers={"x-apisports-key": _key()}, timeout=25).json()
        except Exception:
            time.sleep(1)
    return {"response": [], "errors": ["conn"]}


def _af_to_fifa(conn):
    return {int(r["source_code"]): r["fifa_code"] for r in conn.execute(
        "SELECT source_code, fifa_code FROM team_code_map WHERE source='api_football'")}


def _seed_severity(reason: str) -> int:
    r = (reason or "").lower()
    if any(k in r for k in ("acl", "season", "surgery", "ruptur", "broken", "out for")): return 5
    if any(k in r for k in ("doubt", "question", "knock", "fitness", "late test")): return 3
    return 4  # default: a listed injury = likely missing


def _internal_player_id(conn, af_pid, name: str | None, code: str) -> int | None:
    """Map an API-Football player id to OUR players.id (the injuries FK target).
    Writing the AF id straight into the FK column either points at a random row
    or — with PRAGMA foreign_keys=ON — aborts the whole ingest batch."""
    if af_pid is not None:
        r = conn.execute("SELECT id FROM players WHERE api_football_id=?", (af_pid,)).fetchone()
        if r:
            return r["id"]
    if name:
        r = conn.execute("SELECT id FROM players WHERE name=? AND nationality_code=?",
                         (name, code)).fetchone()
        if r:
            if af_pid is not None:
                conn.execute("UPDATE players SET api_football_id=COALESCE(api_football_id,?) WHERE id=?",
                             (af_pid, r["id"]))
            return r["id"]
    if not (name or af_pid is not None):
        return None
    return conn.execute("INSERT INTO players (api_football_id, name, nationality_code) VALUES (?,?,?)",
                        (af_pid, name or f"AF#{af_pid}", code)).lastrowid


def ingest_injuries(league: int = 1, season: int = 2026, db_path=DEFAULT_DB_PATH) -> dict:
    conn = get_conn(db_path)
    try:
        af = _af_to_fifa(conn)
        data = _get("/injuries", league=league, season=season)
        resp = data.get("response", [])
        cap = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        n_inj = n_news = 0
        n_bad = 0
        for x in resp:
            try:
                tid = x.get("team", {}).get("id")
                code = af.get(int(tid)) if tid else None
                if not code:
                    continue
                pl = x.get("player", {}) or {}
                pname = pl.get("name"); reason = pl.get("reason") or pl.get("type") or ""
                pid = _internal_player_id(conn, pl.get("id"), pname, code)
                if pid is None:
                    continue
                conn.execute("INSERT INTO injuries (player_id,team_code,status,detail,source,captured_at) "
                             "VALUES (?,?,?,?,?,?)",
                             (pid, code, pl.get("type") or "injury", reason, "api_football", cap))
                n_inj += 1
                # clean news_alerts row → existing LLM scorer + clamp + injury_lag will consume it
                title = f"{pname} ({code}) — {reason}"
                exists = conn.execute("SELECT 1 FROM news_alerts WHERE source='api_football' AND title=?",
                                      (title,)).fetchone()
                if not exists:
                    conn.execute("""INSERT INTO news_alerts (url,source,title,published,team_code,player_name,
                                    severity,composite,keywords,captured_at)
                                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                                 (f"apifootball://injury/{pl.get('id')}", "api_football", title, cap, code, pname,
                                  _seed_severity(reason), 0.0, "injury", cap))
                    n_news += 1
            except Exception as e:  # one bad row must not sink the whole batch
                n_bad += 1
                print(f"  [injuries] skipped one row ({type(e).__name__}: {e})")
        conn.commit()
        return {"league": league, "season": season, "injuries_fetched": len(resp),
                "injuries_stored": n_inj, "news_rows_added": n_news, "rows_skipped": n_bad,
                "errors": data.get("errors")}
    finally:
        conn.close()


def fetch_lineup(fixture_id: int) -> dict:
    """{team_name: {formation, startXI: [names], subs: [names]}} — confirmed XI (~40min pre-match)."""
    data = _get("/fixtures/lineups", fixture=fixture_id)
    out = {}
    for t in data.get("response", []):
        out[t.get("team", {}).get("name")] = {
            "formation": t.get("formation"),
            "startXI": [p.get("player", {}).get("name") for p in t.get("startXI", [])],
            "subs": [p.get("player", {}).get("name") for p in t.get("substitutes", [])],
        }
    return out


def lineup_absences(fixture_id: int, key_players: dict[str, list[str]]) -> dict:
    """key_players = {team_name: [key player names]}. Returns key players NOT in the XI = the
    actionable lineup-lag signal (star benched/out → re-run conditional_mc, check slow venues)."""
    lu = fetch_lineup(fixture_id)
    out = {}
    for team, kp in key_players.items():
        xi = set(lu.get(team, {}).get("startXI", []))
        if xi:  # only if lineup is out
            missing = [p for p in kp if p not in xi]
            if missing:
                out[team] = missing
    return out


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=int, help="fetch + print a fixture's lineup")
    ap.add_argument("--league", type=int, default=1)
    ap.add_argument("--season", type=int, default=2026)
    args = ap.parse_args()
    if args.fixture:
        print(json.dumps(fetch_lineup(args.fixture), ensure_ascii=False, indent=2))
        return
    print("=== API-Football structured injuries ingest ===")
    print(json.dumps(ingest_injuries(args.league, args.season), ensure_ascii=False, indent=2))
    print("\n(WC injuries populate during the tournament; rows feed llm_news_scorer → injury_lag.)")


if __name__ == "__main__":
    main()
