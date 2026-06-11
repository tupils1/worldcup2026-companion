"""RLM / crowd-divergence signal — the China-accessible analog of reverse line movement.

The 'read 庄家意图' question, made concrete + honest. We can't read intent, but we CAN
read where the CROWD is vs where the SHARP line is. Two uses given our venues:

1. CROWD-ERROR detector (竞彩): crowd piles on outcome X but the sharp no-vig line gives X a
   LOW probability → crowd is on the wrong side. Reverse-line-movement: if the SHARP line
   ALSO moved away from the crowd's side, informed money confirms. (竞彩's own odds are fixed
   post-open, so 'line move' = the sharp reference moving, not 竞彩.)
2. 任九 co-winner input: crowd% IS the 选择比例 the pari-mutuel optimiser needed.

DATA: 投注比例 from 500 (Chrome-MCP scrape near close, or manual `record`). HONEST CAVEAT:
500's OWN users (a sample/proxy), not the national pool; populates only near deadline.
SHARP reference: our odds DB (Pinnacle/Betfair de-vig) — available for WC matches.

Run:
    PYTHONPATH=src python -m worldcup.strategy.rlm record --match USA-PAR --home 62 --draw 24 --away 14
    PYTHONPATH=src python -m worldcup.strategy.rlm analyze
    PYTHONPATH=src python -m worldcup.strategy.rlm demo
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.strategy.value_bets import devig_shin

DDL = """
CREATE TABLE IF NOT EXISTS bet_ratios (
    match TEXT, source TEXT, home_pct REAL, draw_pct REAL, away_pct REAL,
    captured_at TEXT, PRIMARY KEY (match, source, captured_at)
);
"""
SHARP_BOOKS = ("pinnacle", "betfair_ex_eu", "betfair_ex_uk", "betfair_ex_au")
OUT = ("home", "draw", "away")


def rlm_signal(crowd: dict, sharp: dict) -> dict:
    """crowd/sharp are {home,draw,away}. Returns most over-backed outcome + contrarian read."""
    over = {k: crowd[k] - sharp[k] for k in OUT}      # crowd − sharp (positive = crowd over-backs)
    crowd_side = max(over, key=over.get)
    contra_side = min(over, key=over.get)
    return {"crowd_side": crowd_side, "over_pp": over[crowd_side] * 100,
            "contra_side": contra_side, "contra_gap_pp": -over[contra_side] * 100, "over": over}


def sharp_1x2(conn, match_id):
    rows = conn.execute("""
        SELECT o.bookmaker,o.selection,o.price FROM odds o
        JOIN (SELECT bookmaker,selection,MAX(captured_at) mc FROM odds
              WHERE match_id=? AND market_scope='match' AND market='1X2' GROUP BY bookmaker,selection) l
          ON o.bookmaker=l.bookmaker AND o.selection=l.selection AND o.captured_at=l.mc
        WHERE o.match_id=? AND o.market_scope='match' AND o.market='1X2'
    """, (match_id, match_id)).fetchall()
    byb = defaultdict(dict)
    for r in rows:
        if r["bookmaker"] in SHARP_BOOKS and r["price"] and r["price"] > 1:
            byb[r["bookmaker"]][r["selection"]] = r["price"]
    per = defaultdict(list)
    for q in byb.values():
        if all(k in q for k in OUT):
            for k, p in zip(OUT, devig_shin([q["home"], q["draw"], q["away"]])):
                per[k].append(p)
    return {k: float(np.mean(v)) for k, v in per.items()} if per else None


def line_move(conn, match_id):
    """Sharp home-implied change open→latest (pp). + = line drifted toward home."""
    rows = conn.execute("""
        SELECT captured_at, AVG(1.0/price) inv FROM odds
        WHERE match_id=? AND market_scope='match' AND market='1X2' AND selection='home'
          AND bookmaker IN ('pinnacle','betfair_ex_eu','betfair_ex_uk','betfair_ex_au')
        GROUP BY captured_at ORDER BY captured_at
    """, (match_id,)).fetchall()
    return (rows[-1]["inv"] - rows[0]["inv"]) * 100 if len(rows) >= 2 else None


def cmd_record(args):
    conn = get_conn(DEFAULT_DB_PATH); conn.executescript(DDL)
    conn.execute("INSERT OR REPLACE INTO bet_ratios VALUES (?,?,?,?,?,?)",
                 (args.match, args.source, args.home, args.draw, args.away,
                  dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")))
    conn.commit(); conn.close()
    print(f"recorded crowd% {args.match}: H{args.home}/D{args.draw}/A{args.away} ({args.source})")


def cmd_analyze(args):
    conn = get_conn(DEFAULT_DB_PATH); conn.executescript(DDL)
    rows = list(conn.execute("SELECT * FROM bet_ratios"))
    if not rows:
        print("No 投注比例 recorded. Scrape 500 (Chrome MCP) or `record` manually, then re-run.")
        conn.close(); return
    idmap = {f"{r['home_code']}-{r['away_code']}": r["id"]
             for r in conn.execute("SELECT id,home_code,away_code FROM matches WHERE finished=0")}
    print(f"{'match':<11}{'crowd HDA':<14}{'sharp HDA':<14}{'over-backed':<15}{'value side':<14}{'lineMove'}")
    print("-" * 88)
    for r in rows:
        mid = idmap.get(r["match"])
        sh = sharp_1x2(conn, mid) if mid else None
        if not sh:
            print(f"{r['match']:<11}  (no sharp 1X2 in DB — WC matches only)"); continue
        s = (r["home_pct"] + r["draw_pct"] + r["away_pct"]) or 100
        crowd = {"home": r["home_pct"]/s, "draw": r["draw_pct"]/s, "away": r["away_pct"]/s}
        sig = rlm_signal(crowd, sh)
        lm = line_move(conn, mid)
        conf = ""
        if lm is not None and sig["crowd_side"] in ("home", "away"):
            moved_to = "home" if lm > 0 else "away"
            if sig["crowd_side"] != moved_to:
                conf = " ✓RLM"
        cstr = f"{crowd['home']*100:.0f}/{crowd['draw']*100:.0f}/{crowd['away']*100:.0f}"
        sstr = f"{sh['home']*100:.0f}/{sh['draw']*100:.0f}/{sh['away']*100:.0f}"
        ob = f"{sig['crowd_side']} +{sig['over_pp']:.0f}pp"
        vs = f"{sig['contra_side']} ({sig['contra_gap_pp']:.0f}pp)"
        lmstr = (f"{lm:+.1f}pp" if lm is not None else "—") + conf
        print(f"{r['match']:<11}{cstr:<14}{sstr:<14}{ob:<15}{vs:<14}{lmstr}")
    conn.close()
    print("\nover-backed = crowd most ABOVE sharp (crowd's favourite). value side = crowd most BELOW")
    print("sharp (contrarian value, if your venue's price lags). ✓RLM = sharp line also moved away")
    print("from the crowd → informed money confirms. Crowd% = 500-users proxy; act with the GATE.")


def cmd_demo(args):
    crowd = {"home": 0.70, "draw": 0.18, "away": 0.12}
    sharp = {"home": 0.55, "draw": 0.26, "away": 0.19}
    sig = rlm_signal(crowd, sharp)
    print("DEMO (illustrative): crowd over-loads the favourite vs sharp")
    print("  crowd H70/D18/A12  vs  sharp H55/D26/A19")
    print(f"  → over-backed: {sig['crowd_side']} +{sig['over_pp']:.0f}pp (crowd too heavy on home)")
    print(f"  → value side:  {sig['contra_side']} (crowd {sig['contra_gap_pp']:.0f}pp UNDER sharp = contrarian)")
    print("  If sharp line ALSO drifted toward away/draw = ✓RLM (informed money agrees).")
    print("  Live: powers 任九 contrarian (crowd% = the 选择比例 input) + flags 竞彩 crowd errors.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("record")
    rc.add_argument("--match", required=True); rc.add_argument("--source", default="500users")
    rc.add_argument("--home", type=float, required=True); rc.add_argument("--draw", type=float, required=True)
    rc.add_argument("--away", type=float, required=True); rc.set_defaults(func=cmd_record)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
