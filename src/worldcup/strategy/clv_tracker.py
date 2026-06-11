"""CLV (Closing Line Value) + line-movement tracker — the edge truth-serum.

Winning money short-term is noise. The gold-standard test of whether you ACTUALLY have
edge is CLV: did your entry price consistently beat the CLOSING line (the sharpest
estimate)? Positive CLV → you're on the right side of where informed money moves → real
edge. No CLV → you don't have edge; wins were luck. This logs our bets and scores them.

CLV metric = beat-close EV = P_close_novig × entry_decimal_odds − 1.
  (If the closing no-vig line is "truth", this is your bet's EV at the price you got.)
  > 0 → you beat the close (+CLV). Aggregate: mean CLV and % of bets with +CLV.

Closing line resolved from our odds DB snapshots (latest before kickoff) via a benchmark
key per bet:
  pinn_group:CODE / pinn_group_NO:CODE   — Pinnacle group-winner de-vig (within group)
  champ:CODE / champ_NO:CODE             — Betfair/book champion de-vig
  manual:P                               — user-supplied closing no-vig prob

Run:
    PYTHONPATH=src python -m worldcup.strategy.clv_tracker log --venue polymarket \\
        --label "ENG NO win Group L" --entry-odds 3.23 --stake 8.70 --bench pinn_group_NO:ENG
    PYTHONPATH=src python -m worldcup.strategy.clv_tracker clv         # score all logged bets
    PYTHONPATH=src python -m worldcup.strategy.clv_tracker movement --champ POR   # line time-series
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.strategy.value_bets import devig_shin, devig_proportional

TEAMS_YAML = Path(__file__).resolve().parents[3] / "configs" / "teams.yaml"
DDL = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY,
    placed_at TEXT, venue TEXT, label TEXT,
    entry_odds REAL, stake REAL, bench TEXT,
    close_prob REAL, clv_pct REAL, scored_at TEXT
);
"""


def _groups():
    cfg = yaml.safe_load(TEAMS_YAML.read_text())
    return {l: g["teams"] for l, g in cfg["groups"].items()}


def _latest(conn, scope, market):
    """Latest price per (bookmaker, selection) for a scope/market."""
    return conn.execute("""
        SELECT o.bookmaker, o.selection, o.price FROM odds o
        JOIN (SELECT bookmaker, selection, MAX(captured_at) mc FROM odds
              WHERE market_scope=? AND market=? GROUP BY bookmaker, selection) l
          ON o.bookmaker=l.bookmaker AND o.selection=l.selection AND o.captured_at=l.mc
        WHERE o.market_scope=? AND o.market=?
    """, (scope, market, scope, market)).fetchall()


def resolve_close_prob(conn, bench: str):
    """Closing no-vig probability the bet pays on, from the latest DB snapshot."""
    kind, _, arg = bench.partition(":")
    if kind == "manual":
        try: return float(arg)
        except ValueError: return None

    if kind in ("pinn_group", "pinn_group_NO"):
        groups = _groups()
        grp = next((L for L, ts in groups.items() if arg in ts), None)
        if not grp: return None
        rows = {r["selection"]: r["price"] for r in _latest(conn, "group_winner", "group_winner")
                if r["bookmaker"] == "pinnacle"}
        sub = {t: rows[t] for t in groups[grp] if t in rows and rows[t] > 1}
        if len(sub) < 3 or arg not in sub: return None
        dv = dict(zip(sub.keys(), devig_proportional(list(sub.values()))))
        return (1 - dv[arg]) if kind.endswith("_NO") else dv[arg]

    if kind in ("champ", "champ_NO"):
        by_book = defaultdict(dict)
        for r in _latest(conn, "outright", "winner"):
            if r["price"] and r["price"] > 1: by_book[r["bookmaker"]][r["selection"]] = r["price"]
        prefer = [b for b in by_book if b.startswith("betfair_ex_")] or \
                 [b for b in by_book if b != "polymarket"]
        per = defaultdict(list)
        for b in prefer:
            q = by_book[b]
            if len(q) >= 10:
                for t, pp in zip(q.keys(), devig_shin(list(q.values()))): per[t].append(pp)
        if arg not in per: return None
        p = float(np.mean(per[arg]))
        return (1 - p) if kind.endswith("_NO") else p
    return None


def cmd_log(args):
    conn = get_conn(DEFAULT_DB_PATH); conn.executescript(DDL)
    conn.execute("INSERT INTO bets (placed_at,venue,label,entry_odds,stake,bench) VALUES (?,?,?,?,?,?)",
                 (dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  args.venue, args.label, args.entry_odds, args.stake, args.bench))
    conn.commit(); conn.close()
    print(f"logged: {args.label}  @{args.entry_odds}  stake ${args.stake}  bench={args.bench}")


def cmd_clv(args):
    conn = get_conn(DEFAULT_DB_PATH); conn.executescript(DDL)
    bets = list(conn.execute("SELECT * FROM bets"))
    if not bets:
        print("No bets logged. Use `log` first."); conn.close(); return
    print(f"{'label':<28}{'venue':<11}{'@entry':>7}{'P_close':>8}{'CLV':>8}{'$stake':>7}")
    print("-" * 70)
    clvs = []
    for b in bets:
        pc = resolve_close_prob(conn, b["bench"] or "")
        if pc is None:
            print(f"{(b['label'] or '')[:28]:<28}{b['venue']:<11}{b['entry_odds']:>7.2f}{'—':>8}{'unresolved':>9}")
            continue
        clv = pc * b["entry_odds"] - 1.0
        clvs.append(clv)
        conn.execute("UPDATE bets SET close_prob=?, clv_pct=?, scored_at=datetime('now') WHERE id=?",
                     (pc, clv * 100, b["id"]))
        print(f"{(b['label'] or '')[:28]:<28}{b['venue']:<11}{b['entry_odds']:>7.2f}{pc*100:>7.1f}%{clv*100:>+7.1f}%{b['stake'] or 0:>7.0f}")
    conn.commit(); conn.close()
    if clvs:
        pos = sum(1 for c in clvs if c > 0)
        print("-" * 70)
        print(f"  n={len(clvs)}  mean CLV {np.mean(clvs)*100:+.2f}%  +CLV rate {pos/len(clvs)*100:.0f}%")
        print("  VERDICT: mean CLV >0 AND +CLV rate >55% (over MANY bets) = real edge; else luck/no edge.")
        print("  (Closing line = latest DB snapshot; for true CLV refresh odds right before kickoff.)")


def cmd_movement(args):
    conn = get_conn(DEFAULT_DB_PATH)
    if args.champ:
        rows = conn.execute("""SELECT bookmaker,price,captured_at FROM odds
            WHERE market_scope='outright' AND market='winner' AND selection=?
            ORDER BY captured_at""", (args.champ,)).fetchall()
        print(f"Champion line movement — {args.champ} (decimal, implied%):")
        for r in rows[-24:]:
            print(f"  {r['captured_at'][:16]}  {r['bookmaker']:<14} {r['price']:6.2f}  ({100/r['price']:.1f}%)")
    elif args.group:
        rows = conn.execute("""SELECT price,captured_at FROM odds
            WHERE market_scope='group_winner' AND bookmaker='pinnacle' AND selection=?
            ORDER BY captured_at""", (args.group,)).fetchall()
        print(f"Pinnacle group-winner movement — {args.group}:")
        for r in rows[-24:]:
            print(f"  {r['captured_at'][:16]}  {r['price']:6.2f}  ({100/r['price']:.1f}%)")
    else:
        print("specify --champ CODE or --group CODE")
    conn.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("--venue", required=True); lg.add_argument("--label", required=True)
    lg.add_argument("--entry-odds", type=float, required=True, dest="entry_odds")
    lg.add_argument("--stake", type=float, default=0.0); lg.add_argument("--bench", required=True)
    lg.set_defaults(func=cmd_log)
    sub.add_parser("clv").set_defaults(func=cmd_clv)
    mv = sub.add_parser("movement"); mv.add_argument("--champ"); mv.add_argument("--group")
    mv.set_defaults(func=cmd_movement)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
