#!/usr/bin/env python
"""Auto value detector: our MC model vs Pinnacle group/qualify markets.

Pinnacle is the sharpest book. Their group-winner + to-qualify markets are the
best benchmark available for sub-markets (which Polymarket/Odds API don't carry).
This script:
    1. Runs MC → model P(win group) + P(advance) per team.
    2. Reads latest Pinnacle odds from DB (ingested by worldcup.ingest.pinnacle).
    3. De-vigs Pinnacle per group, compares to model, computes edge + Kelly.

Because Pinnacle is SHARP, large model-vs-Pinnacle gaps are more likely MODEL
error than value. Treat 3-8% gaps as candidates, >10% gaps as model bias to
investigate. The real edge is when SOFTER books deviate from Pinnacle — so this
report doubles as a calibration check (model vs the sharpest sub-market price).

Run:
    PYTHONPATH=src python scripts/pinnacle_value.py --n-sims 50000 --bankroll 1000
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.strategy.group_winner_value import compute_group_probs
from worldcup.strategy.value_bets import devig_shin, kelly_fraction

ROOT = Path(__file__).resolve().parents[1]


def load_pinnacle_odds(db_path=DEFAULT_DB_PATH) -> dict[str, dict[str, float]]:
    """Returns {market: {team: decimal_odds}} for latest Pinnacle snapshot."""
    conn = sqlite3.connect(str(db_path))
    out = defaultdict(dict)
    for market in ("group_winner", "to_qualify"):
        rows = conn.execute("""
            SELECT subject_code, price FROM odds
            WHERE bookmaker='pinnacle' AND market=?
              AND captured_at = (SELECT MAX(captured_at) FROM odds WHERE bookmaker='pinnacle' AND market=?)
        """, (market, market)).fetchall()
        for code, price in rows:
            out[market][code] = price
    conn.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=50000)
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--bankroll", type=float, default=1000)
    ap.add_argument("--min-edge", type=float, default=0.03)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "teams.yaml").read_text())
    groups = {l: g["teams"] for l, g in cfg["groups"].items()}
    code_to_group = {c: l for l, g in groups.items() for c in g}

    print(f"Computing model group probabilities ({args.n_sims:,} sims) ...")
    model = compute_group_probs(n_sims=args.n_sims, prior=args.prior)
    pin = load_pinnacle_odds()
    print(f"Pinnacle odds loaded: {len(pin['group_winner'])} group-winner, "
          f"{len(pin['to_qualify'])} to-qualify\n")

    # ─── GROUP WINNER ───
    print("=" * 92)
    print("①  GROUP WINNER: model vs Pinnacle (de-vigged)")
    print("=" * 92)
    gw_bets = []
    for letter in sorted(groups):
        teams = groups[letter]
        odds = {t: pin["group_winner"].get(t) for t in teams}
        if any(odds[t] is None for t in teams):
            continue
        devig = devig_shin([odds[t] for t in teams])
        pin_p = dict(zip(teams, devig))
        print(f"\nGroup {letter}:")
        print(f"  {'team':<5} {'Pin odds':>9} {'Pin(devig)':>11} {'model':>8} {'edge':>7} {'Kelly1/4$':>10}")
        rows = []
        for t in teams:
            mp = model[letter][t]["win"]
            pp = pin_p[t]
            edge = mp - pp
            kelly = kelly_fraction(mp, odds[t], scaling=0.25)
            rows.append((t, odds[t], pp, mp, edge, kelly))
        rows.sort(key=lambda r: -r[4])
        for t, o, pp, mp, edge, kelly in rows:
            stake = args.bankroll * kelly
            mark = "  ✅" if edge >= args.min_edge else ("  ⚠️ model-low" if edge <= -0.08 else "")
            print(f"  {t:<5} {o:9.2f} {pp*100:10.1f}% {mp*100:7.1f}% {edge*100:+6.1f}% ${stake:9.2f}{mark}")
            if edge >= args.min_edge:
                gw_bets.append((letter, t, o, mp, pp, edge, stake))

    # ─── TO QUALIFY ───
    print("\n" + "=" * 92)
    print("②  TO QUALIFY (advance): model vs Pinnacle")
    print("=" * 92)
    q_bets = []
    for letter in sorted(groups):
        teams = groups[letter]
        odds = {t: pin["to_qualify"].get(t) for t in teams}
        if any(odds[t] is None for t in teams):
            continue
        # Each "To Qualify" is an independent Yes/No market with its own vig.
        # Pinnacle 2-way vig ≈ 2.5%; haircut the raw implied to approximate true P.
        # (Proper de-vig needs the No price, which we don't store separately.)
        pin_p = {t: min(1.0 / odds[t] * 0.975, 0.995) for t in teams}
        print(f"\nGroup {letter}:")
        print(f"  {'team':<5} {'Pin odds':>9} {'Pin(adj)':>9} {'model':>8} {'edge':>7} {'Kelly1/4$':>10}")
        rows = []
        for t in teams:
            mp = model[letter][t]["advance"]
            pp = min(pin_p[t], 0.99)
            edge = mp - pp
            kelly = kelly_fraction(mp, odds[t], scaling=0.25)
            rows.append((t, odds[t], pp, mp, edge, kelly))
        rows.sort(key=lambda r: -r[4])
        for t, o, pp, mp, edge, kelly in rows:
            stake = args.bankroll * kelly
            mark = "  ✅" if edge >= args.min_edge else ("  ⚠️ model-low" if edge <= -0.08 else "")
            print(f"  {t:<5} {o:9.2f} {pp*100:8.1f}% {mp*100:7.1f}% {edge*100:+6.1f}% ${stake:9.2f}{mark}")
            if edge >= args.min_edge:
                q_bets.append((letter, t, o, mp, pp, edge, stake))

    # ─── SUMMARY ───
    print("\n" + "=" * 92)
    print("③  VALUE BET SUMMARY (model edge ≥ {:.0%} vs SHARP Pinnacle)".format(args.min_edge))
    print("=" * 92)
    all_bets = [("GroupWin", *b) for b in gw_bets] + [("Qualify", *b) for b in q_bets]
    all_bets.sort(key=lambda x: -x[6])
    if not all_bets:
        print("  No value vs Pinnacle. (Sharp book = hard to beat; this is expected/healthy.)")
    else:
        print(f"  {'market':<9} {'grp':>3} {'team':<5} {'odds':>6} {'model':>7} {'pin':>7} {'edge':>7} {'stake':>8}")
        for mkt, letter, t, o, mp, pp, edge, stake in all_bets:
            print(f"  {mkt:<9} {letter:>3} {t:<5} {o:6.2f} {mp*100:6.1f}% {pp*100:6.1f}% "
                  f"{edge*100:+6.1f}% ${stake:7.2f}")
    print(f"\n  ⚠️ vs Pinnacle (sharpest book): edges >8% likely MODEL bias not value.")
    print(f"     Real edge = find SOFTER books deviating from Pinnacle. Use Pinnacle as truth.")


if __name__ == "__main__":
    main()
