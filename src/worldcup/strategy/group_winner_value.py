"""Group-winner / advancement value detector.

The Odds API doesn't aggregate WC group-winner sub-markets and Polymarket
doesn't list them. But bookmaker websites (Bet365, Pinnacle, William Hill)
DO offer "Group Winner" and "To Qualify" markets pre-tournament — and these
sub-markets are MUCH less efficiently priced than the champion outright
(less sharp money, more algorithmic auto-pricing).

This tool:
    1. Runs MC to get each team's P(win group) and P(advance) — model side.
    2. Accepts bookmaker odds you paste in (from their website).
    3. Computes edge + Kelly-1/4 stake per selection.

Why this is one of our best edge sources: group-winner markets are exactly
where our full-tournament MC has signal AND the market is soft.

Usage (compute model probs only):
    PYTHONPATH=src python -m worldcup.strategy.group_winner_value --n-sims 50000

Usage (with book odds for a group):
    PYTHONPATH=src python -m worldcup.strategy.group_winner_value \
        --group D --market winner \
        --odds "USA:3.5,AUS:3.0,PAR:3.25,TUR:4.5"

    PYTHONPATH=src python -m worldcup.strategy.group_winner_value \
        --group D --market advance \
        --odds "USA:1.7,AUS:1.5,PAR:1.55,TUR:2.1" --bankroll 1000
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.models.dixon_coles import fit
from worldcup.simulator.monte_carlo import (
    MatchSampler,
    _simulate_group,
)
from worldcup.strategy.value_bets import devig_shin, kelly_fraction

ROOT = Path(__file__).resolve().parents[3]


def compute_group_probs(n_sims: int = 50000, prior: float = 0.5, seed: int = 42) -> dict:
    """Returns {group: {team: {win: p, advance: p}}}."""
    params = fit(elo_prior_strength=prior)
    cfg = yaml.safe_load((ROOT / "configs" / "teams.yaml").read_text())
    groups = {l: g["teams"] for l, g in cfg["groups"].items()}

    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    elo = {r[0]: float(r[1]) for r in conn.execute(
        "SELECT team_code, value FROM team_ratings WHERE rating_type='elo'")}
    conn.close()

    sampler = MatchSampler(rho=params.rho, home_adv=params.home_advantage,
                           attack=params.attack, defense=params.defense)
    rng = np.random.default_rng(seed)

    win_counts = {l: defaultdict(int) for l in groups}
    advance_counts = defaultdict(int)

    for _ in range(n_sims):
        winners, runners, thirds = [], [], []
        for letter in sorted(groups):
            ranked = _simulate_group(groups[letter], elo, sampler, rng)
            win_counts[letter][ranked[0].team] += 1
            winners.append(ranked[0].team)
            runners.append(ranked[1].team)
            thirds.append(ranked[2])
        best_thirds = [r.team for r in sorted(
            thirds, key=lambda r: (-r.points, -r.gd, -r.gf, -elo.get(r.team, 1500)))[:8]]
        for t in set(winners + runners + best_thirds):
            advance_counts[t] += 1

    result = {}
    for letter, teams in groups.items():
        result[letter] = {}
        for t in teams:
            result[letter][t] = {
                "win": win_counts[letter][t] / n_sims,
                "advance": advance_counts[t] / n_sims,
            }
    return result


def parse_odds(odds_str: str) -> dict[str, float]:
    """'USA:3.5,AUS:3.0' -> {'USA': 3.5, 'AUS': 3.0}"""
    out = {}
    for part in odds_str.split(","):
        code, price = part.split(":")
        out[code.strip().upper()] = float(price)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=50000)
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--group", help="Group letter to analyze with book odds")
    ap.add_argument("--market", choices=["winner", "advance"], default="winner")
    ap.add_argument("--odds", help="Book odds, e.g. 'USA:3.5,AUS:3.0,PAR:3.25,TUR:4.5'")
    ap.add_argument("--bankroll", type=float, default=1000)
    ap.add_argument("--min-edge", type=float, default=0.03)
    args = ap.parse_args()

    print(f"Computing group probabilities ({args.n_sims:,} MC sims) ...")
    probs = compute_group_probs(n_sims=args.n_sims, prior=args.prior)

    if not args.odds:
        # Just print full table — paste into your book comparison
        print("\n=== Model group probabilities (compare vs bookmaker website) ===\n")
        print(f"{'grp':>3}  {'team':<5}  {'P(win group)':>13}  {'P(advance)':>11}  "
              f"{'fair win odds':>13}  {'fair adv odds':>13}")
        print("-" * 75)
        for letter in sorted(probs):
            team_sorted = sorted(probs[letter].items(), key=lambda x: -x[1]["win"])
            for t, p in team_sorted:
                fw = 1/p["win"] if p["win"] > 0 else 999
                fa = 1/p["advance"] if p["advance"] > 0 else 999
                print(f"{letter:>3}  {t:<5}  {p['win']*100:12.1f}%  {p['advance']*100:10.1f}%  "
                      f"{fw:13.2f}  {fa:13.2f}")
            print()
        print("To find value: open your bookmaker's 'Group Winner' / 'To Qualify' market.")
        print("If book odds > our fair odds (i.e. book implies LOWER prob than us), it's value.")
        print(f"\nThen run with --group X --market winner --odds 'TEAM:price,...' for edge calc.")
        return

    # Value calc for a specific group + book odds
    book = parse_odds(args.odds)
    g = args.group.upper()
    if g not in probs:
        print(f"Group {g} not found.")
        return

    print(f"\n=== Group {g} — {args.market.upper()} market value check ===\n")
    # De-vig the book quotes (for winner: sum of all 4; for advance: sum/2 since 2 advance)
    teams = list(book.keys())
    prices = [book[t] for t in teams]
    # Winner market: 4 mutually exclusive → de-vig sums to 1
    # Advance market: 2 of 4 advance → "yes" probs sum to ~2, de-vig differently
    if args.market == "winner":
        implied = devig_shin(prices)
        implied_map = dict(zip(teams, implied))
    else:
        # advance: each is independent yes/no; just invert with vig estimate
        raw = {t: 1/book[t] for t in teams}
        total = sum(raw.values())
        # Should sum to ~2 (two teams advance); normalize to 2
        implied_map = {t: raw[t] / total * 2.0 for t in teams}

    print(f"{'team':<5}  {'book odds':>9}  {'book impl':>10}  {'model':>8}  {'edge':>7}  "
          f"{'EV%':>7}  {'Kelly1/4 $':>11}")
    print("-" * 70)
    rows = []
    for t in teams:
        model_p = probs[g][t]["win" if args.market == "winner" else "advance"]
        book_p = implied_map[t]
        edge = model_p - book_p
        price = book[t]
        ev = (model_p * price - 1) * 100
        kelly = kelly_fraction(model_p, price, scaling=0.25)
        stake = args.bankroll * kelly
        rows.append((t, price, book_p, model_p, edge, ev, stake))
    rows.sort(key=lambda r: -r[4])
    for t, price, bp, mp, edge, ev, stake in rows:
        marker = "  ✅ VALUE" if edge >= args.min_edge else ""
        print(f"{t:<5}  {price:9.2f}  {bp*100:9.1f}%  {mp*100:7.1f}%  "
              f"{edge*100:+6.1f}%  {ev:+6.1f}%  ${stake:10.2f}{marker}")
    print(f"\n  (model = our {args.n_sims//1000}k-sim MC; book impl = de-vigged from your pasted odds)")
    print(f"  ⚠️ Remember: group markets are softer than champion outright — model has real")
    print(f"     edge here, BUT still cross-check against any sharp source if available.")


if __name__ == "__main__":
    main()
