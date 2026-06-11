"""Polymarket edge scanner — read full sharp consensus → rank divergences →
deduct cost → Kelly-size → actionable bet list.

The user can bet Polymarket (no-vig, two-sided YES/NO, deep WC pool) but NOT the
international books/exchange. So we READ the sharp venues for free as the TRUTH and
bet the divergence ON Polymarket where it's softer (retail-led / laggy).

TRUTH benchmark (priority order):
  1. Betfair EXCHANGE (betfair_ex_uk/eu/au) — peer-to-peer, ~0 margin = the sharpest
     champion-outright signal. De-vig each region (Shin), average. PRIMARY.
  2. Full-book de-vigged consensus (incl. softer books) — robustness cross-check.
  Edge is only HIGH-confidence when Betfair and the book consensus AGREE on direction
  vs Polymarket (else it's noise / a stale book).

THE BET (champion outright = "win WC 2026"; same market on Polymarket & Betfair):
  - Polymarket UNDERprices team (poly% < truth%): BUY YES  @ poly price.
  - Polymarket OVERprices  team (poly% > truth%): BUY NO   @ (1 − poly price).
  EV(side) = true_prob(side) / price(side) − 1.  Kelly on the chosen side.

COST: a haircut on the edge for Polymarket entry (half-spread + gas + buffer). Held
to resolution (WC final, ~July) there's no exit cost; entry half-spread on liquid WC
markets ≈ 0.5pp. Net edge must clear `--cost-pp` + `--min-edge`.

SIZING: fractional Kelly (default ¼), capped at `--max-stake` of bankroll AND at
`--liq-cap` of the market's Polymarket liquidity (so you can actually get filled).

Run:
    PYTHONPATH=src python -m worldcup.strategy.polymarket_lag
    PYTHONPATH=src python -m worldcup.strategy.polymarket_lag --bankroll 1000 --min-edge 0.02
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.strategy.value_bets import devig_shin, kelly_fraction

BETFAIR_BOOKS = ("betfair_ex_uk", "betfair_ex_eu", "betfair_ex_au")


def _latest_champion_odds(conn) -> dict[str, dict[str, float]]:
    """{bookmaker: {team: decimal_odds}} — latest champion-outright price per book/team."""
    by_book: dict[str, dict[str, float]] = defaultdict(dict)
    for bk, team, price in conn.execute("""
        SELECT o.bookmaker, o.subject_code, o.price
        FROM odds o JOIN (
            SELECT bookmaker, subject_code, MAX(captured_at) mc
            FROM odds WHERE market_scope='outright' AND market='winner'
            GROUP BY bookmaker, subject_code
        ) l ON o.bookmaker=l.bookmaker AND o.subject_code=l.subject_code AND o.captured_at=l.mc
        WHERE o.market_scope='outright' AND o.market='winner'
    """):
        if price and price > 1.0:
            by_book[bk][team] = price
    return by_book


def _devig_consensus(book_quotes: dict[str, dict[str, float]], books) -> dict[str, float]:
    """Average de-vigged (Shin) prob per team across the given books."""
    per_team: dict[str, list[float]] = defaultdict(list)
    for bk in books:
        q = book_quotes.get(bk)
        if not q or len(q) < 10:
            continue
        teams = list(q.keys())
        for t, p in zip(teams, devig_shin([q[t] for t in teams])):
            per_team[t].append(p)
    return {t: float(np.mean(v)) for t, v in per_team.items() if v}


def fetch_polymarket(conn) -> dict[str, dict]:
    """Latest Polymarket champion quote per team (price = YES price you pay)."""
    out = {}
    for team, price, liq, cap in conn.execute("""
        SELECT o.subject_code, o.price, o.line, o.captured_at
        FROM odds o JOIN (
            SELECT subject_code, MAX(captured_at) mc FROM odds
            WHERE bookmaker='polymarket' AND market_scope='outright' AND market='winner'
            GROUP BY subject_code
        ) l ON o.subject_code=l.subject_code AND o.captured_at=l.mc
        WHERE o.bookmaker='polymarket' AND o.market_scope='outright' AND o.market='winner'
    """):
        if price and price > 1.0:
            out[team] = {"yes_price": 1.0 / price, "liquidity": liq or 0.0, "captured_at": cap}
    return out


def scan_edges(db_path: Path | str = DEFAULT_DB_PATH, *, bankroll: float = 1000.0,
               min_edge: float = 0.015, cost_pp: float = 0.5, kelly_scaling: float = 0.25,
               max_stake_frac: float = 0.03, liq_cap_frac: float = 0.02,
               min_liquidity: float = 5000.0) -> dict:
    conn = sqlite3.connect(str(db_path))
    book_quotes = _latest_champion_odds(conn)
    betfair = _devig_consensus(book_quotes, BETFAIR_BOOKS)
    allbooks = _devig_consensus(book_quotes, [b for b in book_quotes if b != "polymarket"])
    poly = fetch_polymarket(conn)
    conn.close()

    cost = cost_pp / 100.0
    bets, skipped_illiquid = [], 0
    for team in sorted(set(poly) & set(betfair)):
        q = poly[team]["yes_price"]            # price you pay for YES (= poly implied)
        truth = betfair[team]                  # sharpest fair prob (Betfair exchange de-vig)
        book = allbooks.get(team)
        liq = poly[team]["liquidity"]
        raw_edge = abs(q - truth)
        net_edge = raw_edge - cost
        if net_edge < min_edge:
            continue
        if liq < min_liquidity:
            skipped_illiquid += 1
            continue

        if q < truth:   # Polymarket underprices → BUY YES
            side, price, p_side = "BUY YES", q, truth
        else:           # Polymarket overprices → BUY NO
            side, price, p_side = "BUY NO", 1.0 - q, 1.0 - truth
        dec = 1.0 / price
        ev = p_side * dec - 1.0
        # confidence: does the full-book consensus agree with Betfair's direction?
        agree = book is not None and ((book > q) == (truth > q))
        kf = kelly_fraction(p_side, dec, scaling=kelly_scaling)
        stake = min(kf * bankroll, max_stake_frac * bankroll, liq_cap_frac * liq)
        bets.append({
            "team": team, "side": side, "price": price, "dec_odds": dec,
            "poly_pct": q * 100, "betfair_pct": truth * 100,
            "book_pct": (book * 100 if book is not None else None),
            "raw_edge_pp": raw_edge * 100, "net_edge_pp": net_edge * 100,
            "ev_pct": ev * 100, "kelly_pct": kf * 100, "stake": stake,
            "liquidity": liq, "agree": agree,
            "captured_at": poly[team]["captured_at"],
        })
    bets.sort(key=lambda x: -x["ev_pct"])
    return {"bets": bets, "skipped_illiquid": skipped_illiquid,
            "n_betfair": len(betfair), "n_poly": len(poly),
            "captured": poly[next(iter(poly))]["captured_at"] if poly else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-edge", type=float, default=0.015,
                    help="Min NET edge (after cost) in decimal, default 1.5pp")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Alias for --min-edge (backward compat with daily_refresh)")
    ap.add_argument("--cost-pp", type=float, default=0.5, help="Polymarket entry-cost haircut (pp)")
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly scaling (default 1/4)")
    ap.add_argument("--max-stake", type=float, default=0.03, help="Max stake as frac of bankroll")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    min_edge = args.threshold if args.threshold is not None else args.min_edge

    r = scan_edges(bankroll=args.bankroll, min_edge=min_edge, cost_pp=args.cost_pp,
                   kelly_scaling=args.kelly, max_stake_frac=args.max_stake)
    bets = r["bets"]

    print("=" * 104)
    print(f"POLYMARKET EDGE SCANNER — sharp truth = Betfair exchange (de-vig); bet on Polymarket")
    print("=" * 104)
    print(f"Betfair teams {r['n_betfair']} · Polymarket teams {r['n_poly']} · "
          f"bankroll ${args.bankroll:,.0f} · ¼-Kelly · net edge ≥ {min_edge*100:.1f}pp "
          f"(after {args.cost_pp:.1f}pp cost) · Poly captured {r['captured']}")
    if not bets:
        print("\n  No Polymarket edge above threshold. (Polymarket aligned with sharp truth — expected;")
        print("  it's a $1.2B market. Re-run after new info / each refresh; act when retail flow drifts.)")
        if r["skipped_illiquid"]:
            print(f"  ({r['skipped_illiquid']} divergences skipped: Polymarket liquidity < min.)")
        return

    print(f"\n{'team':<5}{'action':<9}{'@price':>7}{'poly%':>7}{'sharp%':>7}{'book%':>7}"
          f"{'netEdge':>8}{'EV':>7}{'Kelly':>7}{'stake$':>8}{'liq$':>10}  conf")
    print("-" * 104)
    for b in bets[:args.top]:
        bookpct = f"{b['book_pct']:.1f}" if b["book_pct"] is not None else "—"
        conf = "✓agree" if b["agree"] else "⚠book-disagrees"
        print(f"{b['team']:<5}{b['side']:<9}{b['price']:>7.3f}{b['poly_pct']:>6.1f}%"
              f"{b['betfair_pct']:>6.1f}%{bookpct:>7}{b['net_edge_pp']:>+7.1f}%{b['ev_pct']:>+6.1f}%"
              f"{b['kelly_pct']:>6.1f}%{b['stake']:>8.0f}{b['liquidity']:>10,.0f}  {conf}")
    if r["skipped_illiquid"]:
        print(f"  ({r['skipped_illiquid']} edge(s) skipped: Polymarket liquidity too thin to fill.)")
    print("\nReading: BUY YES = Polymarket too cheap vs sharp; BUY NO = too expensive. Truth = Betfair")
    print("exchange (≈0 vig). ⚠book-disagrees = soft-book consensus contradicts Betfair → lower trust.")
    print("Stake = min(¼-Kelly, max-stake, 2%·liquidity). Held to WC final resolution (~July).")
    print("Re-check each refresh; the edge is retail drift/lag — act while the gap is open.")


if __name__ == "__main__":
    main()
