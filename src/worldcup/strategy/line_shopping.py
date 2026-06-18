"""Cross-book best-line + arbitrage scanner (50+ books via The Odds API).

Pure market alpha — does NOT depend on our model (which can't beat sharp books).
Three things this finds:

1. **Arbitrage** (risk-free): when the BEST odds across books for all outcomes
   of a market imply < 100% total probability, you can back every outcome at
   different books and lock in guaranteed profit.

2. **Best-line +EV**: when the best available odds for a selection imply a LOWER
   probability than the sharp consensus (de-vigged across all books), that book
   is mispricing — positive expected value to back it there.

3. **Shopping guide**: once you decide to bet a selection, which book gives the
   highest odds (max payout, min effective vig).

This is the single biggest auto-available alpha pool: 72 matches × 50 books ×
{1X2, AH, OU} markets. Even if you have no model edge, always betting the best
line across books saves the vig and occasionally captures soft-book errors.

Run:
    PYTHONPATH=src python -m worldcup.strategy.line_shopping
    PYTHONPATH=src python -m worldcup.strategy.line_shopping --min-ev 0.02 --arb-only
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.strategy.value_bets import devig_shin


def load_latest_match_odds(conn) -> list[dict]:
    """Latest odds per (match, book, market, selection, line)."""
    rows = conn.execute("""
        SELECT o.match_id, o.bookmaker, o.market, o.selection, o.line, o.price
        FROM odds o
        JOIN (
            SELECT match_id, bookmaker, market, selection, COALESCE(line, -999.0) AS lk,
                   MAX(captured_at) AS mc
            FROM odds WHERE market_scope='match'
            GROUP BY match_id, bookmaker, market, selection, COALESCE(line, -999.0)
        ) latest
          ON o.match_id=latest.match_id AND o.bookmaker=latest.bookmaker
             AND o.market=latest.market AND o.selection=latest.selection
             AND COALESCE(o.line,-999.0)=latest.lk AND o.captured_at=latest.mc
        WHERE o.market_scope='match'
          AND o.match_id IN (
              SELECT id FROM matches
              WHERE finished=0 AND match_date >= date('now','-1 day'))
    """)
    return [dict(r) for r in rows]


def match_labels(conn) -> dict[int, str]:
    out = {}
    for r in conn.execute("SELECT id, home_code, away_code, match_date FROM matches"):
        out[r["id"]] = f"{r['home_code']}-{r['away_code']} {r['match_date']}"
    return out


def analyze(odds_rows, labels, min_ev=0.02, max_odds=12.0, min_books=6,
            accessible_books=None):
    """Returns (arbitrages, best_line_value, shopping_guide).

    max_odds: ignore longshot selections above this (stale/error-prone, low limits).
    min_books: require this many books quoting for robust consensus.

    accessible_books: set of bookmaker keys you can ACTUALLY fund + bet from your
    jurisdiction. When given, the EXECUTABLE prices (arbitrage legs, best-line back
    price, shopping best) are taken ONLY from these books — you can't bet anywhere
    else. The CONSENSUS / fair value is STILL de-vigged across ALL books (read the
    whole market for the sharpest truth, bet only where you can). When None, all
    books are treated as bettable (RESEARCH mode — may surface arb you can't take).
    """
    acc = set(accessible_books) if accessible_books else None

    def best_bettable(plist):
        """Highest price among books you can bet at (all books if acc is None)."""
        cand = [(p, b) for p, b in plist if acc is None or b in acc]
        return max(cand, key=lambda x: x[0]) if cand else None

    # Group by (match, market, line) → selection → list of (price, book)
    groups = defaultdict(lambda: defaultdict(list))
    for r in odds_rows:
        key = (r["match_id"], r["market"], r["line"])
        groups[key][r["selection"]].append((r["price"], r["bookmaker"]))

    arbitrages = []
    best_line_value = []
    shopping = []

    for (match_id, market, line), sels in groups.items():
        label = labels.get(match_id, f"match#{match_id}")
        # Best BETTABLE odds per selection (accessible books only, if restricted)
        best = {s: bb for s, plist in sels.items() if (bb := best_bettable(plist))}

        # Define the complete outcome set per market
        if market == "1X2":
            needed = ("home", "draw", "away")
        elif market == "OU":
            needed = ("over", "under")
        elif market == "AH":
            needed = ("home", "away")
        else:
            continue
        if not all(s in best for s in needed):
            continue

        # ── Arbitrage check ──
        # Only 1X2 (home/draw/away) and OU (over/under) are truly complementary
        # within one (match, market, line) group. AH home@-L and away@+L live in
        # DIFFERENT line groups, so same-group AH "arb" is spurious — skip it.
        if market == "AH":
            # Still do best-line value + shopping below, but no arbitrage.
            inv_sum = 99.0
        else:
            inv_sum = sum(1.0 / best[s][0] for s in needed)
        if inv_sum < 1.0:
            profit_pct = (1.0 / inv_sum - 1.0) * 100
            arbitrages.append({
                "match": label, "market": market, "line": line,
                "profit_pct": profit_pct,
                "legs": {s: best[s] for s in needed},
                "inv_sum": inv_sum,
            })

        # ── Consensus de-vig (across ALL books quoting this market) ──
        # Build per-book complete quotes to de-vig properly
        book_quotes = defaultdict(dict)
        for s in needed:
            for price, book in sels[s]:
                book_quotes[book][s] = price
        devigged_per_book = []
        for book, q in book_quotes.items():
            if all(s in q for s in needed):
                dv = devig_shin([q[s] for s in needed])
                devigged_per_book.append(dict(zip(needed, dv)))
        if not devigged_per_book:
            continue
        consensus = {s: float(np.mean([d[s] for d in devigged_per_book])) for s in needed}

        # ── Best-line +EV: consensus from ALL books; back at your best BETTABLE price ──
        if len(devigged_per_book) >= min_books:
            for s in needed:
                cons = consensus[s]
                # Skip longshots (stale/error-prone, low limits)
                if cons < 1.0 / max_odds:
                    continue
                bettable = sorted([(p, b) for p, b in sels[s]
                                   if acc is None or b in acc], key=lambda x: -x[0])
                if not bettable:
                    continue  # can't bet this selection anywhere you have access
                abs_best = sorted(sels[s], key=lambda x: -x[0])[0]  # all-book best (reference)
                if acc is None:
                    # RESEARCH mode: 2nd-best across all books (robust vs single-book error)
                    if len(bettable) < 2:
                        continue
                    exec_price, exec_book = bettable[1]
                else:
                    # EXECUTABLE mode: best price among the books you can actually bet
                    exec_price, exec_book = bettable[0]
                if exec_price > max_odds:
                    continue
                ev = cons * exec_price - 1.0
                if ev >= min_ev:
                    best_line_value.append({
                        "match": label, "market": market, "line": line,
                        "selection": s,
                        "best_price": exec_price, "best_book": exec_book,
                        "abs_best": abs_best[0], "abs_best_book": abs_best[1],
                        "best_implied": 1.0 / exec_price, "consensus": cons,
                        "ev_pct": ev * 100,
                        "n_books": len(devigged_per_book),
                        "n_bettable": len(bettable),
                    })

        # ── Shopping guide: best odds + spread across the books you can bet ──
        # Exclude longshots + obvious data errors (use median as sanity floor).
        min_quotes = 2 if acc is not None else min_books
        for s in needed:
            prices = sorted(p for p, b in sels[s] if acc is None or b in acc)
            if len(prices) < min_quotes:
                continue
            med = prices[len(prices)//2]
            if med > max_odds:          # longshot selection
                continue
            # Worst = lowest price that's still within 50% of median (drop mislabeled rows)
            sane = [p for p in prices if p >= med * 0.5]
            worst = min(sane) if sane else prices[0]
            shopping.append({
                "match": label, "market": market, "line": line, "selection": s,
                "best_price": best[s][0], "best_book": best[s][1],
                "worst_price": worst, "spread_pct": (best[s][0]/worst - 1)*100,
                "n_books": len(prices),
            })

    arbitrages.sort(key=lambda x: -x["profit_pct"])
    best_line_value.sort(key=lambda x: -x["ev_pct"])
    shopping.sort(key=lambda x: -x["spread_pct"])
    return arbitrages, best_line_value, shopping


ACCESSIBLE_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "accessible_books.yaml"


def load_accessible_books(cli_books: str | None = None) -> list[str] | None:
    """Resolve the bettable-book subset: --books CLI wins, else configs/accessible_books.yaml,
    else None (RESEARCH mode = all books). Returns a list of bookmaker keys or None."""
    if cli_books:
        return [b.strip() for b in cli_books.split(",") if b.strip()]
    if ACCESSIBLE_CONFIG.exists():
        try:
            import yaml
            data = yaml.safe_load(ACCESSIBLE_CONFIG.read_text()) or {}
            books = data.get("accessible_books") or []
            return [str(b).strip() for b in books if str(b).strip()] or None
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-ev", type=float, default=0.02, help="Min EV for best-line value")
    ap.add_argument("--arb-only", action="store_true")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--books", type=str, default=None,
                    help="Comma-separated bettable bookmaker keys (overrides config). "
                         "Omit → configs/accessible_books.yaml → else all books (research).")
    args = ap.parse_args()

    accessible = load_accessible_books(args.books)

    conn = get_conn(DEFAULT_DB_PATH)
    odds = load_latest_match_odds(conn)
    labels = match_labels(conn)
    conn.close()

    n_books = len({r["bookmaker"] for r in odds})
    n_matches = len({r["match_id"] for r in odds})
    print(f"Loaded {len(odds)} odds rows: {n_matches} matches × {n_books} books")
    if accessible:
        present = sorted({r["bookmaker"] for r in odds} & set(accessible))
        missing = sorted(set(accessible) - {r["bookmaker"] for r in odds})
        print(f"EXECUTABLE mode — betting only at: {', '.join(present) or '(none of your books quote!)'}")
        if missing:
            print(f"  (configured but absent from feed: {', '.join(missing)})")
        print(f"  Consensus/fair value still read from all {n_books} books; arb+best-line use your subset.")
    else:
        print("RESEARCH mode — all books treated as bettable (set configs/accessible_books.yaml "
              "to restrict to books you can actually fund).")
    print()

    arbs, blv, shop = analyze(odds, labels, min_ev=args.min_ev, accessible_books=accessible)

    scope = "your bettable books" if accessible else "all books"
    # ── 1. Arbitrage ──
    print("=" * 90)
    print(f"①  ARBITRAGE (risk-free: best odds across {scope} imply < 100%)")
    print("=" * 90)
    if not arbs:
        none_note = ("None among your books. (Restricting to a few books makes arb rare —"
                     " expected; add more accessible books to find more.)") if accessible \
            else "None found. (Arbitrage is rare on liquid markets — expected.)"
        print(f"  {none_note}")
    else:
        for a in arbs[:args.top]:
            print(f"\n  🎯 {a['match']}  {a['market']} {a['line'] or ''}  "
                  f"PROFIT {a['profit_pct']:.2f}%  (inv-sum {a['inv_sum']:.4f})")
            for sel, (price, book) in a["legs"].items():
                stake_share = (1.0/price) / a["inv_sum"] * 100
                print(f"      {sel:<6} @ {price:.2f} ({book})  stake {stake_share:.1f}%")

    if args.arb_only:
        return

    # ── 2. Best-line +EV ──
    print("\n" + "=" * 90)
    print(f"②  BEST-LINE +EV (back at {scope} vs all-book sharp consensus, EV ≥ {args.min_ev:.0%})")
    print("=" * 90)
    if not blv:
        print("  None above threshold (after longshot + single-book-outlier filtering).")
    else:
        price_note = ("'@' = best price at YOUR books; 'abs' = best across ALL books "
                      "(reference — you can't bet there)") if accessible else \
            "Using 2nd-best price (robust). 'abs' = absolute best (verify it's not stale)."
        print(f"  {price_note}")
        print(f"  {'match':<22} {'mkt':<4} {'line':>5} {'sel':<6} {'@':>6} {'book':<13} "
              f"{'cons':>6} {'EV':>6}  {'abs':>6}")
        for b in blv[:args.top]:
            ls = f"{b['line']:+.1f}" if b['line'] is not None else ""
            print(f"  {b['match']:<22} {b['market']:<4} {ls:>5} {b['selection']:<6} "
                  f"{b['best_price']:6.2f} {b['best_book'][:13]:<13} "
                  f"{b['consensus']*100:5.1f}% {b['ev_pct']:+5.1f}%  {b['abs_best']:6.2f}")

    # ── 3. Shopping guide (biggest cross-book spreads) ──
    print("\n" + "=" * 90)
    print("③  SHOPPING GUIDE (biggest cross-book price spreads — always bet the best)")
    print("=" * 90)
    print(f"  {'match':<22} {'mkt':<4} {'sel':<6} {'best':>6} {'@book':<14} {'worst':>6} {'spread':>7} {'#bk':>4}")
    for s in shop[:args.top]:
        print(f"  {s['match']:<22} {s['market']:<4} {s['selection']:<6} "
              f"{s['best_price']:6.2f} {s['best_book'][:14]:<14} {s['worst_price']:6.2f} "
              f"{s['spread_pct']:6.1f}% {s['n_books']:>4}")

    print(f"\n  Summary: {len(arbs)} arbs, {len(blv)} best-line +EV bets, "
          f"avg cross-book spread {np.mean([s['spread_pct'] for s in shop]):.1f}%")
    print(f"  → Even with zero model edge, betting the best line saves the gap between")
    print(f"    best and worst book (avg shown above) on every bet.")


if __name__ == "__main__":
    main()
