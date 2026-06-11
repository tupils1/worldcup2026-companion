"""Correlated parlay analyzer.

Books typically price multi-leg parlays as: P(A and B) ≈ P(A) × P(B).
Reality: if A and B are correlated (e.g. "Home wins" and "Over 2.5 goals"),
the true joint probability P(A∩B) ≠ P(A) × P(B).

Positive correlation (P(A∩B) > P(A)×P(B)):
    - Books underprice the parlay → BACK is value
Negative correlation (P(A∩B) < P(A)×P(B)):
    - Books overprice the parlay → FADE (or just don't bet)

We compute true joint probabilities from the Dixon-Coles score matrix, which
naturally captures all correlations between 1X2 / O/U / BTTS / clean sheet
outcomes (they all derive from the same scoreline distribution).

Edge magnitudes seen in real markets:
    - "Home win + over 2.5" parlay: 5-20% positive edge typical
    - "Home win + clean sheet" parlay: 15-40% positive edge
    - "Over 2.5 + BTTS": 5-15% positive (almost always)

Run:
    PYTHONPATH=src python -m worldcup.strategy.correlated_parlays --home FRA --away ENG
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.models.dixon_coles import fit
from worldcup.models.markets import fair_decimal_odds, score_matrix


# ─── Outcome conditions (functions of home_goals, away_goals) ───────────────

def home_win(h, a):  return h > a
def draw(h, a):      return h == a
def away_win(h, a):  return a > h
def over(line):      return lambda h, a: h + a > line
def under(line):     return lambda h, a: h + a < line
def btts(h, a):      return h >= 1 and a >= 1
def no_btts(h, a):   return h == 0 or a == 0
def home_score(h, a):  return h >= 1
def away_score(h, a):  return a >= 1
def home_clean(h, a):  return a == 0
def away_clean(h, a):  return h == 0
def home_2plus(h, a):  return h >= 2
def away_2plus(h, a):  return a >= 2


# ─── Core math ──────────────────────────────────────────────────────────────

def marginal(M: np.ndarray, condition) -> float:
    """P(condition) summed over score matrix."""
    max_g = M.shape[0]
    return float(sum(M[h, a] for h in range(max_g) for a in range(max_g)
                     if condition(h, a)))


def joint(M: np.ndarray, cond_a, cond_b) -> float:
    """P(A ∩ B) over the score matrix."""
    max_g = M.shape[0]
    return float(sum(M[h, a] for h in range(max_g) for a in range(max_g)
                     if cond_a(h, a) and cond_b(h, a)))


@dataclass
class ParlayAnalysis:
    label_a: str
    label_b: str
    p_a: float
    p_b: float
    p_joint: float        # true probability from score matrix
    p_product: float      # what books typically price (= p_a × p_b)
    correlation_ratio: float  # p_joint / p_product
    edge_pp: float        # (p_joint − p_product) × 100
    fair_odds: float


def analyze(M, cond_a, label_a, cond_b, label_b) -> ParlayAnalysis:
    pa = marginal(M, cond_a)
    pb = marginal(M, cond_b)
    pj = joint(M, cond_a, cond_b)
    prod = pa * pb
    return ParlayAnalysis(
        label_a=label_a, label_b=label_b,
        p_a=pa, p_b=pb, p_joint=pj, p_product=prod,
        correlation_ratio=pj / prod if prod > 0 else float("nan"),
        edge_pp=(pj - prod) * 100,
        fair_odds=fair_decimal_odds(pj),
    )


# ─── Standard parlay menu ──────────────────────────────────────────────────

STANDARD_PARLAYS = [
    # (cond_a, label_a, cond_b, label_b, note)
    (home_win, "Home win", over(2.5), "Over 2.5", "Win games tend to be high-scoring"),
    (home_win, "Home win", under(2.5), "Under 2.5", "Tight 1-0 / 2-0 win"),
    (home_win, "Home win", btts,       "BTTS",      "Win but conceded"),
    (home_win, "Home win", no_btts,    "No BTTS",   "Clean sheet win"),
    (home_win, "Home win", home_clean, "Clean sheet (home)", "Strongest positive corr"),
    (home_win, "Home win", home_2plus, "Home 2+ goals", "Convincing win"),
    (away_win, "Away win", over(2.5),  "Over 2.5", ""),
    (away_win, "Away win", under(2.5), "Under 2.5", ""),
    (away_win, "Away win", away_clean, "Clean sheet (away)", ""),
    (draw,     "Draw",     under(2.5), "Under 2.5", "0-0 / 1-1 draws"),
    (draw,     "Draw",     btts,       "BTTS",      "1-1 / 2-2 draws"),
    (over(2.5),"Over 2.5", btts,       "BTTS",      "Both score in goal-fest"),
    (over(3.5),"Over 3.5", btts,       "BTTS",      "Heavy scoring both sides"),
    (under(2.5),"Under 2.5", no_btts,  "No BTTS",   "Defensive game"),
]


def report(home: str, away: str, M: np.ndarray, min_edge_pp: float = 1.0) -> None:
    print(f"\n=== {home} vs {away}: Correlated parlay analysis ===\n")
    print(f"  Score matrix shape {M.shape}, total P = {M.sum():.4f}\n")

    rows = []
    for cond_a, lab_a, cond_b, lab_b, note in STANDARD_PARLAYS:
        # Suffix labels with team names for 1X2 conditions
        la = lab_a.replace("Home", home).replace("Away", away)
        lb = lab_b.replace("Home", home).replace("Away", away)
        a = analyze(M, cond_a, la, cond_b, lb)
        rows.append((a, note))

    # Sort by absolute edge descending
    rows.sort(key=lambda r: -abs(r[0].edge_pp))

    print(f"{'Parlay':<48}  {'P(true)':>8} {'P(prod)':>8} {'ratio':>7} {'edge':>7}  {'fair@':>6}")
    print("-" * 100)
    for a, note in rows:
        edge_marker = "  ✅" if a.edge_pp >= min_edge_pp else ("  ❌" if a.edge_pp <= -min_edge_pp else "    ")
        label = f"{a.label_a} + {a.label_b}"
        print(
            f"{label[:48]:<48}  "
            f"{a.p_joint*100:7.2f}% {a.p_product*100:7.2f}% "
            f"{a.correlation_ratio:6.3f}x {a.edge_pp:+6.1f}pp  "
            f"{a.fair_odds:5.2f}{edge_marker}"
        )
    print()
    print("Interpretation:")
    print("  ratio > 1 = books underprice (parlay is + EV vs simple product)")
    print("  ratio < 1 = books overprice")
    print("  Real-world parlays SHOULD match true probability; if book quotes")
    print("  ~product price, the edge is the ratio. Most book parlays use product")
    print("  pricing in the absence of specialized prop trader oversight.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", required=True, help="3-letter code")
    ap.add_argument("--away", required=True)
    ap.add_argument("--neutral", action="store_true", help="Neutral venue")
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--min-edge", type=float, default=1.0,
                    help="Min |edge| (pp) to flag with ✅/❌")
    args = ap.parse_args()

    print(f"Fitting Dixon-Coles ...")
    params = fit(elo_prior_strength=args.prior)
    lh, la = params.predict_lambda(args.home, args.away, neutral=args.neutral)
    M = score_matrix(lh, la, rho=params.rho)
    print(f"  λ_{args.home}={lh:.2f}  λ_{args.away}={la:.2f}  ρ={params.rho:.3f}")

    report(args.home, args.away, M, min_edge_pp=args.min_edge)


if __name__ == "__main__":
    main()
