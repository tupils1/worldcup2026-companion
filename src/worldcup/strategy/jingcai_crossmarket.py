"""竞彩足球 single-match cross-market scanner (joint-engine value, parlay-ban-aware).

WHY NOT a correlated-parlay tool (like correlated_parlays.py for Pinnacle):
    竞彩 BANS combining different markets of the SAME match in a 串关/过关 — and bans it
    PRECISELY because the outcomes are score-correlated (official rule: once the score is
    out, 胜平负/总进球/半全场 are determined, so they "can't be a parlay"). So the
    Pinnacle-style "Home win + Over 2.5" correlation edge is structurally unavailable here.

WHAT the joint engine DOES give us in 竞彩 (and it's arguably better — single bets, no
margin compounding, and it hits the soft 比分 market):
    竞彩 prices 胜平负 / 让球胜平负 / 总进球 / 比分 as SEPARATE, independently-set,
    once-fixed markets. They are all marginals of the SAME score distribution, so they must
    be mutually consistent — but independent pricing makes them inconsistent. score_matrix
    is the unifying joint that exposes it:

    1. SINGLE-MARKET VALUE: derive fair prices for every market from one (market-anchored,
       bias-free) joint; flag the most mispriced 竞彩 market. The 比分 (correct-score)
       market is usually the softest — casual bettors overbet "pretty" scores (1:0, 2:1, 1:1).

    2. CROSS-MARKET CONSISTENCY (the real same-match edge, and it's ALLOWED): you can back
       the SAME outcome two ways — e.g. "home win" directly via 胜平负, or synthetically by
       backing every home-win 比分 cell (a dutch of single bets, not a parlay). If 竞彩's
       两 representations imply different prices, back the cheaper one. We compute the
       synthetic odds and compare to the direct line, scored against the fair joint.

    3. CROSS-MATCH parlay (the only 串关 竞彩 allows) is correctly product-priced (different
       matches ARE independent) → no correlation edge, only margin COMPOUNDING: an n-leg
       parlay multiplies per-leg margin to (1+m)^n. Quantified here as a tax, mostly −EV.

DATA: live 竞彩 odds are JS-rendered CN sites (not WebFetch-able), same wall as 选择比例.
This demos with synthetic 竞彩-style prices (realistic margins + a soft 比分 market baked in);
live use needs Chrome MCP / manual odds. Feed market-anchored p_true (bias-free) as the joint.

Run:
    PYTHONPATH=src python -m worldcup.strategy.jingcai_crossmarket --home FRA --away ENG
"""

from __future__ import annotations

import argparse

import numpy as np

from worldcup.models.dixon_coles import fit
from worldcup.models.markets import (
    fair_decimal_odds, goal_margin_dist, prob_1x2, score_matrix,
)

# 竞彩 比分 (correct-score) listed cells; everything else falls into 胜/平/负其他.
JC_HOME_SCORES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
                  (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)]
JC_DRAW_SCORES = [(0, 0), (1, 1), (2, 2), (3, 3)]
JC_AWAY_SCORES = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3),
                  (0, 4), (1, 4), (2, 4), (0, 5), (1, 5), (2, 5)]
TOTAL_BUCKETS = ["0", "1", "2", "3", "4", "5", "6", "7+"]


# ─────────────────────────── fair prices from the joint ──────────────────────
def fair_prices(M: np.ndarray, handicap: int = -1) -> dict:
    """Every 竞彩 single-match market's fair prob, all derived from one score matrix."""
    ph, pd, pa = prob_1x2(M)
    out = {"spf": {"home": ph, "draw": pd, "away": pa}}

    # 让球胜平负 (3-way on an integer goal handicap applied to home)
    margins = goal_margin_dist(M)
    rh = rd = ra = 0.0
    for k, p in margins.items():
        adj = k + handicap
        rh += p if adj > 0 else 0.0
        rd += p if adj == 0 else 0.0
        ra += p if adj < 0 else 0.0
    out["rspf"] = {"line": handicap, "home": rh, "draw": rd, "away": ra}

    # 总进球数 buckets 0..7+
    max_g = M.shape[0]
    tot = {b: 0.0 for b in TOTAL_BUCKETS}
    for h in range(max_g):
        for a in range(max_g):
            s = h + a
            tot["7+" if s >= 7 else str(s)] += float(M[h, a])
    out["totals"] = tot

    # 比分: listed cells + 胜/平/负其他 (residual within each 1X2 outcome)
    score = {}
    for (h, a) in JC_HOME_SCORES + JC_DRAW_SCORES + JC_AWAY_SCORES:
        score[f"{h}:{a}"] = float(M[h, a]) if h < max_g and a < max_g else 0.0
    score["胜其他"] = ph - sum(score[f"{h}:{a}"] for h, a in JC_HOME_SCORES)
    score["平其他"] = pd - sum(score[f"{h}:{a}"] for h, a in JC_DRAW_SCORES)
    score["负其他"] = pa - sum(score[f"{h}:{a}"] for h, a in JC_AWAY_SCORES)
    out["score"] = score
    return out


# ─────────────────────────── de-vig helper ───────────────────────────────────
def devig(odds: dict[str, float]) -> dict[str, float]:
    """Normalise 1/odds within one market to a proper distribution (removes overround)."""
    inv = {k: 1.0 / o for k, o in odds.items() if o and o > 1.0}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()} if s > 0 else {}


def overround(odds: dict[str, float]) -> float:
    return sum(1.0 / o for o in odds.values() if o and o > 1.0)


# ─────────────────────────── scans ───────────────────────────────────────────
def single_market_value(fair: dict, jc_odds: dict, min_ev=0.03, min_prob=0.04) -> list[dict]:
    """For each 竞彩 selection, EV of backing it = fair_prob × 竞彩_odds − 1.

    min_prob floors the selection's true probability: a +40% EV on a 1.6% correct-score
    is a trap — ruinous variance AND our model's tail estimate is unreliable there. Only
    selections with fair_prob ≥ min_prob are actionable (mirrors the 任九 winner-floor)."""
    found = []
    flat_fair = {**{f"spf:{k}": v for k, v in fair["spf"].items()},
                 **{f"rspf:{k}": v for k, v in fair["rspf"].items() if k != "line"},
                 **{f"totals:{k}": v for k, v in fair["totals"].items()},
                 **{f"score:{k}": v for k, v in fair["score"].items()}}
    skipped_tail = 0
    for mkt, odds in jc_odds.items():
        for sel, o in odds.items():
            key = f"{mkt}:{sel}"
            fp = flat_fair.get(key)
            if fp is None or not o or o <= 1.0:
                continue
            ev = fp * o - 1.0
            if ev >= min_ev:
                if fp < min_prob:
                    skipped_tail += 1
                    continue
                found.append({"market": mkt, "selection": sel, "jc_odds": o,
                              "fair_prob": fp, "fair_odds": fair_decimal_odds(fp), "ev": ev})
    found.sort(key=lambda x: -x["ev"])
    return found, skipped_tail


def synthetic_vs_direct(fair: dict, jc_odds: dict) -> list[dict]:
    """Back each 1X2 outcome two ways: direct 胜平负 vs dutch of 比分 cells. Report the
    better (higher) odds + its EV vs the fair joint. ALLOWED (separate single bets)."""
    score_odds = jc_odds.get("score", {})
    spf_odds = jc_odds.get("spf", {})
    groups = {"home": JC_HOME_SCORES + ["胜其他"], "draw": JC_DRAW_SCORES + ["平其他"],
              "away": JC_AWAY_SCORES + ["负其他"]}
    rows = []
    for outcome, cells in groups.items():
        keys = [f"{h}:{a}" if isinstance(c, tuple) else c
                for c in cells for (h, a) in [c if isinstance(c, tuple) else (None, None)]]
        inv = sum(1.0 / score_odds[k] for k in keys if k in score_odds and score_odds[k] > 1.0)
        synth_odds = 1.0 / inv if inv > 0 else float("inf")
        direct_odds = spf_odds.get(outcome, float("inf"))
        best, via = (synth_odds, "比分-synthetic") if synth_odds > direct_odds else (direct_odds, "胜平负-direct")
        fp = fair["spf"][outcome]
        rows.append({"outcome": outcome, "direct_odds": direct_odds, "synth_odds": synth_odds,
                     "best_odds": best, "best_via": via, "fair_prob": fp,
                     "ev_best": fp * best - 1.0,
                     "gap_pp": (1.0 / direct_odds - 1.0 / synth_odds) * 100
                               if direct_odds > 1 and synth_odds > 1 else float("nan")})
    return rows


def parlay_margin_tax(per_leg_overround: float, legs: int) -> list[dict]:
    """Cross-match parlay: margin compounds. Effective take after n legs."""
    return [{"legs": n, "eff_overround": per_leg_overround ** n,
             "eff_take_pct": (1.0 - 1.0 / per_leg_overround ** n) * 100}
            for n in range(1, legs + 1)]


# ─────────────────────────── demo synthetic 竞彩 prices ───────────────────────
def synth_jc_odds(fair: dict) -> dict:
    """Build illustrative 竞彩 odds = the CROWD's distorted view + a per-market margin.

    竞彩 odds reflect crowd/trader pricing, NOT our fair joint — that gap is the alpha.
    So: form a crowd distribution (casuals overweight 'pretty' scores & favourites,
    underweight the rest), renormalise it to a proper distribution, then price with
    margin: odds_k = 1/(crowd_k · margin). Value = fair_k·odds_k − 1 = fair_k/crowd_k/margin
    − 1, so a selection the crowd UNDER-bets by more than the margin is +EV. The 比分
    market also distorts the 1X2 it IMPLIES differently from the 胜平负 market → the
    cross-market inconsistency in scan ②."""
    POPULAR = {"1:0", "2:1", "1:1", "2:0", "0:0", "1:2", "0:1", "3:0", "2:2"}

    def crowd_then_price(probs, margin, pop=None, pop_mult=1.7, other_mult=0.72):
        if pop is None:  # non-score markets: mild favourite-longshot skew
            fav = max(probs, key=probs.get)
            d = {k: probs[k] * (1.30 if k == fav else 0.85) for k in probs}
        else:            # score market: crowd piles into 'pretty' scores
            d = {k: probs[k] * (pop_mult if k in pop else other_mult) for k in probs}
        s = sum(d.values()) or 1.0
        crowd = {k: v / s for k, v in d.items()}
        return {k: (max(1.01, 1.0 / (p * margin)) if p > 0 else 999.0)
                for k, p in crowd.items()}

    return {
        "spf": crowd_then_price(fair["spf"], margin=1.13),
        "rspf": crowd_then_price({k: v for k, v in fair["rspf"].items() if k != "line"},
                                 margin=1.13),
        "totals": crowd_then_price(fair["totals"], margin=1.18),
        "score": crowd_then_price(fair["score"], margin=1.25, pop=POPULAR),  # softest
    }


def report(home, away, M, min_ev=0.03):
    fair = fair_prices(M)
    jc = synth_jc_odds(fair)

    print(f"\n=== {home} vs {away}: 竞彩 single-match cross-market scan ===")
    print("(demo uses synthetic 竞彩 odds; feed live odds + market-anchored joint for real use)\n")
    print("market overrounds (竞彩 margin): "
          + "  ".join(f"{m}={overround(o):.3f}" for m, o in jc.items()))

    print("\n① SINGLE-MARKET VALUE (back where fair_prob × 竞彩_odds − 1 ≥ "
          f"{min_ev:.0%}, fairP ≥ 4%); 比分 is usually softest:")
    sv, skipped = single_market_value(fair, jc, min_ev=min_ev)
    if not sv:
        print("   none above threshold")
    print(f"   {'market':<8}{'sel':<8}{'竞彩@':>8}{'fair@':>8}{'fairP':>8}{'EV':>8}")
    for r in sv[:12]:
        print(f"   {r['market']:<8}{r['selection']:<8}{r['jc_odds']:>8.2f}"
              f"{r['fair_odds']:>8.2f}{r['fair_prob']*100:>7.1f}%{r['ev']*100:>+7.1f}%")
    if skipped:
        print(f"   (+{skipped} tail selections fairP<4% with nominal +EV SUPPRESSED — "
              f"ruinous variance + unreliable model tail)")
    print("   (demo crowd uses one under-bet multiplier → identical EVs are a synthetic")
    print("    artifact; real 竞彩 odds vary per selection.)")

    print("\n② CROSS-MARKET CONSISTENCY — back each 1X2 outcome the cheaper way")
    print("   (direct 胜平负 vs dutch of 比分 cells; both are legal single bets, NOT a parlay):")
    print(f"   {'outcome':<8}{'direct@':>9}{'synth@':>9}{'best via':<16}{'fairP':>7}{'EV best':>9}")
    for r in synthetic_vs_direct(fair, jc):
        print(f"   {r['outcome']:<8}{r['direct_odds']:>9.2f}{r['synth_odds']:>9.2f}"
              f"  {r['best_via']:<14}{r['fair_prob']*100:>6.1f}%{r['ev_best']*100:>+8.1f}%")

    print("\n③ CROSS-MATCH parlay margin tax (the only 串关 竞彩 allows — no correlation edge,")
    print("   different matches are independent; product pricing is correct, margin COMPOUNDS):")
    print("   " + "  ".join(f"{t['legs']}串={t['eff_take_pct']:.0f}%"
                            for t in parlay_margin_tax(1.16, 8)))
    print("   → an 8-leg parlay pays ~43% to the house; per-leg +EV must be huge to survive.")

    print("\nTakeaway: same-match parlay is banned, so the joint engine's 竞彩 edge is")
    print("①soft 比分 value + ②cross-market cheaper-representation — both single bets.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--min-ev", type=float, default=0.03)
    args = ap.parse_args()

    print("Fitting Dixon-Coles ...")
    params = fit(elo_prior_strength=args.prior)
    lh, la = params.predict_lambda(args.home, args.away, neutral=args.neutral)
    M = score_matrix(lh, la, rho=params.rho)
    print(f"  λ_{args.home}={lh:.2f}  λ_{args.away}={la:.2f}  ρ={params.rho:.3f}")
    print("  NOTE: for live use feed a MARKET-ANCHORED joint (bias-free), not the raw model —")
    print("  the raw DC λ carries the ENG-under/MAR-over bias. See market_anchored_findings.")
    report(args.home, args.away, M, min_ev=args.min_ev)


if __name__ == "__main__":
    main()
