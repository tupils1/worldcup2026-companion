"""Blowout 胜其他 (favorite-wins-by-an-unlisted-score) scanner.

⚠ REFUTED 2026-05-31 — the central 'thinning' premise did NOT survive a rigorous
backtest (scripts/blowout_other_backtest.py, 2014-26, per-match Poisson + bootstrap CI):
high-stakes blowout 胜其他 realised/Poisson = 1.92 [0.78, 3.25] — point estimate FATTER
than Poisson, CI straddles 1.0, single-digit events per cell → uncalibratable. tail_cal is
now NEUTRALISED to 1.0 (raw Poisson). Treat 胜其他 as UNVALIDATED: size tiny or skip.
The 窄胜 NARROW_CAL=1.18 DID survive (1.18 [1.00, 1.36], n=379) and is kept.
The historical narrative below is RETAINED for context but is superseded by the backtest.

HYPOTHESIS (user, 2026-05-30): in WC mismatches the 竞彩 比分 "胜其他"/"负其他" bucket
(favorite wins by a score beyond the listed grid: 6:0, 7:1, 5:3, …) might be underpriced
because casuals under-imagine extreme scores.

EMPIRICAL CHECK FIRST (martj42 internationals 2006+, the decisive input):
  real "其他"-bucket frequency vs Poisson-predicted —
    胜其他 overall 0.86×   ·   负其他 overall 0.67×   ·   BLOWOUT (max P>70%, n=123) **0.61×**
  → reality's blowout tail is ~40% THINNER than Poisson. Favorites EASE OFF when well
  ahead (garbage time) — so vanilla Poisson OVER-states 胜其他, and the crowd's "6:0 is
  rare" intuition is partly RATIONAL. This LEANS AGAINST the hypothesis.

REVIVAL (2026-05-30): the 0.61× was FRIENDLY-DOMINATED. Splitting blowouts by competition:
    friendly 0.59  ·  major-tournament 0.75  ·  qualifier 0.93.
High-stakes games barely thin — favourites run up scores when it matters. WC GROUP games
(high stakes + GD tiebreaker + full squads) belong at ~0.85, NOT 0.61. So this tool now
uses a WC-group thinning of ~0.85 (vs the friendly-blended 0.61), which DROPS the breakeven
竞彩 odds and moves the hypothesis from "almost certainly NO BET" to "plausibly live —
check real June odds." Still applies a mild extra taper for extreme blowouts.

It does NOT fatten the tail (negative-binomial = wrong direction). It reports the breakeven
竞彩 odds; fire only if a real 竞彩 quote exceeds it. Use in June when WC matches list on
竞彩 — scrape the 胜其他/负其他 odds and compare.

CAVEATS: thinning factor from a SMALL sample (n=123 blowouts, ~6-10 events) → wide error
bars; treat as a rough prior. Model also under-predicts mismatch goals (see λ-calibration).

Run:
    PYTHONPATH=src python -m worldcup.strategy.blowout_other
    PYTHONPATH=src python -m worldcup.strategy.blowout_other --jc-odds 18   # eval top fixture vs a 竞彩 胜其他 quote
"""

from __future__ import annotations

import argparse

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.models.dixon_coles import fit
from worldcup.models.markets import score_matrix, goal_margin_dist

HOME = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2)]
AWAY = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (0, 5), (1, 5), (2, 5)]
JC_RETURN = 0.72   # 竞彩 比分 effective return (~28% take) — context for how far below fair 竞彩 prices
# Narrow-win (#4): favourite-by-exactly-1 occurs ~1.18× Poisson (favourites ease off → win
# by less), so margin-1 scores (1:0,2:1,3:2) are FATTER than naive Poisson.
# ✅ VALIDATED 2026-05-31 (blowout_other_backtest.py): high-stakes realised/Poisson =
# 1.18 [1.00, 1.36] (n=379) — point matches exactly, CI just excludes 1.0. The one
# blowout-tail effect that survived. TENSION for betting: true prob is higher (good) BUT
# casuals OVERBET pretty narrow scores (1:0,2:1) → 竞彩 may shade them SHORT. Net sign
# unknown until real June 竞彩 odds.
NARROW_CAL = 1.18


def tail_cal(p_fav: float) -> float:
    """胜其他 Poisson-tail adjustment — NEUTRALISED to 1.0 (raw Poisson).

    REFUTED 2026-05-31 (scripts/blowout_other_backtest.py; 2014-26 internationals,
    PER-MATCH Poisson preds, bootstrap CI). The earlier 0.85 'thinning' (assume real
    blowouts are RARER than Poisson) is both unsupported and likely WRONG-SIGNED:
        high-stakes blowout 胜其他 realised/Poisson = 1.92, 95% CI [0.78, 3.25]
    i.e. the point estimate is FATTER than Poisson (overdispersion — red cards/collapses,
    which DC's rho doesn't touch), and the CI straddles 1.0 because there are only
    single-digit 胜其他 events per high-stakes cell. It cannot be calibrated in EITHER
    direction with the data we have. So use raw Poisson (1.0) as the honest neutral and
    treat any 胜其他 'edge' as UNVALIDATED — size tiny or skip.
    (NARROW_CAL below SURVIVED the same backtest: 1.18, CI [1.00, 1.36], n=379 — kept.)
    p_fav kept in the signature for callers/back-compat."""
    return 1.0


def favorite_other(lh: float, la: float, rho: float) -> dict:
    """Favourite-side 'other' bucket: Poisson prob, empirically-thinned prob, breakeven odds."""
    M = score_matrix(lh, la, rho=rho, max_goals=12)
    ph = float(np.tril(M, -1).sum()); pa = float(np.triu(M, 1).sum())
    home_fav = lh >= la
    p_fav = ph if home_fav else pa
    listed = HOME if home_fav else AWAY
    side_p = ph if home_fav else pa
    poisson_other = side_p - float(sum(M[h, a] for h, a in listed))
    cal = tail_cal(p_fav)
    cal_other = poisson_other * cal
    # 窄胜 (#4): favourite wins by exactly 1 goal, fattened by NARROW_CAL (1.18×).
    md = goal_margin_dist(M)
    margin1 = md.get(1, 0.0) if home_fav else md.get(-1, 0.0)
    narrow_true = margin1 * NARROW_CAL
    return {
        "bucket": "胜其他" if home_fav else "负其他",
        "p_fav": p_fav, "poisson_other": poisson_other, "cal_factor": cal,
        "cal_other": cal_other,
        "fair_odds": (1.0 / cal_other) if cal_other > 0 else float("inf"),       # breakeven 竞彩 odds
        "typical_jc": (JC_RETURN / cal_other) if cal_other > 0 else float("inf"),
        "narrow_poisson": margin1, "narrow_true": narrow_true,
        "narrow_fair": (1.0 / narrow_true) if narrow_true > 0 else float("inf"),  # fair odds for 'fav by exactly 1'
    }


def scan(min_p_fav: float = 0.60, db_path=DEFAULT_DB_PATH) -> list[dict]:
    p = fit(elo_prior_strength=0.5)
    conn = get_conn(db_path)
    fx = conn.execute(
        "SELECT home_code,away_code,match_date FROM matches WHERE finished=0 "
        "AND match_date BETWEEN '2026-06-01' AND '2026-07-31'"
    ).fetchall()
    conn.close()
    seen, out = set(), []
    for r in fx:
        h, a = r["home_code"], r["away_code"]
        key = tuple(sorted((h, a)))
        if key in seen or h not in p.attack or a not in p.attack:
            continue
        seen.add(key)
        lh, la = p.predict_lambda(h, a, neutral=True)
        d = favorite_other(lh, la, p.rho)
        if d["p_fav"] < min_p_fav:
            continue
        fav = h if lh >= la else a
        d.update({"match": f"{h}-{a}", "fav": fav, "lh": lh, "la": la})
        out.append(d)
    out.sort(key=lambda x: -x["cal_other"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--jc-odds", type=float, default=None,
                    help="A 竞彩 胜其他 odds to evaluate against the TOP fixture's thinned fair prob")
    args = ap.parse_args()

    print("=" * 92)
    print("BLOWOUT 胜其他 — tail_cal NEUTRALISED to 1.0 (raw Poisson); thinning REFUTED by backtest")
    print("⚠ 胜其他 = UNVALIDATED (realised/Poisson 1.92 [0.78,3.25], n tiny) → size tiny or SKIP.")
    print("=" * 92)
    rows = scan()
    print(f"\n{'match':<11}{'fav':<5}{'λfav':>5}{'λdog':>5}{'bucket':<7}{'Poisson':>8}"
          f"{'×cal':>6}{'→true':>7}{'breakeven竞彩@':>14}")
    print("-" * 92)
    for d in rows[:args.top]:
        print(f"{d['match']:<11}{d['fav']:<5}{d['lh'] if d['lh']>=d['la'] else d['la']:>5.2f}"
              f"{d['la'] if d['lh']>=d['la'] else d['lh']:>5.2f}{d['bucket']:<7}"
              f"{d['poisson_other']*100:>7.1f}%{d['cal_factor']:>6.2f}{d['cal_other']*100:>6.1f}%"
              f"{d['fair_odds']:>13.0f}")
    print("\n── 窄胜桶 (#4: favourite wins by EXACTLY 1 — 1:0/2:1/3:2; fattened 1.18×) ──")
    print(f"{'match':<11}{'fav':<5}{'narrow Poisson':>15}{'×1.18→true':>12}{'fair@':>7}")
    for d in rows[:args.top]:
        print(f"{d['match']:<11}{d['fav']:<5}{d['narrow_poisson']*100:>14.1f}%"
              f"{d['narrow_true']*100:>11.1f}%{d['narrow_fair']:>7.1f}")
    print("  June: sum 竞彩's de-vig (1:0+2:1+3:2) implied vs 'true' above. TENSION: true is HIGHER")
    print("  (fav eases off → more narrow wins) BUT casuals OVERBET 1:0/2:1 → 竞彩 may shade SHORT.")
    print("  So check the SIGN on real odds — narrow-win could be over- OR under-priced.")

    print("\nbreakeven竞彩@ = you need 竞彩 to offer MORE than this to be +EV (true thinned prob).")
    print(f"竞彩 typically prices the bucket at ~{JC_RETURN:.0%}×fair (post-margin), i.e. BELOW breakeven —")
    print("so the default verdict is NO BET. Only fire if a real 竞彩 quote exceeds breakeven.")

    if args.jc_odds and rows:
        top = rows[0]
        ev = top["cal_other"] * args.jc_odds - 1.0
        print(f"\nEVAL top fixture {top['match']} {top['bucket']} vs 竞彩 @{args.jc_odds:.1f}:")
        print(f"  true thinned prob {top['cal_other']*100:.1f}%  → EV = {ev*100:+.1f}%  "
              f"({'+EV ✓ (rare!)' if ev>0 else 'NO BET — below breakeven'})")

    print("\n⚠ 胜其他 thinning REFUTED 2026-05-31 (scripts/blowout_other_backtest.py): high-stakes")
    print("  realised/Poisson 1.92 [0.78,3.25] — point FATTER, CI straddles 1, single-digit events.")
    print("  Now raw Poisson. The ONLY blowout effect that survived is 窄胜 1.18× [1.00,1.36].")


if __name__ == "__main__":
    main()
