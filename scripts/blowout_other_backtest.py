"""Rigorous validation of the 血洗局 '胜其他' thinning calibration (blowout_other.tail_cal).

The live tool assumes Poisson OVER-states the 胜其他 tail (favourites coast / garbage
time), thinning it by ~0.85 for WC-stakes games (vs ~0.59 friendly). Before betting real
money we check that empirically on 2014-2026 internationals, WITH confidence intervals.

Method (model-free, unbiased — uses ALL finished matches, no densest-slot sampling):
  - favourite = higher current Elo; mismatch = Elo-implied expected score E_fav.
  - 胜其他 = favourite wins by a score OUTSIDE the 竞彩 12-score listed grid (blowout_other.HOME).
  - per (competition-class × E_fav bin): realized 胜其他 rate vs Poisson-predicted rate
    (Poisson λ = that cell's mean fav/dog goals → score_matrix), thinning = realized/predicted.
  - bootstrap 95% CI (resample matches in the cell; λ + rate recomputed each draw) so the
    ratio carries its sampling error. Small cells → wide CI = the honest verdict.

CAVEAT: current Elo is a proxy for strength-at-the-time (fine for recent, noisier pre-2018);
prints n per cell so thin strata are visible. Run: PYTHONPATH=src python scripts/blowout_other_backtest.py
"""
from __future__ import annotations
import sqlite3
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "src")
from worldcup.models.markets import score_matrix
from worldcup.strategy.blowout_other import HOME  # the 12 listed home-win scores

DB = "data/worldcup.db"
LISTED = set(HOME)  # {(1,0),(2,0),...,(5,2)} oriented favourite-as-home


def fit_elo_to_lambda(rows):
    """Poisson regression log λ = b0 + b1·(E_fav-0.5) for favourite & underdog goals.
    Gives each match its OWN (λ_fav, λ_dog) → avoids the Jensen bias of plugging a
    cell-mean λ into a convex tail event."""
    x = np.array([r["e_fav"] - 0.5 for r in rows])
    yf = np.array([r["fav_g"] for r in rows], float)
    yd = np.array([r["dog_g"] for r in rows], float)

    def fit_one(y):
        def nll(b):
            lam = np.exp(b[0] + b[1] * x)
            return float(np.sum(lam - y * np.log(lam + 1e-12)))
        return minimize(nll, [np.log(y.mean() + 1e-6), 0.0], method="Nelder-Mead",
                        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 5000}).x

    bf, bd = fit_one(yf), fit_one(yd)
    for r in rows:
        xi = r["e_fav"] - 0.5
        r["lf"] = float(np.exp(bf[0] + bf[1] * xi))
        r["ld"] = float(np.exp(bd[0] + bd[1] * xi))
        r["p_other"] = poisson_other(r["lf"], r["ld"])
        r["p_narrow"] = poisson_narrow(r["lf"], r["ld"])
    return bf, bd


def comp_class(c: str | None) -> str:
    c = (c or "").lower()
    if "friendly" in c:
        return "friendly"
    if "qualif" in c:
        return "qualifier"
    if any(k in c for k in ("world cup", "euro", "copa", "nations", "asian cup",
                            "cup of nations", "gold cup", "confederations")):
        return "major"
    return "other"


def elo_expected(d_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** (-d_elo / 400.0))


def load():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    elo = {r["team_code"]: r["value"]
           for r in c.execute("SELECT team_code, value FROM team_ratings WHERE rating_type='elo'")}
    rows = []
    for m in c.execute("""SELECT home_code h, away_code a, home_score hs, away_score as_,
                                 competition comp FROM matches
                          WHERE home_score IS NOT NULL AND finished=1"""):
        if m["h"] not in elo or m["a"] not in elo:
            continue
        # orient favourite-as-home
        if elo[m["h"]] >= elo[m["a"]]:
            fav_g, dog_g, d = m["hs"], m["as_"], elo[m["h"]] - elo[m["a"]]
        else:
            fav_g, dog_g, d = m["as_"], m["hs"], elo[m["a"]] - elo[m["h"]]
        rows.append({"fav_g": fav_g, "dog_g": dog_g, "e_fav": elo_expected(d),
                     "cls": comp_class(m["comp"]),
                     "other": 1 if (fav_g > dog_g and (fav_g, dog_g) not in LISTED) else 0,
                     "narrow": 1 if (fav_g - dog_g == 1) else 0})
    c.close()
    return rows


def poisson_other(lf: float, ld: float) -> float:
    """Poisson-predicted P(favourite wins by an UNLISTED score)."""
    M = score_matrix(lf, ld, rho=0.0, max_goals=12)
    p_fav_win = float(np.tril(M, -1).sum())          # favourite (home-oriented) wins
    listed = float(sum(M[h, a] for h, a in LISTED))
    return p_fav_win - listed


def poisson_narrow(lf: float, ld: float) -> float:
    M = score_matrix(lf, ld, rho=0.0, max_goals=12)
    return float(sum(M[d + 1, d] for d in range(11)))  # favourite by exactly 1


def cell_stats(rows: list[dict], kind: str, nboot: int = 3000):
    """Return (n, realized_rate, poisson_pred, thinning, lo, hi) for 'other' or 'narrow'.
    Uses PER-MATCH Poisson predictions (r['p_other']/r['p_narrow']) — no Jensen bias."""
    if not rows:
        return None
    n = len(rows)
    hit = np.array([r[kind] for r in rows], float)                  # realized 0/1
    pred_arr = np.array([r["p_other" if kind == "other" else "p_narrow"] for r in rows], float)
    realized = hit.mean()
    pred = pred_arr.mean()
    thin = realized / pred if pred > 0 else float("nan")
    idx = np.arange(n)
    ratios = []
    rng_state = np.random.RandomState(12345)  # fixed seed (deterministic)
    for _ in range(nboot):
        b = rng_state.choice(idx, n, replace=True)
        p = pred_arr[b].mean()
        if p > 0:
            ratios.append(hit[b].mean() / p)
    lo, hi = (np.percentile(ratios, [2.5, 97.5]) if ratios else (float("nan"), float("nan")))
    return n, realized, pred, thin, lo, hi


def report(title, groups, kind):
    print(f"\n{'='*92}\n{title}\n{'='*92}")
    print(f"{'cell':<34}{'n':>5}{'realized':>10}{'poisson':>10}{'thin×':>8}{'95% CI':>18}")
    print("-" * 92)
    for label, rows in groups:
        s = cell_stats(rows, kind)
        if not s:
            print(f"{label:<34}{'0':>5}  (empty)")
            continue
        n, real, pred, thin, lo, hi = s
        flag = "  ⚠tiny" if n < 25 else ("  ~thin" if n < 60 else "")
        print(f"{label:<34}{n:>5}{real*100:>9.1f}%{pred*100:>9.1f}%{thin:>8.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>18}{flag}")


def main():
    rows = load()
    print(f"loaded {len(rows)} finished matches with Elo for both teams (2014-2026)")
    bf, bd = fit_elo_to_lambda(rows)
    lo_l = (float(np.exp(bf[0] + bf[1] * 0.10)), float(np.exp(bd[0] + bd[1] * 0.10)))
    hi_l = (float(np.exp(bf[0] + bf[1] * 0.45)), float(np.exp(bd[0] + bd[1] * 0.45)))
    print(f"Elo→λ fit: E_fav 0.60 → (λfav {lo_l[0]:.2f}, λdog {lo_l[1]:.2f}); "
          f"E_fav 0.95 → (λfav {hi_l[0]:.2f}, λdog {hi_l[1]:.2f}) [per-match preds, no Jensen bias]")

    # blowout stratum = big favourites (E_fav > 0.70). Stratify by competition class.
    blow = [r for r in rows if r["e_fav"] > 0.70]
    print(f"blowout stratum (E_fav>0.70): {len(blow)} matches")

    by_cls = defaultdict(list)
    for r in blow:
        by_cls[r["cls"]].append(r)
    groups_cls = [(f"{c} (E_fav>0.70)", by_cls[c]) for c in ("friendly", "qualifier", "major", "other")]
    # high-stakes = qualifier + major (the WC-relevant proxy)
    hi_stakes = by_cls["qualifier"] + by_cls["major"]
    groups_cls.append(("→ HIGH-STAKES (qual+major)", hi_stakes))
    report("胜其他 THINNING by competition class  (favourite E_fav>0.70)", groups_cls, "other")

    # mismatch gradient (high-stakes only) — does thinning vary with favourite strength?
    bins = [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    hs_all = [r for r in rows if r["cls"] in ("qualifier", "major")]
    groups_bin = [(f"E_fav {lo:.2f}-{hi:.2f}", [r for r in hs_all if lo <= r["e_fav"] < hi])
                  for lo, hi in bins]
    report("胜其他 THINNING by mismatch  (high-stakes: qualifier+major)", groups_bin, "other")

    # narrow-win (#4) check: favourite by exactly 1 — fattened? (NARROW_CAL=1.18 claim)
    report("窄胜 (favourite by EXACTLY 1) — NARROW_CAL claim ≈1.18×",
           [("all matches", rows), ("high-stakes E_fav>0.70", hi_stakes)], "narrow")

    print("\nReading: thin× = realized/Poisson. Live tool assumes ~0.85 (WC) / ~0.59 (friendly).")
    print("If a cell's 95% CI straddles 1.0 → can't even confirm thinning exists there.")
    print("⚠tiny (n<25) / ~thin (n<60) → treat the point estimate as a rough prior, not a number to bet on.")


if __name__ == "__main__":
    main()
