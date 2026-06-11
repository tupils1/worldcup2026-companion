"""串比分 (correct-score parlay) helper — the capped LOTTERY slice, built smart + honest.

This is the user's deliberate fun bucket: parlay several matches' EXACT scores (足彩/竞彩
串关), the hardest-to-hit, biggest-payout bet. It is unavoidably −EV (correct-score margins
are large and parlaying multiplies them). This tool does NOT pretend otherwise — it just:
  1. picks the score combos the model + tactics actually favour (DC score matrix, tilted by
     the VALIDATED 窄胜 1.18× — favourite-by-1 scores are ~18% fatter than naive Poisson),
  2. shows the BRUTAL TRUTH: the real joint hit-probability + fair odds, so you size it as the
     lottery ticket it is — a capped bit of the bankroll, a win is a surprise, not a plan.

Usage:
  # per-match top scores + the max-likelihood parlay:
  PYTHONPATH=src python -m worldcup.strategy.score_parlay --matches "GER-CUW,ESP-MAR,QAT-SUI"
  # evaluate YOUR specific ticket (+ EV if you pass the 竞彩 per-match 比分 odds):
  PYTHONPATH=src python -m worldcup.strategy.score_parlay --pick "GER-CUW:2-0,ESP-MAR:1-0" --jc "8.5,9.0"
"""
from __future__ import annotations
import argparse
import sqlite3

import numpy as np

from worldcup.models.dixon_coles import fit
from worldcup.models.markets import score_matrix
from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

NARROW_CAL = 1.18  # validated 2026-05-31 (blowout_other_backtest): favourite-by-1 ≈1.18× Poisson


def tilt_narrow(M: np.ndarray, home_fav: bool) -> np.ndarray:
    """Boost the favourite-by-exactly-1 scores by the validated 1.18×, renormalise."""
    M = M.copy()
    n = M.shape[0]
    for h in range(n):
        for a in range(n):
            if (home_fav and h - a == 1) or (not home_fav and a - h == 1):
                M[h, a] *= NARROW_CAL
    s = M.sum()
    return M / s if s > 0 else M


def top_scores(M: np.ndarray, k: int = 6) -> list[tuple[int, int, float]]:
    flat = [(h, a, float(M[h, a])) for h in range(M.shape[0]) for a in range(M.shape[1])]
    flat.sort(key=lambda x: -x[2])
    return flat[:k]


def match_scores(p, home: str, away: str, k: int = 6):
    if home not in p.attack or away not in p.attack:
        return None
    lh, la = p.predict_lambda(home, away, neutral=True)
    M = tilt_narrow(score_matrix(lh, la, rho=p.rho, max_goals=8), home_fav=(lh >= la))
    return {"lh": lh, "la": la, "top": top_scores(M, k), "M": M}


def _tac(conn, home, away):
    try:
        r = conn.execute("SELECT arch_h, arch_a, q_lean, t_tg FROM daily_tactics "
                         "WHERE home=? AND away=?", (home, away)).fetchone()
        if r:
            return f"{r['arch_h'] or '?'}×{r['arch_a'] or '?'} 总进球倾向 量化{r['q_lean'] or '—'}/战术{r['t_tg'] or '—'}"
    except Exception:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", help="逗号分隔对阵, e.g. GER-CUW,ESP-MAR,QAT-SUI")
    ap.add_argument("--pick", help="你的串关, e.g. GER-CUW:2-0,ESP-MAR:1-0")
    ap.add_argument("--jc", help="对应各场竞彩比分赔率(逗号分隔), 给了就算 EV")
    ap.add_argument("--top", type=int, default=6)
    args = ap.parse_args()

    print("拟合 Dixon-Coles ...")
    p = fit(elo_prior_strength=0.5)
    conn = get_conn(DEFAULT_DB_PATH)

    if args.pick:
        legs = [x.strip() for x in args.pick.split(",") if x.strip()]
        jc = [float(x) for x in args.jc.split(",")] if args.jc else None
        joint = 1.0
        print("\n=== 你的串比分 ===")
        for i, leg in enumerate(legs):
            mt, sc = leg.split(":")
            h, a = mt.split("-")
            sh, sa = (int(x) for x in sc.replace("：", ":").replace("-", ":").split(":"))
            ms = match_scores(p, h, a)
            if not ms:
                print(f"  {mt} {sc}: 缺球队强度,跳过"); continue
            prob = float(ms["M"][sh, sa]) if sh < ms["M"].shape[0] and sa < ms["M"].shape[1] else 0.0
            joint *= prob
            print(f"  {mt:9} 比分 {sh}:{sa}  模型概率 {prob*100:5.2f}%   {_tac(conn,h,a)}")
        fair = (1.0 / joint) if joint > 0 else float("inf")
        print(f"\n联合命中概率 ≈ {joint*100:.4f}%  → 约 {fair:,.0f} 串 1 中 1 次, 公平赔率 1:{fair:,.0f}")
        if jc and len(jc) == len(legs):
            payout = 1.0
            for o in jc:
                payout *= o
            ev = joint * payout - 1.0
            print(f"竞彩串关赔率 ≈ {payout:,.1f}×  → EV = {ev*100:+.1f}%  "
                  f"({'惊喜但仍−EV' if ev<0 else '罕见+EV,核对赔率!'})")
        print("\n⚠ 这是封顶的彩票切片:明知 −EV,中了是惊喜、不中是看球门票。仓位务必小、亏完即止。")
        conn.close(); return

    if not args.matches:
        print("给 --matches 或 --pick。"); conn.close(); return

    mts = [m.strip() for m in args.matches.split(",") if m.strip()]
    print(f"\n=== 各场最可能比分 (DC + 窄胜1.18×; top {args.top}) ===")
    parlay, joint = [], 1.0
    for mt in mts:
        h, a = mt.split("-")
        ms = match_scores(p, h, a, args.top)
        if not ms:
            print(f"\n{mt}: 缺球队强度,跳过"); continue
        print(f"\n{mt}  (λ {ms['lh']:.2f}-{ms['la']:.2f})  {_tac(conn,h,a)}")
        for sh, sa, pr in ms["top"]:
            print(f"    {sh}:{sa}   {pr*100:5.2f}%")
        bh, ba, bp = ms["top"][0]
        parlay.append((mt, bh, ba, bp)); joint *= bp
    if parlay:
        fair = (1.0 / joint) if joint > 0 else float("inf")
        print("\n=== 最大概率串(每场取最可能比分)===")
        print("  " + "  ".join(f"{mt} {bh}:{ba}" for mt, bh, ba, _ in parlay))
        print(f"  联合命中 ≈ {joint*100:.4f}%  → 约 {fair:,.0f} 串 1 中 1 次  (公平赔率 1:{fair:,.0f})")
        print("\n提示:想要更大赔付就挑更冷的比分(命中更低、赔更高);窄胜分(1:0/2:1)是验证过相对偏厚的。")
        print("⚠ 封顶彩票切片:−EV,小注、亏完即止;中了请当惊喜。")
    conn.close()


if __name__ == "__main__":
    main()
