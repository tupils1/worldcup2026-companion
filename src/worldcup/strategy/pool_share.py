"""Pari-mutuel pool-share analysis for 足彩 胜负彩 (14/14).

Two parts:

A. EMPIRICAL (real data from `lottery_draws`): quantify the two things that decide
   whether 足彩 has alpha at all —
     1. the effective RETURN RATE (the "take" wall — payouts / sales), and
     2. the UPSET PREMIUM (upset-heavy rounds → fewer winners → far bigger payout).
   If the crowd is beatable, it shows up as: winners collapse and payout explodes
   exactly when the result deviates from the crowd's favourite-heavy picks.

B. EV ENGINE: the pari-mutuel math. A ticket's edge is NOT just P(correct) — it's
   P(correct) × payout, and payout = pool / (#co-winners). #co-winners depends on how
   POPULAR your picked combo is. So among near-equally-likely outcomes, picking the
   LESS-popular one (where true_prob >> crowd_prob) raises BOTH terms. We compare a
   chalk ticket (what the crowd does) vs a contrarian-value ticket and check whether
   either clears the take.

Crowd distribution: ideally per-match 选择比例 (not yet ingestible — JS sites). Until
then use an odds/favourite-longshot PROXY (passed in). Demonstrated here on an
illustrative 14-match round; wire real probs (market-anchored) + crowd% for live use.

Run:
    PYTHONPATH=src python -m worldcup.strategy.pool_share
"""

from __future__ import annotations

import math

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

OUTCOMES = ("home", "draw", "away")          # maps to lottery codes 3 / 1 / 0
CODE = {"home": "3", "draw": "1", "away": "0"}
JACKPOT_CAP = 5_000_000.0                     # 胜负彩 头奖单注封顶 500万
TICKET = 2.0                                  # ¥2 per 注


# ─────────────────────────── A. empirical ────────────────────────────────────
def load_sfc(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM lottery_draws WHERE game='sfc' AND results IS NOT NULL "
        "ORDER BY issue DESC"
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["res"] = d["results"].split()
        out.append(d)
    return out


def empirical_report(draws: list[dict]) -> None:
    print("=" * 88)
    print("A. EMPIRICAL POOL DYNAMICS — 胜负彩 (real draws)")
    print("=" * 88)
    print(f"{'issue':<7}{'date':<12}{'H/D/A':>9}{'1st注':>8}{'1st派彩':>11}"
          f"{'2nd派彩':>10}{'sales(万)':>10}{'返奖%':>8}")
    print("-" * 88)
    rets = []
    for d in draws:
        n_h = d["res"].count("3"); n_d = d["res"].count("1"); n_a = d["res"].count("0")
        ret = None
        if d["sales"] and d["first_n"] is not None and d["first_pay"] is not None:
            paid = d["first_n"] * d["first_pay"] + (d["second_n"] or 0) * (d["second_pay"] or 0)
            ret = paid / d["sales"]
            rets.append(ret)
        print(f"{d['issue']:<7}{d['draw_date'] or '?':<12}{f'{n_h}/{n_d}/{n_a}':>9}"
              f"{(d['first_n'] if d['first_n'] is not None else -1):>8}"
              f"{(d['first_pay'] or 0):>11,.0f}{(d['second_pay'] or 0):>10,.0f}"
              f"{(d['sales'] or 0)/1e4:>10,.0f}{(ret*100 if ret is not None else float('nan')):>7.1f}%")
    print("-" * 88)
    print("Reading: more away(0)+draw(1) = more 'upset' = fewer 1st-prize winners → bigger 单注派彩.")
    print("  (头奖封顶 500万 + 滚存 distorts single-issue 返奖%; the 二等奖派彩 swing is the cleaner")
    print("   pari-mutuel signal.) 'crowd skill' = winners are FAR above random — the public picks")
    print("   favourites well, so edge must come from the few outcomes they systematically misjudge.")


# ─────────────────────────── B. EV engine ────────────────────────────────────
def expected_co_winners(combo: list[str], crowd: list[dict], n_tickets: float) -> float:
    """E[# other tickets matching `combo`] = n_tickets · Π crowd_prob(picked outcome)."""
    p = 1.0
    for i, oc in enumerate(combo):
        p *= max(crowd[i][oc], 1e-9)
    return n_tickets * p


def ticket_ev(combo: list[str], p_true: list[dict], crowd: list[dict],
              pool: float, n_tickets: float, cap: float = JACKPOT_CAP) -> dict:
    """EV of one ¥2 ticket on `combo`. Payout if correct = min(cap, pool/(co-winners+1))."""
    p_correct = math.prod(p_true[i][oc] for i, oc in enumerate(combo))
    co = expected_co_winners(combo, crowd, n_tickets)
    payout = min(cap, pool / (co + 1.0))
    ev = p_correct * payout - TICKET
    return {"p_correct": p_correct, "co_winners": co, "payout": payout,
            "ev": ev, "roi": ev / TICKET}


def pick_chalk(crowd: list[dict]) -> list[str]:
    return [max(OUTCOMES, key=lambda o: m[o]) for m in crowd]


def pick_value(p_true: list[dict], crowd: list[dict]) -> list[str]:
    """Per match pick the outcome with the best true/crowd ratio among outcomes that
    are still reasonably likely (true prob ≥ 20%) — i.e. underbet-but-plausible."""
    out = []
    for i, m in enumerate(p_true):
        cand = [o for o in OUTCOMES if m[o] >= 0.20] or list(OUTCOMES)
        out.append(max(cand, key=lambda o: m[o] / max(crowd[i][o], 1e-9)))
    return out


def ev_demo() -> None:
    print("\n" + "=" * 88)
    print("B. EV ENGINE — chalk vs contrarian-value ticket (ILLUSTRATIVE 14-match round)")
    print("=" * 88)
    # Illustrative round: 14 matches. p_true = our (market-anchored) estimate;
    # crowd = favourite-longshot-skewed (public over-weights the favourite).
    def skew(p):  # push crowd toward the favourite outcome vs true probs
        fav = max(p, key=p.get)
        return {o: (p[o] * (2.1 if o == fav else 0.6)) for o in OUTCOMES}
    def norm(d):
        s = sum(d.values()); return {k: v / s for k, v in d.items()}

    # A spread of match types (some near-locks, some coin-flips with a live underdog)
    base = [
        {"home": .70, "draw": .20, "away": .10}, {"home": .55, "draw": .27, "away": .18},
        {"home": .45, "draw": .28, "away": .27}, {"home": .38, "draw": .30, "away": .32},
        {"home": .60, "draw": .25, "away": .15}, {"home": .33, "draw": .33, "away": .34},
        {"home": .50, "draw": .27, "away": .23}, {"home": .42, "draw": .29, "away": .29},
        {"home": .65, "draw": .22, "away": .13}, {"home": .40, "draw": .31, "away": .29},
        {"home": .48, "draw": .28, "away": .24}, {"home": .36, "draw": .30, "away": .34},
        {"home": .58, "draw": .26, "away": .16}, {"home": .44, "draw": .30, "away": .26},
    ]
    p_true = [norm(m) for m in base]
    crowd = [norm(skew(m)) for m in base]

    pool, sales = 12_000_000.0, 16_000_000.0
    n_tickets = sales / TICKET

    for label, combo in [("chalk (crowd favourites)", pick_chalk(crowd)),
                         ("contrarian-value", pick_value(p_true, crowd))]:
        r = ticket_ev(combo, p_true, crowd, pool, n_tickets)
        agree = sum(a == b for a, b in zip(combo, pick_chalk(crowd)))
        print(f"\n  {label}:  picks={' '.join(CODE[o] for o in combo)}  (差异 {14-agree} 场 vs chalk)")
        print(f"    P(all 14 correct) = {r['p_correct']:.2e}   E[co-winners] = {r['co_winners']:,.0f}")
        print(f"    payout if hit = ¥{r['payout']:,.0f}   EV/¥2 = {r['ev']:+.4f}  (ROI {r['roi']*100:+.1f}%)")
    print("\n  Mechanism: the contrarian ticket sacrifices a little P(correct) but lands on")
    print("  far fewer co-winners → much bigger payout-if-hit. Net EV beats chalk; whether it")
    print("  clears the ~33% take depends on how biased the crowd is THIS round.")
    print("  → Real use: feed market-anchored p_true + real 选择比例 (or odds proxy) for the")
    print("    actual 14 matches; only fire when the optimised ticket's EV > 0.")


def main() -> None:
    conn = get_conn(DEFAULT_DB_PATH)
    draws = load_sfc(conn)
    conn.close()
    if draws:
        empirical_report(draws)
    else:
        print("No 胜负彩 draws ingested yet — run `python -m worldcup.ingest.lottery_cn` first.")
    ev_demo()
    print(f"\n(n={len(draws)} draws — East Money gives ~5 latest/fetch; accumulates over time.")
    print(" For a deep backtest now: Chrome-MCP scrape 500/竞彩网 history, or user-provided.)")


if __name__ == "__main__":
    main()
