"""任选九场 (Pick-9-of-14) pari-mutuel EV model.

WHY 任九, NOT 胜负彩(14/14): pool_share.py showed 14/14 is a jackpot lottery
(P(all 14)~1e-5, payout capped ¥5M → deeply −EV). 任九 is structurally different and
is the real alpha candidate because:
  1. You CHOOSE which 9 of the 14 to predict → skip the 5 hardest → P(win) jumps to
     a monetizable 1-10% range (vs 1e-5).
  2. 任九 头奖 is UNCAPPED (奖金 = 任九奖金 / 中奖注数) → contrarian upside isn't clipped.

THE PARI-MUTUEL EV (one ¥2 单式 ticket = pick exactly 9 matches, 1 outcome each, win
if all 9 correct):
    P(win)      = Π_{i∈your9} p_true(pick_i)
    payout      = pool / (E[co-winners] + 1)              # 任九 uncapped
    E[co-winners] ≈ f · N · Π_{i∈your9} crowd(pick_i)     # N = sales/¥2  (Model A, see below)
    EV          = P(win)·payout − 2

THE HURDLE (clean, falsifiable): when winners ≫ 1, payout ≈ (pool/N)/Π crowd, so
    EV ≈ (pool/N)·Π_{i∈9} [p_true/crowd] − 2.
With pool ≈ return_rate·sales and N = sales/2, pool/N ≈ 2·return_rate, so EV>0 ⟺
    Π_{i∈9} (p_true/crowd) > 1/return_rate ≈ 1/0.65 ≈ 1.54.
i.e. your 9 picks must compound a >54% value-ratio edge — about a 4.9% crowd-mispricing
PER match on average (1.049^9 ≈ 1.54). Concrete, and usually only reachable in rounds
where the crowd is systematically biased (over-loading favourites / 主队).

CO-WINNER MODEL A (approximation, labelled): E[co-winners] ≈ f·N·Π crowd(pick_i)
treats the field as if everyone bet YOUR 9 matches with crowd-marginal outcomes. Real
任九 has bettors choosing different 9-subsets + heavy 复式, so this under-counts absolute
winners — `field_concentration f` (default 3.0) inflates it to a realistic level and
should be CALIBRATED against observed `lottery_draws.first_n` once per-match 选择比例 is
ingestible. It correctly captures the RELATIVE lever (contrarian combos → fewer co-winners).

INPUT p_true: use MARKET-ANCHORED per-match W/D/L (sharp, bias-free) — see
[[market-anchored-findings]]. crowd: per-match 选择比例 when available, else odds proxy.

Run:
    PYTHONPATH=src python -m worldcup.strategy.renjiu_ev
"""

from __future__ import annotations

import math

from worldcup.strategy.pool_share import CODE, OUTCOMES, TICKET

P_FLOOR = 0.15                 # don't pick an outcome below this true prob (survival)
DEFAULT_FIELD_CONCENTRATION = 3.0
# Min co-winners: in a market of millions of 注 (incl. 复式 covering many subsets),
# you can NEVER be the sole winner — even upset-heavy real 任九 rounds had ≥~100 winners
# (observed first_n: 591 / 111 / 21,815). This floor caps max payout at pool/floor and is
# what TAMES the naive-contrarian trap (without it, Model A lets payout→whole pool at
# P(win)≈0). CALIBRATE to the observed minimum first_n once 选择比例 data is in.
DEFAULT_WINNER_FLOOR = 80.0


def pool_from_sales(sales: float, return_rate: float = 0.65) -> float:
    return return_rate * sales


def ticket_pwin(picks: list[str], p_true: list[dict]) -> float:
    return math.prod(p_true[i][o] for i, o in enumerate(picks))


def ticket_ev(idx9: list[int], picks: dict[int, str], p_true, crowd,
              pool: float, n_tickets: float, f: float,
              winner_floor: float = DEFAULT_WINNER_FLOOR) -> dict:
    """EV of a 9-match 单式 ticket (idx9 = chosen match indices; picks = idx→outcome)."""
    pwin = math.prod(p_true[i][picks[i]] for i in idx9)
    co_model = f * n_tickets * math.prod(crowd[i][picks[i]] for i in idx9)
    co = max(co_model, winner_floor)          # can't be sole winner in a market this big
    payout = pool / (co + 1.0)
    ev = pwin * payout - TICKET
    return {"idx9": idx9, "pwin": pwin, "co_winners": co, "payout": payout,
            "ev": ev, "roi": ev / TICKET,
            "value_product": math.prod(p_true[i][picks[i]] / max(crowd[i][picks[i]], 1e-9)
                                        for i in idx9)}


def _pick_value_outcome(p: dict, c: dict) -> str:
    """Best value (true/crowd) outcome for a match, among outcomes still plausible."""
    cand = [o for o in OUTCOMES if p[o] >= P_FLOOR] or list(OUTCOMES)
    return max(cand, key=lambda o: p[o] / max(c[o], 1e-9))


def _pick_fav(p: dict) -> str:
    return max(OUTCOMES, key=lambda o: p[o])


def optimize(p_true: list[dict], crowd: list[dict], pool: float, sales: float,
             f: float = DEFAULT_FIELD_CONCENTRATION,
             winner_floor: float = DEFAULT_WINNER_FLOOR) -> dict:
    """Best +EV ticket (pure EV hill-climb), plus the chalk baseline, the naive-
    contrarian comparison, and the break-even hurdle.

    No P(win) floor is needed: real 任九 winning tickets inherently have P(win)~0.1-0.5%,
    so a P(win) floor is wrong. Instead the WINNER FLOOR caps payout at pool/floor, which
    auto-kills the degenerate longshot tickets (their EV = P(win)·capped-payout ≈ −2) while
    leaving the legitimate moderate-value interior optimum intact."""
    n = len(p_true)
    n_tickets = sales / TICKET

    def ev_of(idx9, picks):
        return ticket_ev(idx9, picks, p_true, crowd, pool, n_tickets, f, winner_floor)

    fav = {i: _pick_fav(p_true[i]) for i in range(n)}
    val = {i: _pick_value_outcome(p_true[i], crowd[i]) for i in range(n)}

    # chalk: 9 most-confident matches, favourite outcome (= what the crowd does)
    chalk_idx = sorted(range(n), key=lambda i: -p_true[i][fav[i]])[:9]
    chalk = ev_of(sorted(chalk_idx), fav)

    # naive max value-ratio (for comparison — now tamed by the winner floor)
    val_ratio = {i: p_true[i][val[i]] / max(crowd[i][val[i]], 1e-9) for i in range(n)}
    naive_idx = sorted(range(n), key=lambda i: -val_ratio[i])[:9]
    naive = ev_of(sorted(naive_idx), val)

    # ── steepest-ascent hill-climb from chalk: each sweep finds the single best move
    #    (swap a match, or flip an outcome), applies it, repeats. Never mutates the
    #    working set mid-scan, so the ticket stays exactly 9 matches. ──
    cur_idx = frozenset(chalk_idx)
    cur_picks = dict(fav)
    best = ev_of(sorted(cur_idx), cur_picks)
    while True:
        move = None  # (delta_ev, new_idx_frozenset, new_picks)
        # swap one in-ticket match for one out-ticket match (any plausible outcome)
        for i_in in cur_idx:
            for i_out in range(n):
                if i_out in cur_idx:
                    continue
                for oc in OUTCOMES:
                    if p_true[i_out][oc] < P_FLOOR:
                        continue
                    new_idx = (cur_idx - {i_in}) | {i_out}
                    new_picks = {**cur_picks, i_out: oc}
                    cand = ev_of(sorted(new_idx), new_picks)
                    if cand["ev"] > best["ev"] + 1e-9 and (move is None or cand["ev"] > move[0]):
                        move = (cand["ev"], new_idx, new_picks, cand)
        # flip the outcome of an in-ticket match
        for i in cur_idx:
            for oc in OUTCOMES:
                if oc == cur_picks[i] or p_true[i][oc] < P_FLOOR:
                    continue
                new_picks = {**cur_picks, i: oc}
                cand = ev_of(sorted(cur_idx), new_picks)
                if cand["ev"] > best["ev"] + 1e-9 and (move is None or cand["ev"] > move[0]):
                    move = (cand["ev"], cur_idx, new_picks, cand)
        if move is None:
            break
        _, cur_idx, cur_picks, best = move

    hurdle = sales / pool   # Π(p_true/crowd) over 9 picks must exceed this for +EV
    return {"chalk": chalk, "naive_trap": naive, "optimized": best,
            "picks_opt": cur_picks, "hurdle_product": hurdle,
            "per_match_avg_ratio_needed": hurdle ** (1.0 / 9.0),
            "winner_floor": winner_floor,
            "n_tickets": n_tickets, "return_rate": pool / sales}


# ─────────────────────────── demo ────────────────────────────────────────────
def _norm(d):
    s = sum(d.values()); return {k: v / s for k, v in d.items()}


def _demo_round():
    """14 illustrative matches; crowd over-weights the favourite (favourite-longshot bias)."""
    base = [
        {"home": .70, "draw": .20, "away": .10}, {"home": .55, "draw": .27, "away": .18},
        {"home": .45, "draw": .28, "away": .27}, {"home": .38, "draw": .30, "away": .32},
        {"home": .60, "draw": .25, "away": .15}, {"home": .33, "draw": .33, "away": .34},
        {"home": .50, "draw": .27, "away": .23}, {"home": .42, "draw": .29, "away": .29},
        {"home": .65, "draw": .22, "away": .13}, {"home": .40, "draw": .31, "away": .29},
        {"home": .48, "draw": .28, "away": .24}, {"home": .36, "draw": .30, "away": .34},
        {"home": .58, "draw": .26, "away": .16}, {"home": .44, "draw": .30, "away": .26},
    ]
    p_true = [_norm(m) for m in base]
    # Crowd over-weights the favourite (favourite-longshot bias). 1.5×/0.8× is a
    # MILD, realistic skew — strong enough to demonstrate the mechanism without the
    # absurd EV that an extreme skew produces. Real rounds vary; only the biased ones win.
    def skew(p):
        fav = max(p, key=p.get)
        return _norm({o: p[o] * (1.5 if o == fav else 0.8) for o in OUTCOMES})
    crowd = [skew(m) for m in base]
    return p_true, crowd


def main() -> None:
    p_true, crowd = _demo_round()
    sales = 12_000_000.0
    pool = pool_from_sales(sales, return_rate=0.65)
    r = optimize(p_true, crowd, pool, sales)

    print("=" * 90)
    print("任选九场 EV MODEL — pick 9 of 14, win if all 9 correct (ILLUSTRATIVE round)")
    print("=" * 90)
    print(f"sales ¥{sales:,.0f}  pool ¥{pool:,.0f}  return {r['return_rate']*100:.0f}%  "
          f"tickets {r['n_tickets']:,.0f}")
    print(f"\nBREAK-EVEN HURDLE: Π(p_true/crowd) over your 9 picks must exceed "
          f"{r['hurdle_product']:.3f}")
    print(f"  → avg {r['per_match_avg_ratio_needed']:.3f}× per match "
          f"(~{(r['per_match_avg_ratio_needed']-1)*100:.1f}% crowd-mispricing each, compounded).")

    print(f"\n{'strategy':<24}{'P(win)':>9}{'E[co-win]':>11}{'payout¥':>11}"
          f"{'EV/¥2':>9}{'ROI':>8}{'Πratio':>9}")
    print("-" * 92)
    for name, t in [("chalk-9 (crowd)", r["chalk"]),
                    ("naive max-ratio", r["naive_trap"]),
                    ("optimized (EV-max)", r["optimized"])]:
        print(f"{name:<24}{t['pwin']*100:>8.3f}%{t['co_winners']:>11,.0f}"
              f"{t['payout']:>11,.0f}{t['ev']:>+9.3f}{t['roi']*100:>+7.0f}%{t['value_product']:>9.1f}")
    print(f"  (winner floor = {r['winner_floor']:.0f}: caps payout at pool/floor — encodes 'can't be")
    print("   sole winner in a market this big'; tames the naive longshot trap. Calibrate to data.)")

    opt = r["optimized"]; pk = r["picks_opt"]
    print(f"\noptimized ticket picks (match#:outcome): "
          + " ".join(f"{i}:{CODE[pk[i]]}" for i in opt["idx9"]))
    print(f"  skipped (hardest 5): {sorted(set(range(14)) - set(opt['idx9']))}")
    print("\nReading: chalk (copy the crowd) is deeply −EV — in pari-mutuel the popular")
    print("combo splits the pool among everyone after the ~35% take. The edge is the")
    print("contrarian tilt: pick outcomes the crowd UNDER-bets so few co-win → big payout.")
    print("任九 makes this monetizable (drop the 5 hardest → P(win) not vanishing; payout")
    print("uncapped). The optimized ticket's SIGN hinges on TWO things, both needing real data:")
    print("  1. how biased the crowd actually is (needs real 选择比例, not this demo's skew)")
    print(f"  2. the winner floor (here {r['winner_floor']:.0f}) — caps payout at pool/floor; if the true")
    print("     floor is higher, payout shrinks and EV can flip negative. Calibrate to first_n.")
    print("\nLIVE USE: feed market-anchored p_true + real 选择比例 for the actual 14-match card;")
    print("fire ONLY when optimized EV > 0 after calibrating the winner floor to observed first_n.")


if __name__ == "__main__":
    main()
