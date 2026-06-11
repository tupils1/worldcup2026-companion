"""Injury-lag scanner — reverse-engineer the λ-multiplier the MARKET implies for
each active injury, and compare it to the LLM/MC multiplier to find genuine lag.

The idea (closes the loop on the LLM-news → conditional-MC workflow):

    1. The LLM assigns an injury a λ-multiplier `m_llm` (e.g. Kudus out → GHA att×0.72).
    2. Conditional MC turns that into a probability move for a market outcome:
         baseline P0  →  conditional Pc   (e.g. GHA "to qualify" 59% → 48%).
    3. The MARKET currently sits at Pm. Where Pm falls between P0 and Pc tells us
       how much of the injury the market has ALREADY priced:
         frac = (P0 - Pm) / (P0 - Pc)
           frac ≈ 0  → market still at the no-injury baseline (hasn't reacted) → LAG
           frac ≈ 1  → market fully agrees with our injury estimate (priced)
           frac > 1  → market priced MORE than our estimate (knows something / our mult too mild)
    4. Reverse-implied multiplier:  m_mkt = 1 + frac · (m_llm − 1).
       Compare m_mkt vs m_llm — the gap is the (dis)agreement between our news read
       and the market's.

WHERE THE ALPHA IS — and isn't:
    On a LIQUID market (champion outright, 9 books incl. Polymarket) a low `frac`
    almost never means lag: it means the market judges the player replaceable (see
    the Neymar episode — model said −2pp, market moved ~−0.3pp). The market is right.

    On a THIN market (Pinnacle-only to_qualify / group sub-markets) a low `frac` for a
    CONFIRMED, multi-source injury to a key player is the real (rare) lag signal: the
    sharp book may simply not have re-priced the sub-market yet. Act only when the
    injury is corroborated (n_articles > 1) AND the market is thin AND frac is low.

Run:
    PYTHONPATH=src python -m worldcup.strategy.injury_lag
    PYTHONPATH=src python -m worldcup.strategy.injury_lag --n-sims 30000 --min-sev 4
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.models.dixon_coles import fit
from worldcup.simulator.conditional_mc import apply_team_multipliers
from worldcup.simulator.monte_carlo import (
    MatchSampler,
    ROUND_F,
    ROUND_QF,
    ROUND_R16,
    ROUND_R32,
    ROUND_SF,
    ROUND_WIN,
    simulate_one_tournament,
)

ROUNDS = (ROUND_R32, ROUND_R16, ROUND_QF, ROUND_SF, ROUND_F, ROUND_WIN)
TEAMS_YAML = Path(__file__).resolve().parents[3] / "configs" / "teams.yaml"

# Only injuries at or below this attack-multiplier are "material" — i.e. big enough
# that a slow market could plausibly mis-price them. Elite injuries clamped to
# 0.90/0.95 (Neymar/Messi-type "scares") are deliberately too mild to trade as lag,
# so we don't waste a conditional-MC run on them and never flag them LAG/watch.
MATERIAL_ATT = 0.88

# Markets we reconcile: (label, model_round, scope, liquidity, lag_plausibility).
# Liquidity flags how much weight to give a low-frac signal: thin markets can lag,
# liquid markets almost never do.
MARKETS = {
    "to_qualify": {"round": ROUND_R32, "label": "advance (to-qualify)", "thin": True},
    "champion":   {"round": ROUND_WIN, "label": "champion",            "thin": False},
}


# ─────────────────────────── model side ──────────────────────────────────────
def _load_groups_elo(db_path) -> tuple[dict, dict]:
    cfg = yaml.safe_load(TEAMS_YAML.read_text())
    groups = {lbl: g["teams"] for lbl, g in cfg["groups"].items()}
    conn = sqlite3.connect(str(db_path))
    elo = {r[0]: float(r[1]) for r in conn.execute(
        "SELECT team_code, value FROM team_ratings WHERE rating_type='elo'"
    )}
    conn.close()
    return groups, elo


def _simulate(params, groups, elo, n_sims: int, seed: int = 42) -> dict[str, dict]:
    """Run n_sims tournaments; return {team: {round: reach_prob}}.

    Uses a fixed seed so baseline and per-team conditional runs share common random
    numbers — the *difference* P0−Pc is then low-noise (variance largely cancels)."""
    sampler = MatchSampler(
        rho=params.rho, home_adv=params.home_advantage,
        attack=params.attack, defense=params.defense,
    )
    rng = np.random.default_rng(seed)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _ in range(n_sims):
        prog = simulate_one_tournament(sampler, groups, elo, rng)
        for team, furthest in prog.items():
            if furthest == "group":
                continue
            idx = ROUNDS.index(furthest)
            for r in ROUNDS[:idx + 1]:
                counts[team][r] += 1
    return {t: {r: counts[t][r] / n_sims for r in ROUNDS} for t in counts}


# ─────────────────────────── market side ─────────────────────────────────────
def champion_probs(conn) -> dict[str, float]:
    """De-vigged champion prob per team: proportional de-vig per book, mean across books."""
    rows = conn.execute("""
        SELECT o.bookmaker, o.selection, o.price FROM odds o
        JOIN (SELECT bookmaker, selection, MAX(captured_at) mc FROM odds
              WHERE market_scope='outright' AND market='winner'
              GROUP BY bookmaker, selection) l
          ON o.bookmaker=l.bookmaker AND o.selection=l.selection AND o.captured_at=l.mc
        WHERE o.market_scope='outright' AND o.market='winner'
    """).fetchall()
    bybook: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["price"] and r["price"] > 1.0:
            bybook[r["bookmaker"]][r["selection"]] = r["price"]
    per_team: dict[str, list[float]] = defaultdict(list)
    for _book, q in bybook.items():
        inv = {s: 1.0 / p for s, p in q.items()}
        tot = sum(inv.values())
        if tot <= 0:
            continue
        for s, v in inv.items():
            per_team[s].append(v / tot)
    return {s: float(np.mean(v)) for s, v in per_team.items()}


def to_qualify_probs(conn, vig: float = 1.025) -> dict[str, float]:
    """Pinnacle to-qualify (independent Yes per team) → fair prob via small vig haircut.
    Only the Yes price is published, so we can't normalise a set; the documented
    Pinnacle sub-market margin (~2.5%) is applied as a flat haircut on 1/price."""
    rows = conn.execute("""
        SELECT o.selection, o.price FROM odds o
        JOIN (SELECT selection, MAX(captured_at) mc FROM odds
              WHERE market_scope='to_qualify' GROUP BY selection) l
          ON o.selection=l.selection AND o.captured_at=l.mc
        WHERE o.market_scope='to_qualify'
    """).fetchall()
    return {r["selection"]: (1.0 / r["price"]) / vig
            for r in rows if r["price"] and r["price"] > 1.0}


# ─────────────────────────── injuries ────────────────────────────────────────
def load_active_injuries(conn, min_sev: int = 3) -> list[dict]:
    """One scenario per team: the strongest (lowest att-mult) LLM-scored injury,
    plus a corroboration count (how many articles flag this team)."""
    rows = conn.execute("""
        SELECT llm_team team, llm_attack_mult att, llm_defense_mult deff,
               llm_severity sev, llm_player player, llm_player_zh player_zh,
               llm_confidence conf, llm_impact_type itype
        FROM news_alerts
        WHERE llm_team IS NOT NULL AND llm_attack_mult IS NOT NULL
          AND llm_attack_mult < 1.0 AND llm_severity >= ?
          AND llm_impact_type IN ('injury','suspension','squad_change')
    """, (min_sev,)).fetchall()
    by_team: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_team[r["team"]].append(dict(r))
    out = []
    for team, items in by_team.items():
        strongest = min(items, key=lambda x: x["att"] if x["att"] is not None else 1.0)
        strongest["n_articles"] = len(items)
        out.append(strongest)
    out.sort(key=lambda x: x["att"])  # biggest claimed impact first
    return out


# ─────────────────────────── reconciliation ──────────────────────────────────
def reconcile(injuries, baseline, conditional, markets, min_model_move=0.005):
    """Build per-(team,market) reconciliation rows."""
    out = []
    for inj in injuries:
        team = inj["team"]
        m_llm = inj["att"]
        cond = conditional.get(team, {})
        base = baseline.get(team, {})
        for mkt, spec in MARKETS.items():
            rnd = spec["round"]
            p0 = base.get(rnd, 0.0)
            pc = cond.get(rnd, 0.0)
            pm = markets.get(mkt, {}).get(team)
            if pm is None:
                continue
            move = p0 - pc                      # model's predicted drop from the injury
            if move < min_model_move:           # injury barely moves this outcome → skip
                continue
            frac = (p0 - pm) / move             # share of the drop the market has taken
            m_mkt = 1.0 + frac * (m_llm - 1.0)  # multiplier the market price implies
            out.append({
                "team": team, "player": inj.get("player_zh") or inj["player"], "n_articles": inj["n_articles"],
                "sev": inj["sev"], "market": mkt, "label": spec["label"], "thin": spec["thin"],
                "m_llm": m_llm, "m_mkt": m_mkt, "frac": frac,
                "p0": p0, "pc": pc, "pm": pm, "move": move,
            })
    return out


VERDICT_RANK = {"LAG?": 0, "watch": 1, "priced": 2, "liquid-unmoved": 3,
                "mkt-bearish": 4, "over-priced": 5, "model-bias": 6, "negligible": 7}


def classify(row) -> tuple[str, str]:
    """Return (verdict, action).

    CRITICAL: the model has a known baseline bias vs the sharp market (it can't beat
    it). So a raw frac is only meaningful when model≈market at baseline. We gate on
    the market's position relative to the model's *pre-injury* level P0:
      - market ABOVE P0  → model underrates the team; bias, NOT lag (e.g. GHA).
      - market ≈ P0       → model≈market here, so a no-move market = genuinely unpriced.
      - market in (Pc,P0) → injury partly/fully priced.
      - market below Pc   → market more bearish than us.
    Only 'market ≈ P0' on a thin, corroborated injury is the real lag signal."""
    p0, pc, pm = row["p0"], row["pc"], row["pm"]
    move = p0 - pc                       # model's predicted drop (>0 by construction)
    tol = max(0.02, 0.25 * move)         # "market ≈ model baseline" tolerance band
    thin, corr = row["thin"], row["n_articles"] > 1
    team, label = row["team"], row["label"]

    if pm > p0 + tol:
        return "model-bias", f"market rates {team} ABOVE model's pre-injury level — model underrates {team}, NOT lag"
    if pm < pc - tol:
        return "mkt-bearish", "market below model's injured level — market more bearish than our estimate"

    frac = (p0 - pm) / move
    if frac < 0.40:
        # Market is near the model's pre-injury baseline AND we've ruled out bias
        # (pm not above p0), so model≈market here → the injury looks genuinely unpriced.
        if row["m_llm"] > MATERIAL_ATT:
            return "negligible", "injury too mild (clamped) to move this market — no tradeable signal"
        if thin and corr:
            return "LAG?", f"market still ≈ pre-injury level — consider LAY {team} {label}"
        if thin:
            return "watch", "thin market unmoved but single-source — corroborate before acting"
        return "liquid-unmoved", "liquid market unmoved — judges the player replaceable; trust market"
    if frac <= 1.20:
        return "priced", "market ≈ model injury view — no edge"
    return "over-priced", "market priced more drop than our estimate — it may know more"


# ─────────────────────────── main ────────────────────────────────────────────
def run(n_sims: int = 20000, min_sev: int = 3, db_path=DEFAULT_DB_PATH) -> dict:
    conn = get_conn(db_path)
    injuries = load_active_injuries(conn, min_sev=min_sev)
    champ = champion_probs(conn)
    qual = to_qualify_probs(conn)
    conn.close()
    markets = {"champion": champ, "to_qualify": qual}

    if not injuries:
        return {"injuries": [], "rows": [], "markets": markets}

    # Only sizable injuries can plausibly produce a tradeable lag; elite "scares"
    # clamped to 0.90/0.95 are too mild — skip their conditional MC (speed) and list
    # them as negligible so the pipeline run stays light.
    material = [i for i in injuries if i["att"] <= MATERIAL_ATT]
    skipped = [i for i in injuries if i["att"] > MATERIAL_ATT]

    if not material:
        return {"injuries": injuries, "rows": [], "markets": markets, "skipped": skipped}

    groups, elo = _load_groups_elo(db_path)
    params = fit(db_path=db_path, since="2014-01-01", elo_prior_strength=0.5)
    baseline = _simulate(params, groups, elo, n_sims)

    conditional = {}
    for inj in material:
        team = inj["team"]
        cond_params = apply_team_multipliers(
            params,
            attack_mult={team: inj["att"]},
            defense_mult={team: inj["deff"]} if inj["deff"] and inj["deff"] < 1.0 else None,
        )
        sim = _simulate(cond_params, groups, elo, n_sims)
        conditional[team] = sim.get(team, {})

    rows = reconcile(material, baseline, conditional, markets)
    rows.sort(key=lambda r: (not r["thin"], r["frac"]))  # thin + low-frac first
    return {"injuries": injuries, "rows": rows, "markets": markets,
            "skipped": skipped, "baseline": baseline, "conditional": conditional}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sims", type=int, default=20000)
    ap.add_argument("--min-sev", type=int, default=3,
                    help="Min LLM severity to consider (3=rotation/minor, 4-5=key starter)")
    args = ap.parse_args()

    print("=" * 92)
    print("INJURY-LAG SCAN — market-implied multiplier vs LLM/MC multiplier")
    print("=" * 92)
    res = run(n_sims=args.n_sims, min_sev=args.min_sev)
    injuries, rows = res["injuries"], res["rows"]

    if not injuries:
        print("\n  No active LLM-scored injuries at this severity. (Run llm_news_scorer first.)")
        return
    print(f"\nActive injuries (sev≥{args.min_sev}): "
          + ", ".join(f"{i['team']}({(i.get('player_zh') or i['player']) or '?'} att×{i['att']:.2f}, n={i['n_articles']})"
                      for i in injuries))
    skipped = res.get("skipped", [])
    if skipped:
        print(f"Skipped {len(skipped)} mild/clamped injury(s) (att>{MATERIAL_ATT}, no tradeable lag): "
              + ", ".join(f"{i['team']}({(i.get('player_zh') or i['player']) or '?'})" for i in skipped))
    if not rows:
        print("\n  No material injuries move a market we have prices for. (Nothing to reconcile.)")
        return

    decorated = []
    for r in rows:
        verdict, action = classify(r)
        decorated.append((VERDICT_RANK.get(verdict, 9), r, verdict, action))
    decorated.sort(key=lambda x: (x[0], x[1]["frac"]))

    print(f"\n{'team':<5} {'player':<14} {'market':<20} {'m_llm':>6} {'m_mkt':>6} "
          f"{'model(base->inj)':>16} {'mkt':>6} {'frac':>6} {'n':>2}  verdict / action")
    print("-" * 124)
    for _, r, verdict, action in decorated:
        model_str = f"{r['p0']*100:.1f}->{r['pc']*100:.1f}%"
        thin_tag = "·thin" if r["thin"] else ""
        print(f"{r['team']:<5} {(r['player'] or '?')[:14]:<14} {r['label']+thin_tag:<20} "
              f"{r['m_llm']:6.2f} {r['m_mkt']:6.2f} {model_str:>16} {r['pm']*100:5.1f}% "
              f"{r['frac']:6.2f} {r['n_articles']:>2}  {verdict}: {action}")

    print("\nReading:")
    print("  m_llm = (clamped) multiplier we fed · m_mkt = multiplier the market price implies")
    print("  model = baseline→injured prob · frac = share of that drop the market has taken")
    print("  LAG? (thin+corroborated+frac<0.4) = the rare real signal. On liquid markets,")
    print("  low frac = market judges the player replaceable — trust the market (Neymar lesson).")


if __name__ == "__main__":
    main()
