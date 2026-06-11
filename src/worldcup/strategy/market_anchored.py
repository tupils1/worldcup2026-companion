"""Market-anchored forecast — use the de-vigged sharp market as the baseline.

Our Dixon-Coles model is RESULTS-based and structurally can't beat the sharp
market (which prices talent / squad value we don't see). That's why it underrates
brand teams (England missing from its champion top-10) and overrates recent
over-performers (Morocco). Friendly-down-weighting (DC competition_weighting) helps
England a bit, but nothing results-based fixes Morocco — only the market does.

So for any number we'd actually act on, we ANCHOR to the de-vigged market and
deviate ONLY where we have private info the market is slow on (injuries → a
conditional-MC tilt). With no private info, the anchored forecast == the de-vigged
market (we're not pretending to beat it); the model's job is the tilt + structure.

    anchored(team) = market_devig(team) · tilt(team)        (then renormalised)
    tilt(team)     = MC_conditional(team) / MC_baseline(team)   for material injuries
                     1.0                                          otherwise

Run:
    PYTHONPATH=src python -m worldcup.strategy.market_anchored            # champion, no tilt
    PYTHONPATH=src python -m worldcup.strategy.market_anchored --tilt --n-sims 15000
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.models.dixon_coles import fit
from worldcup.simulator.conditional_mc import apply_team_multipliers
from worldcup.simulator.monte_carlo import (
    MatchSampler,
    ROUND_R32,
    ROUND_WIN,
    _simulate_group,
)
from worldcup.strategy.injury_lag import (
    MATERIAL_ATT,
    _load_groups_elo,
    _simulate,
    champion_probs,
    load_active_injuries,
    to_qualify_probs,
)


def _renorm(probs: dict[str, float]) -> dict[str, float]:
    tot = sum(probs.values())
    return {k: v / tot for k, v in probs.items()} if tot > 0 else probs


def injury_tilt(n_sims: int = 15000, min_sev: int = 4, db_path=DEFAULT_DB_PATH
                ) -> tuple[dict[str, float], dict[str, float], list]:
    """Return ({team: champion_tilt}, {team: advance_tilt}, material_injuries).

    Tilt = conditional / baseline MC probability for each materially-injured team;
    >1 raises, <1 lowers. Teams without a material injury get no entry (→ tilt 1)."""
    conn = get_conn(db_path)
    injuries = load_active_injuries(conn, min_sev=min_sev)
    conn.close()
    material = [i for i in injuries if i["att"] <= MATERIAL_ATT]
    champ_tilt: dict[str, float] = {}
    adv_tilt: dict[str, float] = {}
    if not material:
        return champ_tilt, adv_tilt, material

    groups, elo = _load_groups_elo(db_path)
    params = fit(db_path=db_path, since="2014-01-01", elo_prior_strength=0.5)
    base = _simulate(params, groups, elo, n_sims)
    for inj in material:
        t = inj["team"]
        cp = apply_team_multipliers(
            params, attack_mult={t: inj["att"]},
            defense_mult={t: inj["deff"]} if inj["deff"] and inj["deff"] < 1.0 else None,
        )
        cond = _simulate(cp, groups, elo, n_sims).get(t, {})
        b = base.get(t, {})
        if b.get(ROUND_WIN, 0) > 1e-4:
            champ_tilt[t] = cond.get(ROUND_WIN, 0) / b[ROUND_WIN]
        if b.get(ROUND_R32, 0) > 1e-4:
            adv_tilt[t] = cond.get(ROUND_R32, 0) / b[ROUND_R32]
    return champ_tilt, adv_tilt, material


def market_anchored(market: dict[str, float], tilt: dict[str, float] | None = None
                    ) -> dict[str, float]:
    """De-vigged market baseline, tilted by private-info factors, renormalised."""
    tilt = tilt or {}
    return _renorm({t: p * tilt.get(t, 1.0) for t, p in market.items()})


def model_probs(n_sims: int, db_path=DEFAULT_DB_PATH) -> tuple[dict, dict]:
    """Baseline-model champion + advance probs per team (no injuries)."""
    groups, elo = _load_groups_elo(db_path)
    params = fit(db_path=db_path, since="2014-01-01", elo_prior_strength=0.5)
    sim = _simulate(params, groups, elo, n_sims)
    champ = {t: sim[t].get(ROUND_WIN, 0.0) for t in sim}
    adv = {t: sim[t].get(ROUND_R32, 0.0) for t in sim}
    return champ, adv


def devig_group_winner(conn, groups: dict[str, list[str]]) -> dict[str, float]:
    """Pinnacle group-winner odds → fair prob, de-vigged WITHIN each group (the 4
    members are a complete, mutually-exclusive set, so normalise 1/price to sum 1)."""
    rows = conn.execute("""
        SELECT o.selection, o.price FROM odds o
        JOIN (SELECT selection, MAX(captured_at) mc FROM odds
              WHERE market_scope='group_winner' GROUP BY selection) l
          ON o.selection=l.selection AND o.captured_at=l.mc
        WHERE o.market_scope='group_winner'
    """).fetchall()
    price = {r["selection"]: r["price"] for r in rows if r["price"] and r["price"] > 1.0}
    out: dict[str, float] = {}
    for _letter, teams in groups.items():
        inv = {t: 1.0 / price[t] for t in teams if t in price}
        tot = sum(inv.values())
        if tot > 0:
            for t, v in inv.items():
                out[t] = v / tot
    return out


def model_submarket_probs(params, groups, elo, n_sims: int, seed: int = 42
                          ) -> tuple[dict[str, float], dict[str, float]]:
    """Group-stage-only sim → ({team: P(win group)}, {team: P(advance)}).

    Group-winner and advancement are decided entirely in the group stage (advance =
    top-2 per group + the 8 best 3rd-placed teams across groups), so we skip the
    knockout bracket — much cheaper than a full tournament sim."""
    sampler = MatchSampler(rho=params.rho, home_adv=params.home_advantage,
                           attack=params.attack, defense=params.defense)
    rng = np.random.default_rng(seed)
    gw: dict[str, int] = defaultdict(int)
    adv: dict[str, int] = defaultdict(int)
    for _ in range(n_sims):
        thirds = []
        advanced: list[str] = []
        for letter in sorted(groups):
            ranked = _simulate_group(groups[letter], elo, sampler, rng)
            gw[ranked[0].team] += 1
            advanced.append(ranked[0].team)
            advanced.append(ranked[1].team)
            thirds.append(ranked[2])
        best8 = sorted(
            thirds, key=lambda r: (-r.points, -r.gd, -r.gf, -elo.get(r.team, 1500.0))
        )[:8]
        advanced.extend(r.team for r in best8)
        for t in advanced:
            adv[t] += 1
    return ({t: c / n_sims for t, c in gw.items()},
            {t: c / n_sims for t, c in adv.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sims", type=int, default=20000)
    ap.add_argument("--tilt", action="store_true",
                    help="Apply injury conditional-MC tilt to the market baseline")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--submarkets", action="store_true",
                    help="Also anchor the thin Pinnacle sub-markets (group-winner + to-qualify)")
    args = ap.parse_args()

    conn = get_conn(DEFAULT_DB_PATH)
    mkt_champ = _renorm(champion_probs(conn))
    groups, elo = _load_groups_elo(DEFAULT_DB_PATH)
    mkt_gw = devig_group_winner(conn, groups) if args.submarkets else {}
    mkt_adv = to_qualify_probs(conn) if args.submarkets else {}
    conn.close()

    print("Fitting model + running baseline MC ...")
    params = fit(db_path=DEFAULT_DB_PATH, since="2014-01-01", elo_prior_strength=0.5)
    sim = _simulate(params, groups, elo, args.n_sims)
    m_champ = {t: sim[t].get(ROUND_WIN, 0.0) for t in sim}

    champ_tilt = {}
    material = []
    if args.tilt:
        print("Computing injury tilt (conditional MC) ...")
        champ_tilt, _adv_tilt, material = injury_tilt(n_sims=max(10000, args.n_sims // 2))

    anchored = market_anchored(mkt_champ, champ_tilt)

    print("\n" + "=" * 78)
    print("MARKET-ANCHORED CHAMPION FORECAST  (model vs de-vigged market)")
    print("=" * 78)
    if material:
        print("Injury tilt applied to: "
              + ", ".join(f"{i['team']}({i['player'] or '?'}×{i['att']:.2f})" for i in material))
    rows = sorted(anchored.items(), key=lambda kv: -kv[1])[:args.top]
    print(f"\n{'team':<5} {'MODEL':>8} {'MARKET':>8} {'ANCHORED':>9}  {'model−mkt':>10}  bias flag")
    print("-" * 70)
    for t, ap_ in rows:
        mm = m_champ.get(t, 0.0)
        mk = mkt_champ.get(t, 0.0)
        gap = mm - mk
        flag = ""
        if gap <= -0.02:
            flag = "← model UNDER-rates (brand team)"
        elif gap >= 0.02:
            flag = "← model OVER-rates (over-performer)"
        print(f"{t:<5} {mm*100:7.2f}% {mk*100:7.2f}% {ap_*100:8.2f}%  {gap*100:+9.2f}%  {flag}")

    # Spotlight the two known biases
    print("\nKnown-bias spotlight:")
    for t in ("ENG", "MAR", "CRO"):
        if t in mkt_champ:
            print(f"  {t}: model {m_champ.get(t,0)*100:5.2f}%  →  market {mkt_champ[t]*100:5.2f}%  "
                  f"(anchored {anchored.get(t,0)*100:5.2f}%)")
    print("\nAnchored = de-vigged market (× injury tilt). Use THIS for sizing, not the")
    print("raw model — the model's absolute champion probs carry the ENG/MAR bias.")

    # ── Sub-markets: group-winner + to-qualify (thin Pinnacle markets) ──
    if args.submarkets:
        m_gw, m_adv = model_submarket_probs(params, groups, elo, args.n_sims)

        print("\n" + "=" * 78)
        print("②  GROUP-WINNER  (model vs de-vigged Pinnacle, per group)")
        print("=" * 78)
        for letter in sorted(groups):
            print(f"\nGroup {letter}:   {'team':<5}{'MODEL':>8}{'MARKET':>8}{'Δ':>8}")
            for t in sorted(groups[letter], key=lambda x: -mkt_gw.get(x, 0.0)):
                mm, mk = m_gw.get(t, 0.0), mkt_gw.get(t, 0.0)
                tag = ""
                if mk > 0 and mm - mk <= -0.06:
                    tag = "← model under-rates"
                elif mk > 0 and mm - mk >= 0.06:
                    tag = "← model over-rates"
                print(f"          {t:<5}{mm*100:7.1f}%{mk*100:7.1f}%{(mm-mk)*100:+7.1f}%  {tag}")

        print("\n" + "=" * 78)
        print("③  ADVANCE / TO-QUALIFY  (model vs de-vigged Pinnacle — thin market, biggest gaps)")
        print("=" * 78)
        gaps = sorted(
            ((t, m_adv.get(t, 0.0), mkt_adv[t]) for t in mkt_adv if t in m_adv),
            key=lambda x: -abs(x[1] - x[2]),
        )
        print(f"{'team':<5}{'MODEL':>8}{'MARKET':>8}{'Δ':>8}  note")
        print("-" * 74)
        for t, mm, mk in gaps[:16]:
            note = ""
            if mm - mk >= 0.06:
                note = "model higher — thin-mkt edge? (rule out ENG/MAR-type bias)"
            elif mm - mk <= -0.06:
                note = "market higher — model under-rates"
            print(f"{t:<5}{mm*100:7.1f}%{mk*100:7.1f}%{(mm-mk)*100:+7.1f}%  {note}")
        print("\nThin Pinnacle sub-markets are less efficient than champion outright — a")
        print("model>market gap MIGHT be real edge, but the ENG-under / MAR-over bias still")
        print("applies here too, so corroborate before sizing. Anchored baseline = de-vig market.")


if __name__ == "__main__":
    main()
