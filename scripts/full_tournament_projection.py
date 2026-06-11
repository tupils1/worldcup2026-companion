#!/usr/bin/env python
"""Full tournament projection — end-to-end WC 2026 推演报告.

Runs N detailed Monte Carlo simulations, tracking not just "how far each team
got" but the FULL state of every tournament: group standings, who advanced,
the knockout bracket, and the champion. Then aggregates into a readable
projection from group stage all the way to the final.

Sections:
    1. Group-stage projection: per group, each team's P(win group / advance / out)
    2. Best-8-third-place projection
    3. Knockout path: modal advancer at each round
    4. Final four / final / champion projection
    5. Most likely final matchup

Optionally blends with market (Polymarket) for a calibrated champion view.

Run:
    PYTHONPATH=src python scripts/full_tournament_projection.py --n-sims 50000
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.models.dixon_coles import fit
from worldcup.simulator.monte_carlo import (
    GROUP_PAIRINGS,
    HOSTS,
    MatchSampler,
    _make_r32_bracket,
    _simulate_group,
    _simulate_knockout,
)

ROOT = Path(__file__).resolve().parents[1]


def simulate_detailed(sampler, groups, elo, rng) -> dict:
    """One full tournament. Returns full structured state."""
    group_rankings = {}   # letter -> [1st, 2nd, 3rd, 4th] codes
    winners, runners_up, thirds = [], [], []
    for letter in sorted(groups):
        ranked = _simulate_group(groups[letter], elo, sampler, rng)
        group_rankings[letter] = [r.team for r in ranked]
        winners.append(ranked[0].team)
        runners_up.append(ranked[1].team)
        thirds.append(ranked[2])

    best_thirds_objs = sorted(
        thirds, key=lambda r: (-r.points, -r.gd, -r.gf, -elo.get(r.team, 1500.0))
    )[:8]
    best_thirds = [r.team for r in best_thirds_objs]

    bracket = _make_r32_bracket(winners, runners_up, best_thirds)
    r16 = [_simulate_knockout(h, a, sampler, rng) for h, a in bracket]
    qf = [_simulate_knockout(r16[i], r16[i + 1], sampler, rng) for i in range(0, 16, 2)]
    sf = [_simulate_knockout(qf[i], qf[i + 1], sampler, rng) for i in range(0, 8, 2)]
    finalists = [_simulate_knockout(sf[i], sf[i + 1], sampler, rng) for i in range(0, 4, 2)]
    champion = _simulate_knockout(finalists[0], finalists[1], sampler, rng)

    return {
        "group_rankings": group_rankings,
        "winners": winners,
        "runners_up": runners_up,
        "best_thirds": best_thirds,
        "r16": r16, "qf": qf, "sf": sf,
        "finalists": finalists,
        "champion": champion,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=50000)
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Fitting Dixon-Coles (prior={args.prior}) ...")
    params = fit(elo_prior_strength=args.prior)
    print(f"  {params.n_matches} matches, log-lik {params.fit_loglik:.0f}\n")

    cfg = yaml.safe_load((ROOT / "configs" / "teams.yaml").read_text())
    groups = {l: g["teams"] for l, g in cfg["groups"].items()}
    team_names = {c: m["name"] for c, m in cfg["teams"].items()}

    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    elo = {r[0]: float(r[1]) for r in conn.execute(
        "SELECT team_code, value FROM team_ratings WHERE rating_type='elo'")}
    conn.close()

    sampler = MatchSampler(rho=params.rho, home_adv=params.home_advantage,
                           attack=params.attack, defense=params.defense)
    rng = np.random.default_rng(args.seed)

    # Aggregators
    pos_counts = {l: defaultdict(lambda: [0, 0, 0, 0]) for l in groups}  # team -> [p1,p2,p3,p4]
    advance_counts = defaultdict(int)
    group_win_counts = {l: defaultdict(int) for l in groups}
    best_third_counts = defaultdict(int)
    champion_counts = defaultdict(int)
    finalist_counts = defaultdict(int)
    sf_counts = defaultdict(int)
    qf_counts = defaultdict(int)
    final_matchup_counts = Counter()

    print(f"Running {args.n_sims:,} detailed simulations ...")
    import time
    t0 = time.time()
    for sim in range(args.n_sims):
        res = simulate_detailed(sampler, groups, elo, rng)
        for letter, ranking in res["group_rankings"].items():
            for pos, team in enumerate(ranking):
                pos_counts[letter][team][pos] += 1
            group_win_counts[letter][ranking[0]] += 1
        advanced = set(res["winners"] + res["runners_up"] + res["best_thirds"])
        for t in advanced:
            advance_counts[t] += 1
        for t in res["best_thirds"]:
            best_third_counts[t] += 1
        for t in res["qf"]:
            qf_counts[t] += 1
        for t in res["sf"]:
            sf_counts[t] += 1
        for t in res["finalists"]:
            finalist_counts[t] += 1
        champion_counts[res["champion"]] += 1
        final_matchup_counts[tuple(sorted(res["finalists"]))] += 1
        if (sim + 1) % max(1, args.n_sims // 5) == 0:
            print(f"  {sim+1:,}/{args.n_sims:,}  ({time.time()-t0:.0f}s)")

    N = args.n_sims

    # ─── SECTION 1: Group stage ───
    print("\n" + "=" * 78)
    print("①  小组赛推演 (GROUP STAGE PROJECTION)")
    print("=" * 78)
    for letter in sorted(groups):
        print(f"\nGroup {letter}:")
        print(f"  {'team':<5} {'win grp':>8} {'1st':>6} {'2nd':>6} {'3rd':>6} {'4th':>6} {'ADVANCE':>8}")
        # Sort teams by advance probability
        team_adv = []
        for team in groups[letter]:
            p1, p2, p3, p4 = [c / N for c in pos_counts[letter][team]]
            adv = advance_counts[team] / N
            team_adv.append((team, p1, p2, p3, p4, adv))
        team_adv.sort(key=lambda x: -x[5])
        for team, p1, p2, p3, p4, adv in team_adv:
            star = " ⭐" if adv > 0.5 else ""
            print(f"  {team:<5} {p1*100:7.1f}% {p1*100:5.1f}% {p2*100:5.1f}% "
                  f"{p3*100:5.1f}% {p4*100:5.1f}% {adv*100:7.1f}%{star}")

    # ─── SECTION 2: Best thirds ───
    print("\n" + "=" * 78)
    print("②  最佳第三名出线竞争 (BEST-8 THIRD-PLACE)")
    print("=" * 78)
    third_ranked = sorted(best_third_counts.items(), key=lambda x: -x[1])
    print(f"  Teams most often qualifying as a best-third:")
    print(f"  {'team':<5} {'as best-3rd %':>14}  {'group':>6}")
    code_to_group = {c: l for l, g in groups.items() for c in g}
    for team, cnt in third_ranked[:12]:
        print(f"  {team:<5} {cnt/N*100:13.1f}%  {code_to_group.get(team, '?'):>6}")

    # ─── SECTION 3: Deep run probabilities ───
    print("\n" + "=" * 78)
    print("③  淘汰赛深度 (KNOCKOUT DEPTH)")
    print("=" * 78)
    all_teams = sorted(champion_counts.keys() | qf_counts.keys() | sf_counts.keys(),
                       key=lambda t: -champion_counts[t])
    contenders = sorted(set(groups[l][i] for l in groups for i in range(4)),
                        key=lambda t: -(champion_counts[t]))
    print(f"  {'team':<5} {'QF':>7} {'SF':>7} {'Final':>7} {'CHAMP':>7}")
    for t in contenders[:16]:
        print(f"  {t:<5} {qf_counts[t]/N*100:6.1f}% {sf_counts[t]/N*100:6.1f}% "
              f"{finalist_counts[t]/N*100:6.1f}% {champion_counts[t]/N*100:6.1f}%")

    # ─── SECTION 4: Champion + final ───
    print("\n" + "=" * 78)
    print("④  冠军 & 决赛预测 (CHAMPION & FINAL)")
    print("=" * 78)
    champ_ranked = sorted(champion_counts.items(), key=lambda x: -x[1])
    print(f"\n  最可能冠军 TOP 10:")
    for i, (team, cnt) in enumerate(champ_ranked[:10], 1):
        print(f"    {i:2d}. {team} ({team_names.get(team,'')[:18]:<18}) {cnt/N*100:5.2f}%")

    print(f"\n  最可能决赛对阵 TOP 8:")
    for i, (matchup, cnt) in enumerate(final_matchup_counts.most_common(8), 1):
        a, b = matchup
        print(f"    {i}. {a} vs {b}  —  {cnt/N*100:.2f}% of simulations")

    # ─── SECTION 5: Modal bracket (single most representative run) ───
    print("\n" + "=" * 78)
    print("⑤  模态推演 (MOST-LIKELY SINGLE PATH — illustrative)")
    print("=" * 78)
    print("  基于各阶段最高概率球队的代表性路径 (非严格联合最优):")
    # Group winners (modal)
    modal_winners = {l: max(group_win_counts[l].items(), key=lambda x: x[1])[0]
                     for l in sorted(groups)}
    print(f"\n  小组头名预测:")
    line = "    "
    for l in sorted(groups):
        line += f"{l}:{modal_winners[l]}  "
        if l in ('F', 'L'):
            print(line); line = "    "
    if line.strip():
        print(line)

    champ = champ_ranked[0][0]
    runner = final_matchup_counts.most_common(1)[0][0]
    print(f"\n  📌 最可能冠军: {champ} ({team_names.get(champ,'')}) — {champ_ranked[0][1]/N*100:.1f}%")
    top_final = final_matchup_counts.most_common(1)[0]
    print(f"  📌 最可能决赛: {top_final[0][0]} vs {top_final[0][1]} ({top_final[1]/N*100:.1f}%)")

    print(f"\n  ⏱  {N:,} sims in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
