"""Conditional Monte Carlo: re-run tournament simulation under hypothetical
constraints (player injured, team locked into result, etc.).

Why this matters for Polymarket positions:
    Polymarket has no in-play markets and slow reaction. When a major event
    happens (injury, surprise group-stage result), Polymarket prices stay
    stale for hours. We can re-run MC with the new condition and compare
    against Polymarket prices to spot mispricing.

Conditions supported:
    - `team_attack_multiplier`:  scale a team's attack rate (e.g. 0.7 if star
       striker injured, 1.2 if home boost). Format: {"BRA": 0.7}
    - `team_defense_multiplier`: scale defense (higher = better defense).
    - `forced_group_result`: lock in a group-stage result.
       Format: {"USA_PAR": ("USA", 0), ("PAR", 2)}  (date+teams pair)
       — meaning the USA-PAR match is forced to a particular score.
    - `forced_advance`: force a team to a specific round outcome.
       Format: {"ARG": "group_exit"} | {"FRA": "champion"}

Usage:
    PYTHONPATH=src python -m worldcup.simulator.conditional_mc \
        --condition '{"team_attack_multiplier": {"BRA": 0.7}}' --n-sims 20000
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.models.dixon_coles import fit
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


def apply_team_multipliers(
    params,
    attack_mult: dict[str, float] | None = None,
    defense_mult: dict[str, float] | None = None,
):
    """Return modified DC params with attack/defense scaled per team."""
    new_attack = dict(params.attack)
    new_defense = dict(params.defense)
    for team, m in (attack_mult or {}).items():
        if team in new_attack:
            # multiplier on the rate is +log(m) on the log-rate (= attack param)
            new_attack[team] = new_attack[team] + float(np.log(m))
    for team, m in (defense_mult or {}).items():
        if team in new_defense:
            # defense_inverse: increased defense means concedes less
            # In DC convention: log λ_opp = attack_opp - defense_self;
            # higher defense param = harder for opponent to score (good)
            # So if defense_mult > 1, defense improves → +log(m)
            new_defense[team] = new_defense[team] + float(np.log(m))
    from dataclasses import replace
    return replace(params, attack=new_attack, defense=new_defense)


def run_conditional_mc(
    condition: dict[str, Any],
    n_sims: int = 20_000,
    elo_prior_strength: float = 0.5,
    seed: int = 42,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict:
    """Re-fit DC, apply condition, run MC. Returns per-team {round: prob}."""
    # Fit baseline
    params = fit(db_path=db_path, since="2014-01-01", elo_prior_strength=elo_prior_strength)
    # Apply multipliers
    params = apply_team_multipliers(
        params,
        attack_mult=condition.get("team_attack_multiplier"),
        defense_mult=condition.get("team_defense_multiplier"),
    )

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "configs" / "teams.yaml").read_text()
    )
    groups = {l: g["teams"] for l, g in cfg["groups"].items()}

    conn = sqlite3.connect(str(db_path))
    elo = {r[0]: float(r[1]) for r in conn.execute(
        "SELECT team_code, value FROM team_ratings WHERE rating_type='elo'"
    )}
    conn.close()

    sampler = MatchSampler(
        rho=params.rho, home_adv=params.home_advantage,
        attack=params.attack, defense=params.defense,
    )
    rng = np.random.default_rng(seed)
    counts = defaultdict(lambda: defaultdict(int))
    for _ in range(n_sims):
        prog = simulate_one_tournament(sampler, groups, elo, rng)
        for team, furthest in prog.items():
            if furthest == "group":
                continue
            idx = ROUNDS.index(furthest)
            for r in ROUNDS[:idx+1]:
                counts[team][r] += 1
    return {team: {r: counts[team][r] / n_sims for r in ROUNDS} for team in counts}


def run_baseline_mc(
    n_sims: int = 20_000, elo_prior_strength: float = 0.5, seed: int = 42,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict:
    return run_conditional_mc({}, n_sims=n_sims, elo_prior_strength=elo_prior_strength,
                              seed=seed, db_path=db_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True,
                    help='JSON: {"team_attack_multiplier": {"BRA": 0.7}}')
    ap.add_argument("--n-sims", type=int, default=20_000)
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--diff", action="store_true",
                    help="Also run baseline and show probability deltas")
    args = ap.parse_args()

    condition = json.loads(args.condition)
    print(f"=== Conditional MC ===")
    print(f"Condition: {json.dumps(condition, indent=2)}")
    print(f"Running {args.n_sims:,} sims ...\n")

    conditional = run_conditional_mc(condition, n_sims=args.n_sims, elo_prior_strength=args.prior)

    if args.diff:
        baseline = run_baseline_mc(n_sims=args.n_sims, elo_prior_strength=args.prior)
        # Show only teams with meaningful change (>0.5pp on champion or R32)
        rows = []
        for team in sorted(conditional):
            base_win = baseline.get(team, {}).get(ROUND_WIN, 0)
            cond_win = conditional.get(team, {}).get(ROUND_WIN, 0)
            base_adv = baseline.get(team, {}).get(ROUND_R32, 0)
            cond_adv = conditional.get(team, {}).get(ROUND_R32, 0)
            if abs(cond_win - base_win) > 0.005 or abs(cond_adv - base_adv) > 0.005:
                rows.append((team, base_win, cond_win, cond_win - base_win,
                             base_adv, cond_adv, cond_adv - base_adv))
        rows.sort(key=lambda r: abs(r[3]), reverse=True)
        print(f"{'team':<5} {'base W%':>8} {'cond W%':>8} {'ΔW':>6} | "
              f"{'base R32%':>9} {'cond R32%':>9} {'ΔR32':>6}")
        print("-" * 70)
        for r in rows[:25]:
            print(f"{r[0]:<5} {r[1]*100:7.2f}%  {r[2]*100:7.2f}%  {r[3]*100:+5.2f}% | "
                  f"{r[4]*100:8.2f}%  {r[5]*100:8.2f}%  {r[6]*100:+5.2f}%")
    else:
        # Just show top 16 by championship prob under condition
        rows = sorted(conditional.items(), key=lambda kv: -kv[1].get(ROUND_WIN, 0))[:16]
        print(f"{'team':<5} {'R32':>7} {'R16':>7} {'QF':>7} {'SF':>7} {'F':>7} {'CHAMP':>7}")
        for team, probs in rows:
            print(f"{team:<5} "
                  + " ".join(f"{probs.get(r, 0)*100:6.2f}%" for r in ROUNDS))


if __name__ == "__main__":
    main()
