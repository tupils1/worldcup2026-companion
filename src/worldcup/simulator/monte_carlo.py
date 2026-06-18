"""Monte Carlo simulation of the 2026 FIFA World Cup.

Pipeline:
    1. Load fitted Dixon-Coles parameters (with Elo prior).
    2. For each of N simulations:
       - Run 12 groups × 6 matches (round-robin) and rank teams.
       - Collect 12 group winners, 12 runners-up, 12 third-placers.
       - Rank 3rd-placers; take top 8 → 32 teams in Round of 32.
       - Build R32 bracket by seeding (1v32, 2v31, …, 16v17).
       - Simulate knockout (R32 → R16 → QF → SF → F + champion).
    3. Aggregate per-team progression probabilities.

Tournament rules (2026):
    - 12 groups (A-L), 4 teams each, single round-robin (3 games per team).
    - Top 2 + 8 best 3rd advance to Round of 32.
    - Knockout: 90 min, then 30 min ET (λ scaled to 1/3), then penalty 50/50.
    - 72 group + 32 KO = 104 matches.

Notes / simplifications:
    - Group ties broken by GD → GF → Elo (no H2H — implementation tradeoff).
    - Knockout draws use neutral venue (no host advantage).
    - 2026's exact R32 bracket isn't FIFA-published; seeded 1v32 is the
      conservative default. Swap for the real bracket when FIFA releases it.
    - Penalty shootouts use 50/50 (could be Elo-weighted; coin-flip is the
      academic-standard simplification).
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.models.dixon_coles import fit
from worldcup.models.markets import score_matrix

HOSTS = frozenset({"USA", "CAN", "MEX"})
MAX_GOALS = 10
N_OUTCOMES = (MAX_GOALS + 1) ** 2  # 121

# Rounds ordered from earliest exit to champion
ROUND_GROUP = "group"
ROUND_R32 = "r32"
ROUND_R16 = "r16"
ROUND_QF = "qf"
ROUND_SF = "sf"
ROUND_F = "final"
ROUND_WIN = "champion"
ROUNDS_ADV = (ROUND_R32, ROUND_R16, ROUND_QF, ROUND_SF, ROUND_F, ROUND_WIN)

# Round-robin pairings within a group of 4 (indices into the 4-team list)
GROUP_PAIRINGS = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))


@dataclass
class MatchSampler:
    """Cached flat score-matrix distributions for fast sampling.

    Applies the SAME goal-rate calibration as DCParams.predict_lambda (lam_scale
    then lam_floor) so the simulator and the digest's per-match model agree — the
    two used to diverge (uncalibrated raw λ here vs calibrated there), which biased
    every projected scoreline low (audit B6)."""

    rho: float
    home_adv: float
    attack: dict[str, float]
    defense: dict[str, float]
    lam_floor: float = 0.0
    lam_scale: float = 1.0
    _cache: dict[tuple[str, str, bool], np.ndarray] = field(default_factory=dict)

    def _flat(self, home: str, away: str, neutral: bool) -> np.ndarray:
        key = (home, away, neutral)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        ha = 0.0 if neutral else self.home_adv
        log_lh = self.attack[home] - self.defense[away] + ha
        log_la = self.attack[away] - self.defense[home]
        lh = max(float(np.exp(log_lh)) * self.lam_scale, self.lam_floor)
        la = max(float(np.exp(log_la)) * self.lam_scale, self.lam_floor)
        M = score_matrix(lh, la, rho=self.rho, max_goals=MAX_GOALS)
        flat = M.flatten()
        flat = flat / flat.sum()  # belt-and-suspenders renorm against fp drift
        self._cache[key] = flat
        return flat

    def sample_match(
        self, home: str, away: str, neutral: bool, rng: np.random.Generator
    ) -> tuple[int, int]:
        idx = int(rng.choice(N_OUTCOMES, p=self._flat(home, away, neutral)))
        return idx // (MAX_GOALS + 1), idx % (MAX_GOALS + 1)

    def sample_extra_time(
        self, home: str, away: str, neutral: bool, rng: np.random.Generator
    ) -> tuple[int, int]:
        # 30 min: scale λ by 1/3. Skip DC τ (small effect at low rates).
        ha = 0.0 if neutral else self.home_adv
        lh = float(np.exp(self.attack[home] - self.defense[away] + ha)) / 3.0
        la = float(np.exp(self.attack[away] - self.defense[home])) / 3.0
        return int(rng.poisson(lh)), int(rng.poisson(la))


@dataclass
class GroupStanding:
    team: str
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def sampler_from_params(params, calibrated: bool = True) -> "MatchSampler":
    """Build a MatchSampler from fitted DCParams. calibrated=True applies the same
    lam_scale/lam_floor as predict_lambda (use for honest goal totals); False keeps
    the legacy raw-λ behavior for call sites not yet migrated off it."""
    return MatchSampler(
        rho=params.rho, home_adv=params.home_advantage,
        attack=params.attack, defense=params.defense,
        lam_floor=params.lam_floor if calibrated else 0.0,
        lam_scale=params.lam_scale if calibrated else 1.0,
    )


def _simulate_group(
    teams: list[str],
    elo: dict[str, float],
    sampler: MatchSampler,
    rng: np.random.Generator,
) -> list[GroupStanding]:
    """Six round-robin matches, then rank teams."""
    standings = {t: GroupStanding(team=t) for t in teams}
    for i, j in GROUP_PAIRINGS:
        home, away = teams[i], teams[j]
        # Host country at home in own group games is non-neutral.
        # martj42 flags all WC games as neutral, but the host is effectively home.
        neutral = home not in HOSTS
        hg, ag = sampler.sample_match(home, away, neutral, rng)
        standings[home].gf += hg
        standings[home].ga += ag
        standings[away].gf += ag
        standings[away].ga += hg
        if hg > ag:
            standings[home].points += 3
        elif hg < ag:
            standings[away].points += 3
        else:
            standings[home].points += 1
            standings[away].points += 1

    return sorted(
        standings.values(),
        key=lambda r: (-r.points, -r.gd, -r.gf, -elo.get(r.team, 1500.0)),
    )


def _simulate_knockout(
    home: str,
    away: str,
    sampler: MatchSampler,
    rng: np.random.Generator,
) -> str:
    """Single-elimination match. Returns winner code."""
    hg, ag = sampler.sample_match(home, away, neutral=True, rng=rng)
    if hg != ag:
        return home if hg > ag else away
    eh, ea = sampler.sample_extra_time(home, away, neutral=True, rng=rng)
    if eh != ea:
        return home if eh > ea else away
    return home if rng.random() < 0.5 else away  # penalties (coin flip)


def _make_r32_bracket(
    winners: list[str], runners_up: list[str], best_thirds: list[str]
) -> list[tuple[str, str]]:
    """32 teams → 16 R32 pairs.

    Seeds 1-12 = group winners (A→L order), 13-24 = runners-up (A→L order),
    25-32 = best 3rd-placers in ranked order. Pair seed i vs seed (33-i).
    """
    seeds = list(winners) + list(runners_up) + list(best_thirds)
    assert len(seeds) == 32
    return [(seeds[i], seeds[31 - i]) for i in range(16)]


def simulate_one_tournament(
    sampler: MatchSampler,
    groups: dict[str, list[str]],
    elo: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, str]:
    """Run one full tournament. Returns {team: furthest_round_reached}."""
    progression: dict[str, str] = {}

    winners: list[str] = []
    runners_up: list[str] = []
    thirds: list[GroupStanding] = []

    for letter in sorted(groups):
        ranked = _simulate_group(groups[letter], elo, sampler, rng)
        for r in ranked:
            progression[r.team] = ROUND_GROUP
        winners.append(ranked[0].team)
        runners_up.append(ranked[1].team)
        thirds.append(ranked[2])

    # Best 8 third-placers
    best_thirds_objs = sorted(
        thirds, key=lambda r: (-r.points, -r.gd, -r.gf, -elo.get(r.team, 1500.0))
    )[:8]
    best_thirds = [r.team for r in best_thirds_objs]

    r32_teams = winners + runners_up + best_thirds
    for t in r32_teams:
        progression[t] = ROUND_R32

    # Knockout
    bracket = _make_r32_bracket(winners, runners_up, best_thirds)
    r16 = [_simulate_knockout(h, a, sampler, rng) for h, a in bracket]
    for t in r16:
        progression[t] = ROUND_R16

    qf = [_simulate_knockout(r16[i], r16[i + 1], sampler, rng) for i in range(0, 16, 2)]
    for t in qf:
        progression[t] = ROUND_QF

    sf = [_simulate_knockout(qf[i], qf[i + 1], sampler, rng) for i in range(0, 8, 2)]
    for t in sf:
        progression[t] = ROUND_SF

    finalists = [_simulate_knockout(sf[i], sf[i + 1], sampler, rng) for i in range(0, 4, 2)]
    for t in finalists:
        progression[t] = ROUND_F

    champion = _simulate_knockout(finalists[0], finalists[1], sampler, rng)
    progression[champion] = ROUND_WIN

    return progression


def monte_carlo(
    n_sims: int = 10_000,
    elo_prior_strength: float = 0.5,
    seed: int = 42,
    db_path: Path | str = DEFAULT_DB_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run MC and return per-team probabilities.

    Returns:
        {
          "n_sims": int,
          "model": {...},
          "probabilities": {team: {round_label: prob_reached_at_least}},
          "elapsed_sec": float,
        }
    """
    if verbose:
        print(f"Fitting Dixon-Coles (prior={elo_prior_strength}) ...")
    params = fit(
        db_path=db_path,
        since="2014-01-01",
        elo_prior_strength=elo_prior_strength,
    )

    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "teams.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    groups = {letter: g["teams"] for letter, g in cfg["groups"].items()}

    conn = sqlite3.connect(db_path)
    elo = {
        r[0]: float(r[1])
        for r in conn.execute(
            "SELECT team_code, value FROM team_ratings "
            "WHERE rating_type='elo' AND source='eloratings.net'"
        )
    }
    conn.close()

    all_teams: set[str] = set()
    for g in groups.values():
        all_teams.update(g)
    missing = all_teams - set(params.attack)
    if missing:
        raise ValueError(f"Missing fit params for teams: {missing}")

    # Calibrated λ (lam_scale/floor), same as predict_lambda — keeps projected
    # goal totals consistent with the per-match model instead of running ~4% low (B6).
    sampler = sampler_from_params(params, calibrated=True)

    # Pre-warm cache for all group-stage pairings (the only deterministic part)
    for letter, teams in groups.items():
        for i, j in GROUP_PAIRINGS:
            sampler._flat(teams[i], teams[j], teams[i] not in HOSTS)

    rng = np.random.default_rng(seed)

    # round_counts[team][round] = # simulations team reached AT LEAST that round
    round_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    round_idx = {r: i for i, r in enumerate(ROUNDS_ADV)}

    t0 = time.time()
    for sim in range(n_sims):
        progression = simulate_one_tournament(sampler, groups, elo, rng)
        for team, furthest in progression.items():
            if furthest == ROUND_GROUP:
                continue
            for r in ROUNDS_ADV[: round_idx[furthest] + 1]:
                round_counts[team][r] += 1
        if verbose and (sim + 1) % max(1, n_sims // 10) == 0:
            elapsed = time.time() - t0
            rate = (sim + 1) / max(elapsed, 1e-9)
            eta = (n_sims - sim - 1) / max(rate, 1e-9)
            print(
                f"  {sim + 1:>7,} / {n_sims:,}  ({rate:.0f} sims/s, ETA {eta:5.1f}s)"
            )

    elapsed = time.time() - t0
    probabilities = {
        team: {r: round_counts[team].get(r, 0) / n_sims for r in ROUNDS_ADV}
        for team in all_teams
    }
    return {
        "n_sims": n_sims,
        "elapsed_sec": elapsed,
        "model": {
            "elo_prior_strength": elo_prior_strength,
            "home_advantage": params.home_advantage,
            "rho": params.rho,
            "n_matches_fit": params.n_matches,
            "log_lik": params.fit_loglik,
        },
        "probabilities": probabilities,
    }


def print_report(result: dict[str, Any], top_n: int = 16) -> None:
    probs = result["probabilities"]
    n = result["n_sims"]
    ranked = sorted(probs.items(), key=lambda kv: -kv[1][ROUND_WIN])
    print(
        f"\n=== {n:,} simulations  ({result['elapsed_sec']:.1f}s, "
        f"{n / max(result['elapsed_sec'], 1e-9):.0f} sims/s) ==="
    )
    print(
        f"{'rk':>3}  {'team':4}  "
        f"{'R32→R16':>8}  {'→QF':>6}  {'→SF':>6}  {'→Final':>7}  {'Champ':>6}"
    )
    print("-" * 60)
    for i, (team, p) in enumerate(ranked[:top_n], 1):
        print(
            f"{i:3d}  {team:4}  "
            f"{p[ROUND_R16] * 100:7.1f}%  "
            f"{p[ROUND_QF] * 100:5.1f}%  "
            f"{p[ROUND_SF] * 100:5.1f}%  "
            f"{p[ROUND_F] * 100:6.1f}%  "
            f"{p[ROUND_WIN] * 100:5.1f}%"
        )
    print(f"\n=== Bottom 5 (least likely to advance) ===")
    for i, (team, p) in enumerate(ranked[-5:], len(ranked) - 4):
        print(
            f"{i:3d}  {team:4}  "
            f"{p[ROUND_R16] * 100:7.1f}%  "
            f"{p[ROUND_QF] * 100:5.1f}%  "
            f"{p[ROUND_SF] * 100:5.1f}%  "
            f"{p[ROUND_F] * 100:6.1f}%  "
            f"{p[ROUND_WIN] * 100:5.1f}%"
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prior", type=float, default=0.5)
    args = ap.parse_args()

    result = monte_carlo(
        n_sims=args.n_sims,
        elo_prior_strength=args.prior,
        seed=args.seed,
    )
    print_report(result, top_n=16)


if __name__ == "__main__":
    main()
