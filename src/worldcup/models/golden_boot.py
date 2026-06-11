"""Golden Boot model: P(player wins tournament top-scorer).

Recipe:
    1. Aggregate each player's goals + minutes across recent competitions
       (international + top European club leagues).
    2. Compute `goals_per_90 = total_goals * 90 / total_minutes`.
    3. Estimate `expected_minutes_per_match` from
       `total_minutes / total_games` (capped to [0, 90]).
    4. Get `E[team_matches]` from the team-level MC engine:
         E[m] = 3 + P_R32 + P_R16 + P_QF + 2·P_SF
       (P_SF doubles because reaching SF guarantees Final OR 3rd-place game.)
    5. Per player: λ_p = goals_per_90 · (expected_min/90) · E[team_matches].
    6. MC over players: sample G_p ~ Poisson(λ_p) for all tracked players,
       pick argmax (random tiebreak), count wins. P(boot) = wins / N_sims.

Simplifications:
    - All tracked players assumed nominal starters (uniform 70-80 min/match).
    - No injury / rotation modelling — Phase 2 will plug those in.
    - Outlier `goals_per_90` capped at 1.5 to avoid 1-game wonders inflating.
    - We only track players who appeared in WC-relevant international comps
      (qualifiers, Nations League, Euro, Copa, AFCON, Asian Cup, WC 2022) —
      this filters to actual national-team strikers.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.simulator.monte_carlo import (
    ROUND_QF,
    ROUND_R16,
    ROUND_R32,
    ROUND_SF,
    monte_carlo,
)

# International competition league_ids — used to filter for "actually plays
# for the national team" (otherwise club leagues alone could pull in any
# top-scorer regardless of nationality).
INTL_LEAGUE_IDS = (1, 29, 31, 32, 33, 34, 37, 5, 4, 9, 6, 7)

DEFAULT_MINUTES_PER_MATCH = 75.0      # heuristic — top scorers usually start
GOALS_PER_90_CAP = 1.5                 # outlier guard
MIN_TOTAL_MINUTES = 200                # require enough sample
MIN_TOTAL_GAMES = 3


@dataclass
class GoldenBootCandidate:
    player_id: int
    name: str
    wc_team: str
    goals: int
    minutes: int
    games: int
    goals_per_90: float
    avg_min_per_game: float
    e_team_matches: float
    lam: float                  # Poisson rate
    p_win: float


def expected_team_matches(probs: dict[str, dict[str, float]]) -> dict[str, float]:
    """E[matches per team t] = 3 + P_R32 + P_R16 + P_QF + 2·P_SF."""
    out = {}
    for t, p in probs.items():
        out[t] = 3.0 + p.get(ROUND_R32, 0) + p.get(ROUND_R16, 0) + p.get(ROUND_QF, 0) + 2.0 * p.get(ROUND_SF, 0)
    return out


def fetch_candidates(db_path: Path | str = DEFAULT_DB_PATH) -> list[dict]:
    """Pull every player who appears in international comps for a WC 48 team.

    Aggregates their stats across ALL recorded competitions (intl + club).
    Returns list of dicts ready for λ computation.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    intl_ids_csv = ",".join(str(i) for i in INTL_LEAGUE_IDS)
    cur = conn.execute(
        f"""
        WITH wc_teams AS (
            SELECT code FROM teams WHERE in_worldcup_2026 = 1
        ),
        intl_player_team AS (
            SELECT player_id, team_code, MAX(season) AS s, SUM(minutes) AS m
            FROM player_season_stats
            WHERE team_code IN (SELECT code FROM wc_teams)
              AND league_id IN ({intl_ids_csv})
            GROUP BY player_id, team_code
        ),
        primary_team AS (
            -- Each player's primary national team: latest season, most minutes
            SELECT player_id, team_code FROM (
                SELECT player_id, team_code,
                       ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY s DESC, m DESC) rn
                FROM intl_player_team
            ) WHERE rn = 1
        )
        SELECT
            p.id AS player_id,
            p.name,
            pt.team_code AS wc_team,
            COALESCE(SUM(pss.goals), 0)        AS goals,
            COALESCE(SUM(pss.minutes), 0)      AS minutes,
            COALESCE(SUM(pss.games_played), 0) AS games,
            COALESCE(AVG(pss.rating), 0)       AS avg_rating
        FROM primary_team pt
        JOIN players p ON p.id = pt.player_id
        JOIN player_season_stats pss ON pss.player_id = p.id
        GROUP BY p.id, p.name, pt.team_code
        HAVING minutes >= {MIN_TOTAL_MINUTES} AND games >= {MIN_TOTAL_GAMES}
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def compute_lambdas(
    candidates: list[dict],
    e_matches: dict[str, float],
    minutes_per_match: float = DEFAULT_MINUTES_PER_MATCH,
) -> list[GoldenBootCandidate]:
    out: list[GoldenBootCandidate] = []
    for c in candidates:
        gp90 = c["goals"] * 90.0 / c["minutes"] if c["minutes"] > 0 else 0.0
        gp90 = min(gp90, GOALS_PER_90_CAP)
        avg_min = c["minutes"] / max(c["games"], 1)
        # If player averaged < 30 min/game in our data, they're a sub — reduce expected minutes
        adj_min = min(minutes_per_match, avg_min * 1.2)  # +20% headroom over current avg
        em = e_matches.get(c["wc_team"], 3.0)
        lam = gp90 * (adj_min / 90.0) * em
        out.append(
            GoldenBootCandidate(
                player_id=c["player_id"],
                name=c["name"],
                wc_team=c["wc_team"],
                goals=c["goals"],
                minutes=c["minutes"],
                games=c["games"],
                goals_per_90=gp90,
                avg_min_per_game=avg_min,
                e_team_matches=em,
                lam=lam,
                p_win=0.0,
            )
        )
    return out


def simulate_golden_boot(
    candidates: list[GoldenBootCandidate],
    n_sims: int = 100_000,
    seed: int = 42,
) -> None:
    """Mutates `p_win` on each candidate."""
    rng = np.random.default_rng(seed)
    lambdas = np.array([c.lam for c in candidates])
    wins = np.zeros(len(candidates), dtype=np.int64)

    BATCH = 2000
    for start in range(0, n_sims, BATCH):
        n = min(BATCH, n_sims - start)
        # Shape: (n, len(candidates))
        goals = rng.poisson(lambdas, size=(n, len(lambdas)))
        # Per-row max with random tiebreak (jitter)
        jitter = rng.random(goals.shape) * 0.01
        winners = (goals + jitter).argmax(axis=1)
        for w in winners:
            wins[w] += 1

    for i, c in enumerate(candidates):
        c.p_win = wins[i] / n_sims


def golden_boot(
    n_mc_team: int = 20_000,
    n_mc_player: int = 100_000,
    elo_prior: float = 0.5,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[GoldenBootCandidate]:
    print(f"[1/3] Running team MC ({n_mc_team:,} sims) ...")
    mc = monte_carlo(
        n_sims=n_mc_team, elo_prior_strength=elo_prior, verbose=False, db_path=db_path
    )
    e_matches = expected_team_matches(mc["probabilities"])

    print(f"[2/3] Loading player candidates ...")
    rows = fetch_candidates(db_path=db_path)
    print(f"      {len(rows)} eligible scorers")

    cands = compute_lambdas(rows, e_matches)
    cands.sort(key=lambda c: -c.lam)

    print(f"[3/3] Running player MC ({n_mc_player:,} sims) ...")
    simulate_golden_boot(cands, n_sims=n_mc_player)
    cands.sort(key=lambda c: -c.p_win)
    return cands


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--mc-team", type=int, default=20_000)
    ap.add_argument("--mc-player", type=int, default=100_000)
    ap.add_argument("--prior", type=float, default=0.5)
    args = ap.parse_args()

    cands = golden_boot(
        n_mc_team=args.mc_team,
        n_mc_player=args.mc_player,
        elo_prior=args.prior,
    )

    print(f"\n=== Golden Boot probability (top {args.top}) ===")
    print(
        f"{'rk':>3}  {'player':<26}  {'team':<4}  {'g':>3}  {'min':>5}  "
        f"{'gms':>3}  {'g/90':>5}  {'E[m]':>5}  {'λ':>4}  {'P':>5}  {'fair odds':>9}"
    )
    print("-" * 92)
    for i, c in enumerate(cands[:args.top], 1):
        fair = 1.0 / c.p_win if c.p_win > 0 else float("inf")
        fair_str = f"{fair:.1f}" if fair < 999 else "—"
        print(
            f"{i:3d}  {c.name[:26]:<26}  {c.wc_team:<4}  "
            f"{c.goals:>3}  {c.minutes:>5}  {c.games:>3}  "
            f"{c.goals_per_90:>5.2f}  {c.e_team_matches:>5.2f}  "
            f"{c.lam:>4.2f}  {c.p_win*100:>4.1f}%  {fair_str:>9}"
        )
    print(
        f"\nTotal P sums to {sum(c.p_win for c in cands):.3f} "
        f"(should be ~1; field-effect not modelled)"
    )
    print(
        "\nUsage tip: compare 'fair odds' to bookmaker top-scorer prices.\n"
        "  - If bookmaker price > fair odds × (1+vig)  → value bet candidate\n"
        "  - Top-scorer markets aren't on The Odds API; check bookies' tournament-props page manually."
    )


if __name__ == "__main__":
    main()
