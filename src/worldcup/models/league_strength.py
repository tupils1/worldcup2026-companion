"""League strength normalization.

Each league has its own scoring rate. A team scoring 1.5/match in Ligue 1
(league mean 1.30) is much better than a team scoring 1.5/match in Bundesliga
(league mean 1.60). Without normalization, cross-league predictions over/under
estimate based on which league each team came from.

Recipe:
    offense_index = team_goals_for_per_match / league_mean
    defense_index = team_goals_against_per_match / league_mean

    Cross-league match (e.g. Cup final): expected goals
        λ_home = home_offense_idx * away_defense_idx * cup_mean
        λ_away = away_offense_idx * home_defense_idx * cup_mean

Sources for `league_mean`:
    - API-Football's `/teams/statistics` aggregated across all teams (true mean
      but slow — one API call per league per query).
    - Hard-coded historical 5-year averages (fast, ~5% off true current).
    - Pull on demand and cache.

We use hard-coded current-season averages as defaults, with helper to refresh
from API when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

# Historical mean per-team goals per match (2024-25 season aligned where known).
# Source: well-known league averages; refresh quarterly via refresh_league_means().
LEAGUE_MEANS: dict[tuple[int, int], float] = {
    # (league_id, season) → mean_goals_per_team_per_match
    (39, 2024): 1.40,   # EPL
    (39, 2025): 1.40,   # EPL 2025-26
    (140, 2024): 1.25,  # La Liga
    (140, 2025): 1.25,
    (61, 2024): 1.33,   # Ligue 1
    (61, 2025): 1.33,
    (78, 2024): 1.60,   # Bundesliga
    (78, 2025): 1.60,
    (135, 2024): 1.30,  # Serie A
    (135, 2025): 1.30,
    (88, 2024): 1.45,   # Eredivisie
    (94, 2024): 1.35,   # Primeira Liga
    (203, 2024): 1.40,  # Süper Lig
    (179, 2024): 1.55,  # Scottish Premiership
    # International competitions
    (2, 2025): 1.55,    # UEFA Champions League (KO-heavy)
    (3, 2025): 1.50,    # UEFA Europa League
    (848, 2025): 1.40,  # UEFA Europa Conference League
    (1, 2022): 1.40,    # FIFA World Cup 2022
    (1, 2026): 1.45,    # FIFA World Cup 2026 (estimate)
    # WC qualifications usually higher scoring (mismatch games)
    (29, 2026): 1.70,   # CAF
    (31, 2026): 1.80,   # CONCACAF
    (32, 2024): 1.55,   # UEFA qualifiers
    (33, 2026): 2.10,   # OFC (very mismatched)
    (34, 2026): 1.40,   # CONMEBOL
    # Defaults
    ("default", "club"): 1.40,
    ("default", "international"): 1.45,
}


def get_league_mean(league_id: int, season: int, default: float = 1.40) -> float:
    """Return mean goals per team per match for (league, season)."""
    return LEAGUE_MEANS.get((league_id, season), default)


def refresh_league_mean(
    league_id: int,
    season: int,
    api_key: str,
    update_cache: bool = True,
) -> float:
    """Compute league mean from API-Football's standings (sum goals / 2 / matches).

    Use when you want exact current-season mean instead of historical default.
    """
    r = httpx.get(
        "https://v3.football.api-sports.io/standings",
        headers={"x-apisports-key": api_key},
        params={"league": league_id, "season": season},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json().get("response", [])
    if not data:
        return get_league_mean(league_id, season)
    teams = data[0]["league"]["standings"][0]
    total_gf = sum(t["all"]["goals"]["for"] for t in teams)
    total_played = sum(t["all"]["played"] for t in teams)
    if total_played == 0:
        return get_league_mean(league_id, season)
    # Each match contributes goals to 2 teams' "for", so total_gf == total goals scored
    # Mean goals per team per match = total_gf / total_played (since total_played counts both sides)
    mean = total_gf / total_played
    if update_cache:
        LEAGUE_MEANS[(league_id, season)] = mean
    return mean


def normalize_team_strength(
    goals_for_avg: float,
    goals_against_avg: float,
    league_id: int,
    season: int,
) -> tuple[float, float]:
    """Return (offense_index, defense_index) relative to the league.

    offense_index > 1 → above-average attack
    defense_index < 1 → above-average defense (concedes fewer)
    """
    mean = get_league_mean(league_id, season)
    return goals_for_avg / mean, goals_against_avg / mean


def predict_lambdas(
    home_for_avg: float, home_against_avg: float, home_league_id: int, home_season: int,
    away_for_avg: float, away_against_avg: float, away_league_id: int, away_season: int,
    venue_league_id: int, venue_season: int,
) -> tuple[float, float]:
    """Cross-league match prediction (e.g. Cup final).

    Use *each team's domestic league* to normalize its strength, then apply
    expected goals at the *venue league*'s mean (e.g. UCL mean for Cup final).
    """
    h_off, h_def = normalize_team_strength(home_for_avg, home_against_avg, home_league_id, home_season)
    a_off, a_def = normalize_team_strength(away_for_avg, away_against_avg, away_league_id, away_season)
    venue_mean = get_league_mean(venue_league_id, venue_season)
    lh = h_off * a_def * venue_mean
    la = a_off * h_def * venue_mean
    return lh, la
