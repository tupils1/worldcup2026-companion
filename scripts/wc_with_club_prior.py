#!/usr/bin/env python
"""WC Dixon-Coles refit with CLUB-DERIVED priors.

Pipeline:
    1. Fit Club Dixon-Coles on ~3000 club matches (9 major leagues + Euro cups).
    2. For each WC 48 team, look up its known players' current clubs and
       aggregate their club attack/defense ratings → national-team prior.
    3. Refit WC Dixon-Coles, replacing the Elo prior with the club-derived prior.
    4. Run MC and compare to Polymarket + Books.

Limitations of current squad data:
    - Players sourced from topscorers only (281 unique). Strong-league strikers
      well-represented; mid-tier teams (MAR, CIV, etc.) under-represented.
    - For teams with <3 mapped players, fall back to Elo prior.

Run:
    PYTHONPATH=src python scripts/wc_with_club_prior.py
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.ingest.api_football import APIFootballClient
from worldcup.models.dixon_coles import (
    DEFAULT_DECAY_PER_DAY,
    DCParams,
    _load_elo,
    _load_matches,
)
from worldcup.models.markets import score_matrix
from worldcup.strategy.value_bets import devig_shin
from worldcup.simulator.monte_carlo import (
    MatchSampler,
    ROUND_R32,
    ROUND_R16,
    ROUND_QF,
    ROUND_SF,
    ROUND_F,
    ROUND_WIN,
    simulate_one_tournament,
)
import yaml

LEAGUES = [
    (39, 2024), (140, 2024), (61, 2024), (78, 2024), (135, 2024),
    (88, 2024), (2, 2024), (3, 2024), (848, 2024),
]
MIN_PLAYERS_FOR_CLUB_PRIOR = 3  # need at least 3 mapped players for confidence
ELO_PRIOR_FALLBACK_STRENGTH = 0.5
CLUB_PRIOR_STRENGTH = 2.0
CLUB_ELO_BLEND = 0.7  # weight on club prior when both available


# ─── Step 1: Pull club matches + fit club DC ───────────────────────────────

def pull_club_matches(client: APIFootballClient) -> list[dict]:
    rows = []
    for league, season in LEAGUES:
        data = client.get("/fixtures", league=league, season=season)
        for f in data.get("response", []):
            fx = f["fixture"]
            if fx["status"]["short"] != "FT":
                continue
            rows.append({
                "home": f["teams"]["home"]["name"],
                "away": f["teams"]["away"]["name"],
                "home_id": f["teams"]["home"]["id"],
                "away_id": f["teams"]["away"]["id"],
                "date": fx["date"][:10],
                "league_id": league,
                "home_score": f["score"]["fulltime"]["home"],
                "away_score": f["score"]["fulltime"]["away"],
            })
    return rows


def fit_club_dc(matches: list[dict], ridge: float = 0.5) -> dict:
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    leagues = sorted({m["league_id"] for m in matches})
    league_to_idx = {l: i for i, l in enumerate(leagues)}
    nl = len(leagues)

    today = dt.date.today()
    home_idx = np.fromiter((team_to_idx[m["home"]] for m in matches), dtype=np.int64)
    away_idx = np.fromiter((team_to_idx[m["away"]] for m in matches), dtype=np.int64)
    lg_idx   = np.fromiter((league_to_idx[m["league_id"]] for m in matches), dtype=np.int64)
    home_goals = np.fromiter((m["home_score"] for m in matches), dtype=np.float64)
    away_goals = np.fromiter((m["away_score"] for m in matches), dtype=np.float64)
    days_ago = np.fromiter(
        ((today - dt.date.fromisoformat(m["date"])).days for m in matches),
        dtype=np.float64,
    )
    weights = np.exp(-DEFAULT_DECAY_PER_DAY * days_ago)
    is_neutral = np.zeros(len(matches), dtype=bool)

    def nll(params):
        attack = params[:n]; defense = params[n:2*n]
        league_off_rest = params[2*n: 2*n + nl - 1]
        league_off = np.concatenate([[0.0], league_off_rest])
        home_adv = params[-2]; rho = params[-1]
        ha = np.where(is_neutral, 0.0, home_adv)
        lo = league_off[lg_idx]
        log_lh = attack[home_idx] - defense[away_idx] + ha + lo
        log_la = attack[away_idx] - defense[home_idx]      + lo
        lh = np.exp(log_lh); la = np.exp(log_la)
        log_p_h = home_goals * log_lh - lh - gammaln(home_goals + 1)
        log_p_a = away_goals * log_la - la - gammaln(away_goals + 1)
        tau = np.ones_like(lh)
        m00 = (home_goals == 0) & (away_goals == 0)
        m10 = (home_goals == 1) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m11 = (home_goals == 1) & (away_goals == 1)
        tau[m00] = 1.0 - lh[m00]*la[m00]*rho
        tau[m10] = 1.0 + la[m10]*rho
        tau[m01] = 1.0 + lh[m01]*rho
        tau[m11] = 1.0 - rho
        log_tau = np.log(np.maximum(tau, 1e-10))
        loss = -float(np.sum(weights * (log_p_h + log_p_a + log_tau)))
        loss += ridge * float(np.sum(attack**2 + defense**2))
        return loss

    x0 = np.concatenate([np.zeros(n), np.zeros(n), np.zeros(nl-1), [0.3], [-0.05]])
    constraints = [{"type": "eq", "fun": lambda p: float(np.sum(p[:n]))}]
    bounds = [(-3, 3)] * (2*n) + [(-1, 1)] * (nl-1) + [(0, 1), (-0.2, 0.2)]
    res = minimize(nll, x0, method="SLSQP", constraints=constraints, bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-6})
    return {
        "teams": teams,
        "attack": dict(zip(teams, res.x[:n].tolist())),
        "defense": dict(zip(teams, res.x[n:2*n].tolist())),
        "home_advantage": float(res.x[-2]),
        "rho": float(res.x[-1]),
        "n_matches": len(matches),
        "fit_loglik": -float(res.fun),
    }


# ─── Step 2: Derive WC team priors from club ratings ───────────────────────

def fuzzy_club_match(club_name: str, club_teams: list[str]) -> str | None:
    """Match a player's API-Football club name to club DC's team_name."""
    if not club_name:
        return None
    if club_name in club_teams:
        return club_name
    norm = club_name.lower().strip()
    for t in club_teams:
        if t.lower().strip() == norm:
            return t
    # Substring match on first word
    first = norm.split()[0] if norm.split() else ""
    if len(first) > 3:
        for t in club_teams:
            if first in t.lower():
                return t
    return None


def derive_wc_team_priors(
    club_params: dict,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, dict]:
    """Aggregate each WC team's club-derived attack/defense from their known players."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    wc_codes = {r["code"] for r in conn.execute(
        "SELECT code FROM teams WHERE in_worldcup_2026=1"
    )}

    # Group players by nationality + look up their current club
    team_clubs: dict[str, list[str]] = defaultdict(list)
    team_player_count: dict[str, int] = defaultdict(int)
    unmapped_clubs = []
    for r in conn.execute("""
        SELECT name, nationality_code, current_club_name
        FROM players
        WHERE current_club_name IS NOT NULL
          AND nationality_code IN (SELECT code FROM teams WHERE in_worldcup_2026=1)
    """):
        match = fuzzy_club_match(r["current_club_name"], club_params["teams"])
        if not match:
            unmapped_clubs.append(r["current_club_name"])
            continue
        team_clubs[r["nationality_code"]].append(match)
        team_player_count[r["nationality_code"]] += 1
    conn.close()

    priors = {}
    for team in wc_codes:
        clubs = team_clubs.get(team, [])
        if len(clubs) < MIN_PLAYERS_FOR_CLUB_PRIOR:
            continue
        attacks = [club_params["attack"][c] for c in clubs]
        defenses = [club_params["defense"][c] for c in clubs]
        priors[team] = {
            "attack": float(np.mean(attacks)),
            "defense": float(np.mean(defenses)),
            "n_players": len(clubs),
            "clubs_sample": list(set(clubs))[:5],
        }
    return priors, unmapped_clubs


# ─── Step 3: Refit WC DC with club-derived priors ──────────────────────────

def fit_wc_with_club_prior(
    club_priors: dict[str, dict],
    db_path: Path | str = DEFAULT_DB_PATH,
    elo_scale: float = 0.003,
    club_prior_strength: float = CLUB_PRIOR_STRENGTH,
    elo_prior_strength: float = ELO_PRIOR_FALLBACK_STRENGTH,
) -> DCParams:
    """Modified WC DC fit: use club-derived prior where available, else fall back to Elo."""
    rows = _load_matches(db_path=db_path, since="2014-01-01", until=None)
    teams = sorted({r["home_code"] for r in rows} | {r["away_code"] for r in rows})
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    today = dt.date.today()
    home_idx = np.fromiter((team_to_idx[r["home_code"]] for r in rows), dtype=np.int64)
    away_idx = np.fromiter((team_to_idx[r["away_code"]] for r in rows), dtype=np.int64)
    home_goals = np.fromiter((r["home_score"] for r in rows), dtype=np.float64)
    away_goals = np.fromiter((r["away_score"] for r in rows), dtype=np.float64)
    is_neutral = np.fromiter((bool(r["neutral_venue"]) for r in rows), dtype=bool, count=len(rows))
    days_ago = np.fromiter(
        ((today - dt.date.fromisoformat(r["match_date"][:10])).days for r in rows),
        dtype=np.float64,
    )
    weights = np.exp(-DEFAULT_DECAY_PER_DAY * days_ago)

    # Build prior arrays
    elo_map = _load_elo(db_path)
    elos_in_fit = [elo_map.get(t) for t in teams]
    mean_elo = float(np.nanmean([e for e in elos_in_fit if e is not None]))
    elos = np.array([e if e else mean_elo for e in elos_in_fit])
    elo_attack_prior = elo_scale * (elos - mean_elo)
    elo_defense_prior = -elo_scale * (elos - mean_elo)

    # Where we have club priors, mix them with Elo prior (blend)
    attack_prior = np.copy(elo_attack_prior)
    defense_prior = np.copy(elo_defense_prior)
    prior_strength_arr = np.full(n, elo_prior_strength)
    used_club = []
    for i, t in enumerate(teams):
        if t in club_priors:
            # club_priors has raw club attack/defense values (~ -1 to +1 typical).
            # Need to scale them down — international DC params are smaller (-0.5 to +0.5 typically).
            # Use the club value directly but scaled by 0.5 to match international magnitude
            club_a = club_priors[t]["attack"] * 0.5
            club_d = club_priors[t]["defense"] * 0.5
            # Blend with Elo prior
            attack_prior[i]  = CLUB_ELO_BLEND * club_a + (1 - CLUB_ELO_BLEND) * elo_attack_prior[i]
            defense_prior[i] = CLUB_ELO_BLEND * club_d + (1 - CLUB_ELO_BLEND) * elo_defense_prior[i]
            prior_strength_arr[i] = club_prior_strength
            used_club.append(t)

    def nll(params):
        attack = params[:n]; defense = params[n:2*n]
        home_adv = params[-2]; rho = params[-1]
        ha = np.where(is_neutral, 0.0, home_adv)
        log_lh = attack[home_idx] - defense[away_idx] + ha
        log_la = attack[away_idx] - defense[home_idx]
        lh = np.exp(log_lh); la = np.exp(log_la)
        log_p_h = home_goals * log_lh - lh - gammaln(home_goals + 1)
        log_p_a = away_goals * log_la - la - gammaln(away_goals + 1)
        tau = np.ones_like(lh)
        m00 = (home_goals == 0) & (away_goals == 0)
        m10 = (home_goals == 1) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m11 = (home_goals == 1) & (away_goals == 1)
        tau[m00] = 1.0 - lh[m00]*la[m00]*rho
        tau[m10] = 1.0 + la[m10]*rho
        tau[m01] = 1.0 + lh[m01]*rho
        tau[m11] = 1.0 - rho
        log_tau = np.log(np.maximum(tau, 1e-10))
        loss = -float(np.sum(weights * (log_p_h + log_p_a + log_tau)))
        # Per-team prior with adaptive strength
        loss += float(np.sum(prior_strength_arr * ((attack - attack_prior)**2 + (defense - defense_prior)**2)))
        return loss

    x0 = np.concatenate([attack_prior, defense_prior, [0.3], [-0.05]])
    constraints = [{"type": "eq", "fun": lambda p: float(np.sum(p[:n]))}]
    bounds = [(-3, 3)] * (2*n) + [(0, 1), (-0.2, 0.2)]
    res = minimize(nll, x0, method="SLSQP", constraints=constraints, bounds=bounds,
                   options={"maxiter": 300, "ftol": 1e-7})
    return DCParams(
        teams=tuple(teams),
        attack=dict(zip(teams, res.x[:n].tolist())),
        defense=dict(zip(teams, res.x[n:2*n].tolist())),
        home_advantage=float(res.x[-2]),
        rho=float(res.x[-1]),
        n_matches=len(rows),
        fit_loglik=-float(res.fun),
        fit_converged=bool(res.success),
    ), used_club


# ─── Step 4: MC with new params ────────────────────────────────────────────

def run_mc(params: DCParams, n_sims: int = 30000, db_path=DEFAULT_DB_PATH) -> dict:
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "teams.yaml").read_text()
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
    rng = np.random.default_rng(42)
    counts = defaultdict(lambda: defaultdict(int))
    ROUNDS = (ROUND_R32, ROUND_R16, ROUND_QF, ROUND_SF, ROUND_F, ROUND_WIN)
    for _ in range(n_sims):
        prog = simulate_one_tournament(sampler, groups, elo, rng)
        for team, furthest in prog.items():
            if furthest == "group":
                continue
            idx = ROUNDS.index(furthest)
            for r in ROUNDS[:idx+1]:
                counts[team][r] += 1
    return {team: {r: counts[team][r] / n_sims for r in ROUNDS} for team in counts}


# ─── Step 5: Compare to Polymarket + Books ────────────────────────────────

def get_market_consensus(db_path=DEFAULT_DB_PATH) -> tuple[dict, dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    poly = {}
    for r in conn.execute("""
        SELECT subject_code, price FROM odds
        WHERE bookmaker='polymarket' AND market_scope='outright' AND market='winner'
        GROUP BY subject_code
    """):
        poly[r["subject_code"]] = 1.0 / r["price"]

    book_devigged = defaultdict(list)
    book_quotes = defaultdict(dict)
    for r in conn.execute("""
        SELECT o1.bookmaker, o1.subject_code, o1.price
        FROM odds o1
        JOIN (SELECT bookmaker, subject_code, MAX(captured_at) AS mc
              FROM odds WHERE market_scope='outright' AND bookmaker != 'polymarket'
              GROUP BY bookmaker, subject_code) latest
          ON o1.bookmaker=latest.bookmaker AND o1.subject_code=latest.subject_code
          AND o1.captured_at=latest.mc
        WHERE o1.market_scope='outright' AND o1.bookmaker != 'polymarket'
    """):
        book_quotes[r["bookmaker"]][r["subject_code"]] = r["price"]
    for book, quotes in book_quotes.items():
        if len(quotes) < 10: continue
        teams = list(quotes.keys())
        devig = devig_shin([quotes[t] for t in teams])
        for t, p in zip(teams, devig):
            book_devigged[t].append(p)
    books = {t: float(np.mean(ps)) for t, ps in book_devigged.items() if ps}
    conn.close()
    return poly, books


def main():
    print("="*80)
    print("WC 2026 Champion: CLUB-PRIOR-AUGMENTED MODEL")
    print("="*80)

    print("\n[1/5] Pulling club matches + fitting club DC ...")
    with APIFootballClient() as client:
        club_matches = pull_club_matches(client)
    print(f"  {len(club_matches)} matches across 9 leagues")
    club_params = fit_club_dc(club_matches)
    print(f"  Club DC: {len(club_params['teams'])} teams, log-lik {club_params['fit_loglik']:.0f}")

    print("\n[2/5] Aggregating WC team priors from squad clubs ...")
    priors, unmapped = derive_wc_team_priors(club_params)
    print(f"  Teams with club-prior: {len(priors)}/48")
    print(f"  Sample priors:")
    for t in sorted(priors, key=lambda x: -priors[x]['attack'])[:10]:
        p = priors[t]
        print(f"    {t}: club_attack={p['attack']:+.2f}  club_defense={p['defense']:+.2f}  "
              f"n={p['n_players']}  ({', '.join(p['clubs_sample'][:3])})")

    print("\n[3/5] Refitting WC DC with club-derived priors ...")
    new_params, used = fit_wc_with_club_prior(priors)
    print(f"  Converged={new_params.fit_converged}  log-lik={new_params.fit_loglik:.0f}")
    print(f"  Teams using club prior: {len(used)}")

    print("\n[4/5] Running 30k MC with new params ...")
    new_probs = run_mc(new_params, n_sims=30000)

    print("\n[5/5] Three-way comparison ...")
    poly, books = get_market_consensus()

    # Show key teams
    key = ["ESP", "FRA", "ENG", "BRA", "ARG", "POR", "GER", "NED", "ITA", "BEL",
           "MAR", "COL", "ECU", "SUI", "URU", "MEX", "USA", "JPN", "KOR",
           "CRO", "AUS", "POL"]
    key = [t for t in key if t in new_probs]
    print(f"\n{'team':<5} {'NewMod':>7} {'Poly':>7} {'Books':>7} | "
          f"{'NewMod-Poly':>12} {'(prior)':>9}")
    print("-" * 60)
    for t in key:
        nm = new_probs[t].get(ROUND_WIN, 0)
        pm = poly.get(t, float("nan"))
        bk = books.get(t, float("nan"))
        gap = nm - pm if not np.isnan(pm) else float("nan")
        had_prior = "✓" if t in used else " "
        print(f"{t:<5} {nm*100:6.2f}% {pm*100:6.2f}% {bk*100:6.2f}% | "
              f"{gap*100:+10.2f}%  {had_prior:>9}")


if __name__ == "__main__":
    main()
