#!/usr/bin/env python
"""Club Dixon-Coles ad-hoc predictor.

Pulls ~2000 club matches from big 5 leagues + UCL/UEL/UECL 2024-25 via API-Football,
fits DC in-memory with attack/defense parameters per club + league fixed effects,
then predicts any cross-league cup matchup.

Run:
    PYTHONPATH=src python scripts/predict_club_match.py
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import httpx
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from worldcup.ingest.api_football import APIFootballClient
from worldcup.models.markets import (
    fair_decimal_odds,
    most_likely_score,
    prob_1x2,
    prob_asian_handicap,
    prob_over_under,
    score_matrix,
)

# Leagues to pull. (league_id, season)
LEAGUES = [
    (39, 2024),   # EPL 2024-25
    (140, 2024),  # La Liga 2024-25
    (61, 2024),   # Ligue 1 2024-25
    (78, 2024),   # Bundesliga 2024-25
    (135, 2024),  # Serie A 2024-25
    (88, 2024),   # Eredivisie (for Champions League representation)
    (2, 2024),    # UEFA Champions League 2024-25
    (3, 2024),    # UEFA Europa League 2024-25
    (848, 2024),  # UEFA Europa Conference League 2024-25
]


def pull_matches(client: APIFootballClient) -> list[dict]:
    """Pull all finished fixtures across configured leagues. Returns flat list."""
    all_matches = []
    for league, season in LEAGUES:
        try:
            print(f"  pulling league={league} season={season} ...", end=" ", flush=True)
            data = client.get("/fixtures", league=league, season=season)
            rows = []
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
            print(f"{len(rows)} finished")
            all_matches.extend(rows)
        except Exception as exc:
            print(f"ERROR: {exc}")
    return all_matches


def fit_club_dc(matches: list[dict],
                decay_xi_per_day: float = 0.0019,
                ridge_lambda: float = 0.5) -> dict:
    """Fit Dixon-Coles to club data. Returns dict with team params + meta."""
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
    weights = np.exp(-decay_xi_per_day * days_ago)
    is_neutral = np.zeros(len(matches), dtype=bool)  # club games rarely neutral except finals

    # Parameter vector: [attack_1..n, defense_1..n, league_offsets_1..nl-1, home_adv, rho]
    # League offsets: first league is baseline (0), rest are relative.
    n_params = 2*n + (nl - 1) + 2

    def nll(params):
        attack = params[:n]
        defense = params[n:2*n]
        league_off_rest = params[2*n: 2*n + nl - 1]
        league_off = np.concatenate([[0.0], league_off_rest])
        home_adv = params[-2]
        rho = params[-1]

        ha = np.where(is_neutral, 0.0, home_adv)
        lo = league_off[lg_idx]
        log_lh = attack[home_idx] - defense[away_idx] + ha + lo
        log_la = attack[away_idx] - defense[home_idx]      + lo
        lh = np.exp(log_lh)
        la = np.exp(log_la)
        log_p_h = home_goals * log_lh - lh - gammaln(home_goals + 1)
        log_p_a = away_goals * log_la - la - gammaln(away_goals + 1)

        tau = np.ones_like(lh)
        m00 = (home_goals == 0) & (away_goals == 0)
        m10 = (home_goals == 1) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m11 = (home_goals == 1) & (away_goals == 1)
        tau[m00] = 1.0 - lh[m00] * la[m00] * rho
        tau[m10] = 1.0 + la[m10] * rho
        tau[m01] = 1.0 + lh[m01] * rho
        tau[m11] = 1.0 - rho
        log_tau = np.log(np.maximum(tau, 1e-10))

        loss = -float(np.sum(weights * (log_p_h + log_p_a + log_tau)))
        # Ridge on attack/defense for shrinkage
        loss += ridge_lambda * float(np.sum(attack**2 + defense**2))
        return loss

    x0 = np.concatenate([
        np.zeros(n), np.zeros(n),
        np.zeros(nl - 1),
        [0.3], [-0.05],
    ])
    constraints = [{"type": "eq", "fun": lambda p: float(np.sum(p[:n]))}]
    bounds = [(-3, 3)] * (2*n) + [(-1, 1)] * (nl - 1) + [(0, 1), (-0.2, 0.2)]

    print(f"  Fitting DC on {len(matches)} matches, {n} teams, {nl} leagues ...")
    t0 = time.time()
    res = minimize(nll, x0, method="SLSQP", constraints=constraints, bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-6, "disp": False})
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, success={res.success}, log-lik={-res.fun:.0f}")

    return {
        "teams": teams,
        "attack": dict(zip(teams, res.x[:n].tolist())),
        "defense": dict(zip(teams, res.x[n:2*n].tolist())),
        "league_offset": dict(zip(leagues, [0.0] + res.x[2*n:2*n+nl-1].tolist())),
        "home_advantage": float(res.x[-2]),
        "rho": float(res.x[-1]),
        "n_matches": len(matches),
        "fit_loglik": -float(res.fun),
    }


def predict_match(params: dict, home: str, away: str, venue_league: int, neutral: bool = True):
    """Predict a single match using fitted club DC params."""
    if home not in params["attack"]:
        raise ValueError(f"{home} not in fitted teams")
    if away not in params["attack"]:
        raise ValueError(f"{away} not in fitted teams")
    ha = 0.0 if neutral else params["home_advantage"]
    lo = params["league_offset"].get(venue_league, 0.0)
    log_lh = params["attack"][home] - params["defense"][away] + ha + lo
    log_la = params["attack"][away] - params["defense"][home]      + lo
    return float(np.exp(log_lh)), float(np.exp(log_la))


def report(home, away, lh, la, rho):
    M = score_matrix(lh, la, rho=rho)
    p_h, p_d, p_a = prob_1x2(M)
    mh, ma, p_sc = most_likely_score(M)
    print(f"     λ_{home[:3]}={lh:.2f}  λ_{away[:3]}={la:.2f}  total={lh+la:.2f}  modal {mh}-{ma} ({p_sc:.1%})")
    print(f"     1X2: {home[:14]} {p_h*100:5.1f}% ({fair_decimal_odds(p_h):.2f})  "
          f"Draw {p_d*100:5.1f}% ({fair_decimal_odds(p_d):.2f})  "
          f"{away[:14]} {p_a*100:5.1f}% ({fair_decimal_odds(p_a):.2f})")
    return p_h, p_d, p_a


def main():
    with APIFootballClient() as client:
        st = client.status()
        print(f"API-Football: {st['subscription']['plan']}, "
              f"used {st['requests']['current']}/{st['requests']['limit_day']}")
        print("\n=== Pulling club match data ===")
        matches = pull_matches(client)
        print(f"\nTotal matches: {len(matches)}")

    print("\n=== Fitting club Dixon-Coles ===")
    params = fit_club_dc(matches)
    print(f"  home advantage: {params['home_advantage']:+.3f}")
    print(f"  ρ:              {params['rho']:+.4f}")
    print(f"  league offsets:")
    for lg, off in sorted(params['league_offset'].items()):
        name = {39:"EPL",140:"La Liga",61:"Ligue 1",78:"Bundes",135:"Serie A",
                88:"Eredivisie",2:"UCL",3:"UEL",848:"UECL"}.get(lg, str(lg))
        print(f"    {name:10}  {off:+.3f}")

    # Show top 10 attack-rated teams
    rk_off = sorted(params['attack'].items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top 10 by attack:")
    for t, a in rk_off:
        d = params['defense'][t]
        print(f"    {t:30}  attack={a:+.2f}  defense={d:+.2f}")

    # Predict the three target finals
    targets = [
        ("Crystal Palace", "Rayo Vallecano", 848, "Conference League Final"),
        ("Paris Saint Germain", "Arsenal", 2, "Champions League Final"),
        ("SC Freiburg", "Aston Villa", 3, "Europa League Final"),
    ]
    print("\n=== Predictions ===")
    for h, a, lg, label in targets:
        print(f"\n  >>> {label}: {h} vs {a}")
        try:
            lh, la = predict_match(params, h, a, lg, neutral=True)
            report(h, a, lh, la, params['rho'])
        except ValueError as e:
            print(f"    SKIP — {e}")
            # find similar names
            for t in params["attack"]:
                if h.split()[0].lower() in t.lower() or a.split()[0].lower() in t.lower():
                    print(f"    candidate: {t!r}")


if __name__ == "__main__":
    main()
