"""Dixon-Coles 1997 double-Poisson model for football match prediction.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market", JRSS-C.

Goal rates:
    log λ_home = α_home - β_away + h * 1[match not at neutral venue]
    log λ_away = α_away - β_home

Low-score correction τ(h, a) inflates/deflates (0,0), (1,0), (0,1), (1,1):
    τ(0,0) = 1 - λ_h λ_a ρ
    τ(1,0) = 1 + λ_a ρ
    τ(0,1) = 1 + λ_h ρ
    τ(1,1) = 1 - ρ
    (otherwise 1)

Time decay: each match contributes exp(-ξ · days_ago) to the log-likelihood,
so recent form dominates. ξ defaults to 0.0014 ≈ 1-year half-life — slower
than the 0.0065 from the original paper because international fixtures are
sparser than club seasons.

Identifiability: Σ α_i = 0 (else α and β can drift together).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

DEFAULT_DECAY_PER_DAY = math.log(2) / 365  # ≈ 0.0019, 1-year half-life

# Elo unit → log-rate scale: 100 Elo points ≈ 0.30 shift in log goal rate.
# Calibration: an Elo gap of 100 ≈ 64% win prob in 1v1, ≈ 0.6 log-odds shift,
# split between attack diff and defense diff (each ≈ 0.30).
DEFAULT_ELO_SCALE = 0.003

# Goal-rate calibration (added 2026-05-28 from Pinnacle/market O/U diagnostic).
# The DC fit + Elo prior + ridge compresses weak-team attack, under-predicting
# goals in mismatch games. Floor raises only weak λ (football's "even the worst
# team scores ~0.55 xG"); scale gently lifts overall rate to match market totals.
# Calibrated so median(model − market) P(over 2.5) ≈ 0 across 71 WC fixtures.
DEFAULT_LAM_FLOOR = 0.55
DEFAULT_LAM_SCALE = 1.03

# Competition-type weighting (added 2026-05-28). Friendlies are 39% of post-2014
# international matches and are played with experimental/rotated squads — they
# systematically DRAG DOWN big nations that rest stars (England's friendlies are
# the textbook case) and add noise everywhere. We down-weight them in the fit so
# competitive results dominate. Multiplies the time-decay weight per match.
COMPETITION_WEIGHTS: dict[str, float] = {
    "friendly": 0.40,       # experimental squads, low stakes
    "invitational": 0.50,   # Kirin / FIFA Series / Superclásico / Ashes etc.
    "qualifier": 0.80,      # competitive but uneven opposition strength
    "major": 1.00,          # World Cup / Euro / Copa / AFCON / Asian Cup / Nations League …
    "unknown": 0.70,
}


def competition_weight(competition: str | None) -> float:
    """Map a competition name to a fit weight in (0, 1]. Keyword-based so it is
    robust to the ~25 distinct competition strings in the DB. `qualifier` is
    checked before tournament keywords so 'World Cup qualification' → 0.80."""
    c = (competition or "").lower()
    if "friendly" in c:
        return COMPETITION_WEIGHTS["friendly"]
    if any(k in c for k in (
        "kirin", "fifa series", "superclás", "superclas", "ashes",
        "al ain", "canadian shield", "challenge cup",
    )):
        return COMPETITION_WEIGHTS["invitational"]
    if "qualif" in c:
        return COMPETITION_WEIGHTS["qualifier"]
    if any(k in c for k in (
        "world cup", "euro", "copa am", "african cup", "afcon", "asian cup",
        "gold cup", "nations league", "confederations", "arab cup", "gulf cup",
        "eaff", "waff", "championship",
    )):
        return COMPETITION_WEIGHTS["major"]
    return COMPETITION_WEIGHTS["unknown"]


@dataclass(frozen=True)
class DCParams:
    """Fitted Dixon-Coles parameters."""

    teams: tuple[str, ...]
    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    rho: float
    n_matches: int
    fit_loglik: float
    fit_converged: bool
    lam_floor: float = DEFAULT_LAM_FLOOR
    lam_scale: float = DEFAULT_LAM_SCALE

    def predict_lambda(
        self, home: str, away: str, neutral: bool = False
    ) -> tuple[float, float]:
        """Return (λ_home, λ_away) expected goals, with goal-rate calibration.

        Applies `lam_scale` (gentle overall lift) then `lam_floor` (weak-team
        minimum) — calibrated to match market O/U totals. Set both to identity
        (scale=1, floor=0) for raw uncalibrated rates.
        """
        ha = 0.0 if neutral else self.home_advantage
        log_lh = self.attack[home] - self.defense[away] + ha
        log_la = self.attack[away] - self.defense[home]
        lh = max(float(np.exp(log_lh)) * self.lam_scale, self.lam_floor)
        la = max(float(np.exp(log_la)) * self.lam_scale, self.lam_floor)
        return lh, la


def _load_matches(
    db_path: Path | str, since: str, until: str | None = None
) -> list[dict]:
    conn = get_conn(db_path)
    try:
        if until is None:
            cur = conn.execute(
                """
                SELECT home_code, away_code, match_date,
                       home_score, away_score, neutral_venue, competition
                FROM matches
                WHERE finished = 1
                  AND match_date >= ?
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                """,
                (since,),
            )
        else:
            cur = conn.execute(
                """
                SELECT home_code, away_code, match_date,
                       home_score, away_score, neutral_venue, competition
                FROM matches
                WHERE finished = 1
                  AND match_date >= ?
                  AND match_date <= ?
                  AND home_score IS NOT NULL
                  AND away_score IS NOT NULL
                """,
                (since, until),
            )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _dedup_twins(rows: list[dict]) -> list[dict]:
    """Drop duplicate rows for the SAME fixture double-entered by two feeds (martj42
    vs api_football) — identified by same unordered team pair, identical score, and
    kickoff within 2 days. Two genuinely distinct meetings of the same teams are weeks
    apart (and rarely the exact same score), so this can't merge real fixtures. Without
    it the fit double-weights those games; with the WC's near-full time-decay weight,
    a handful of duplicated blow-outs visibly inflate the recent goal environment.
    (Belt-and-suspenders: scripts/dedup_twins.py cleans the DB itself; this guards the
    fit even before that's run.)"""
    seen: list[tuple[frozenset, int, int, dt.date]] = []
    out: list[dict] = []
    for r in rows:
        try:
            d = dt.date.fromisoformat(r["match_date"][:10])
        except (ValueError, TypeError):
            out.append(r); continue
        pair = frozenset((r["home_code"], r["away_code"]))
        gtot, gdiff = r["home_score"] + r["away_score"], abs(r["home_score"] - r["away_score"])
        if any(p == pair and gt == gtot and gd == gdiff and abs((d - sd).days) <= 2
               for p, gt, gd, sd in seen):
            continue
        seen.append((pair, gtot, gdiff, d))
        out.append(r)
    return out


def _load_elo(db_path: Path | str) -> dict[str, float]:
    """team_code → current Elo (eloratings.net)."""
    conn = get_conn(db_path)
    try:
        rows = conn.execute(
            "SELECT team_code, value FROM team_ratings "
            "WHERE rating_type='elo' AND source='eloratings.net'"
        ).fetchall()
    finally:
        conn.close()
    return {r["team_code"]: float(r["value"]) for r in rows}


def _neg_log_likelihood(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    is_neutral: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    attack_prior: np.ndarray | None = None,
    defense_prior: np.ndarray | None = None,
    ridge_lambda: float | np.ndarray = 0.0,
) -> float:
    """Vectorized Dixon-Coles NLL with optional Elo-anchored ridge prior.

    `ridge_lambda > 0` adds  λ Σ_i [(α_i − α̂_i)² + (β_i − β̂_i)²]  where
    α̂_i and β̂_i come from the Elo-informed prior. Acts like a Bayesian
    posterior with Gaussian prior: weak-data teams shrink toward Elo, strong-
    data teams stay near their likelihood MLE.
    """
    attack = params[:n_teams]
    defense = params[n_teams : 2 * n_teams]
    home_adv = params[-2]
    rho = params[-1]

    ha = np.where(is_neutral, 0.0, home_adv)
    log_lh = attack[home_idx] - defense[away_idx] + ha
    log_la = attack[away_idx] - defense[home_idx]
    lh = np.exp(log_lh)
    la = np.exp(log_la)

    log_p_h = home_goals * log_lh - lh - gammaln(home_goals + 1.0)
    log_p_a = away_goals * log_la - la - gammaln(away_goals + 1.0)

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

    nll = -float(np.sum(weights * (log_p_h + log_p_a + log_tau)))

    # ridge_lambda may be a scalar (uniform prior) or a per-team array (adaptive
    # shrinkage: low-sample teams get a larger ridge → shrink toward Elo more).
    if attack_prior is not None and np.any(np.asarray(ridge_lambda) > 0.0):
        rv = np.asarray(ridge_lambda, dtype=np.float64)
        nll += float(
            np.sum(rv * ((attack - attack_prior) ** 2 + (defense - defense_prior) ** 2))
        )
    return nll


def fit(
    db_path: Path | str = DEFAULT_DB_PATH,
    since: str = "2014-01-01",
    until: str | None = None,
    decay_xi_per_day: float = DEFAULT_DECAY_PER_DAY,
    teams_filter: Iterable[str] | None = None,
    as_of: dt.date | None = None,
    elo_prior_strength: float = 0.0,
    elo_scale: float = DEFAULT_ELO_SCALE,
    competition_weighting: bool = True,
    adaptive_shrinkage: bool = False,  # tested at multiple strengths: degrades log-loss
    shrinkage_clip: tuple[float, float] = (0.5, 4.0),  # the uniform Elo-ridge already
    #                                       does sample-adaptive shrinkage (task #19);
    #                                       this extra layer over-corrects. Kept inert.
) -> DCParams:
    """Fit Dixon-Coles via MLE on matches in `matches` table since `since`.

    `teams_filter` (optional): only include matches where BOTH teams are in
    the given set. Useful for fitting on the WC-48 subset.
    `as_of`: anchor date for time decay (defaults to today).
    `elo_prior_strength` (λ): ridge weight for the Elo-anchored prior. 0 →
    classic MLE (no prior). 5-20 is a reasonable range — stronger values
    shrink weak-team estimates toward Elo more aggressively. Acts roughly
    like adding `λ` synthetic matches per team toward Elo expectation.
    `elo_scale`: conversion from Elo points to log-rate units.
    """
    rows = _load_matches(db_path=db_path, since=since, until=until)
    rows = _dedup_twins(rows)

    if teams_filter is not None:
        ts = set(teams_filter)
        rows = [r for r in rows if r["home_code"] in ts and r["away_code"] in ts]

    if not rows:
        raise ValueError("No matches available for the given filter.")

    teams = sorted({r["home_code"] for r in rows} | {r["away_code"] for r in rows})
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    anchor = as_of or dt.date.today()
    home_idx = np.fromiter((team_to_idx[r["home_code"]] for r in rows), dtype=np.int64)
    away_idx = np.fromiter((team_to_idx[r["away_code"]] for r in rows), dtype=np.int64)
    home_goals = np.fromiter((r["home_score"] for r in rows), dtype=np.float64)
    away_goals = np.fromiter((r["away_score"] for r in rows), dtype=np.float64)
    is_neutral = np.fromiter(
        (bool(r["neutral_venue"]) for r in rows), dtype=bool, count=len(rows)
    )
    days_ago = np.fromiter(
        (
            (anchor - dt.date.fromisoformat(r["match_date"][:10])).days
            for r in rows
        ),
        dtype=np.float64,
    )
    weights = np.exp(-decay_xi_per_day * days_ago)

    # Competition weighting: down-weight friendlies / invitationals so competitive
    # results dominate (helps big nations that rest stars in friendlies).
    if competition_weighting:
        comp_w = np.fromiter(
            (competition_weight(r.get("competition")) for r in rows),
            dtype=np.float64, count=len(rows),
        )
        weights = weights * comp_w

    # Build Elo-anchored prior (optional)
    attack_prior: np.ndarray | None = None
    defense_prior: np.ndarray | None = None
    if elo_prior_strength > 0.0:
        elo_map = _load_elo(db_path)
        if elo_map:
            elos = np.array(
                [elo_map.get(t, np.nan) for t in teams], dtype=np.float64
            )
            mean_elo = float(np.nanmean(elos))
            elos = np.where(np.isnan(elos), mean_elo, elos)
            centered = elos - mean_elo
            attack_prior = elo_scale * centered    # stronger team → higher attack prior
            defense_prior = -elo_scale * centered  # stronger team → lower (better) defense prior

    # Ridge strength: scalar by default. With adaptive_shrinkage, make it per-team —
    # teams with little EFFECTIVE data (few / old / friendly-heavy matches) get a
    # larger ridge and shrink toward the Elo prior more, cutting minnow over-fit
    # (CUW/HAI/CPV). Normalised so a median-sample team keeps the base strength.
    ridge_param: float | np.ndarray = elo_prior_strength
    if adaptive_shrinkage and elo_prior_strength > 0.0 and attack_prior is not None:
        eff_n = np.zeros(n, dtype=np.float64)
        np.add.at(eff_n, home_idx, weights)
        np.add.at(eff_n, away_idx, weights)
        ref = float(np.median(eff_n[eff_n > 0])) if np.any(eff_n > 0) else 1.0
        lo, hi = shrinkage_clip
        ridge_param = elo_prior_strength * np.clip(ref / np.maximum(eff_n, 1e-6), lo, hi)

    # Initial params: zeros (or Elo prior if available), modest home advantage.
    if attack_prior is not None:
        x0 = np.concatenate([attack_prior, defense_prior, [0.3], [-0.05]])
    else:
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.3], [-0.05]])

    # Σ α_i = 0 for identifiability
    constraints = [{"type": "eq", "fun": lambda p, n=n: float(np.sum(p[:n]))}]
    bounds = [(-3.0, 3.0)] * (2 * n) + [(0.0, 1.0), (-0.2, 0.2)]

    res = minimize(
        _neg_log_likelihood,
        x0,
        args=(
            home_idx, away_idx, home_goals, away_goals, is_neutral, weights, n,
            attack_prior, defense_prior, ridge_param,
        ),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-7, "disp": False},
    )

    return DCParams(
        teams=tuple(teams),
        attack=dict(zip(teams, res.x[:n].tolist())),
        defense=dict(zip(teams, res.x[n : 2 * n].tolist())),
        home_advantage=float(res.x[-2]),
        rho=float(res.x[-1]),
        n_matches=len(rows),
        fit_loglik=-float(res.fun),
        fit_converged=bool(res.success),
    )


def main() -> None:
    """CLI: fit and print a sanity report."""
    import sqlite3

    print("Fitting Dixon-Coles on matches since 2014-01-01 ...")
    params = fit(since="2014-01-01")
    print(f"  matches used:      {params.n_matches}")
    print(f"  teams in fit:      {len(params.teams)}")
    print(f"  converged:         {params.fit_converged}")
    print(f"  home advantage h:  {params.home_advantage:+.3f}  "
          f"(≈ {math.exp(params.home_advantage):.2f}× scoring rate at home)")
    print(f"  rho:               {params.rho:+.4f}  (DC low-score correction)")
    print(f"  log-likelihood:    {params.fit_loglik:,.0f}")

    # Top / bottom WC teams by attack
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    wc_codes = [
        r["code"]
        for r in conn.execute(
            "SELECT code FROM teams WHERE in_worldcup_2026 = 1 ORDER BY code"
        )
    ]
    rankings = sorted(
        (
            (params.attack.get(c, float("nan")), -params.defense.get(c, float("nan")), c)
            for c in wc_codes
            if c in params.attack
        ),
        reverse=True,
    )
    print("\n=== WC 48: top 12 by attack strength ===")
    print(f"{'rk':>3}  {'team':4}  {'attack':>7}  {'defense':>7}  {'net':>6}")
    for i, (a, neg_d, t) in enumerate(rankings[:12], 1):
        d = -neg_d
        print(f"{i:3d}  {t:4}  {a:+7.3f}  {d:+7.3f}  {a - d:+6.3f}")
    print("\n=== WC 48: bottom 5 by attack strength ===")
    for i, (a, neg_d, t) in enumerate(rankings[-5:], len(rankings) - 4):
        d = -neg_d
        print(f"{i:3d}  {t:4}  {a:+7.3f}  {d:+7.3f}  {a - d:+6.3f}")

    # Predict λ for the first 8 WC 2026 fixtures in DB
    print("\n=== Expected goals (λ_h, λ_a) for opening WC 2026 matches ===")
    print("  date        host  away  λ_home  λ_away  venue")
    fixtures = conn.execute(
        """
        SELECT match_date, home_code, away_code, neutral_venue, venue
        FROM matches
        WHERE finished = 0 AND match_date >= '2026-06-11'
        ORDER BY match_date LIMIT 12
        """
    ).fetchall()
    for r in fixtures:
        home, away = r["home_code"], r["away_code"]
        if home not in params.attack or away not in params.attack:
            continue
        # Hosts at home country aren't neutral. martj42 marks WC games as neutral
        # but USA/CAN/MEX playing in their own country is effectively home.
        neutral = bool(r["neutral_venue"])
        if home in ("USA", "CAN", "MEX"):
            neutral = False
        lh, la = params.predict_lambda(home, away, neutral=neutral)
        print(
            f"  {r['match_date'][:10]}  {home:4}  {away:4}  "
            f"{lh:6.2f}  {la:6.2f}  {r['venue']}"
        )
    conn.close()


if __name__ == "__main__":
    main()
