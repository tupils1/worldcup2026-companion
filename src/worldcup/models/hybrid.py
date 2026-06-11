"""Hybrid Dixon-Coles ⊕ Elo predictor.

Why hybrid:
    Pure DC overfits on weak-team small samples (e.g. AUS scoring big on weak
    Asian opponents inflates its attack rating). Elo is more conservative because
    it accumulates slowly across all 200+ international teams. Blending the two
    score matrices smooths model overconfidence while preserving DC's recent-form
    signal where it actually has data.

Recipe:
    M_DC  = score_matrix(λ_h^DC,  λ_a^DC,  ρ_DC)
    M_Elo = score_matrix(λ_h^Elo, λ_a^Elo, ρ=0)   # no τ correction on Elo
    M     = w · M_DC + (1 - w) · M_Elo            # default w = 0.5

Elo → λ derivation (symmetric around the league mean):
    λ_h = base_rate · exp(elo_scale · (R_home − R_away) + home_adv · 1[not neutral])
    λ_a = base_rate · exp(elo_scale · (R_away − R_home))

base_rate ≈ 1.3 goals/team (international-match historical mean).
elo_scale = 0.003 ≈ same scale used in DC's Elo prior (100 Elo ≈ 0.30 log-rate).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from worldcup.models.dixon_coles import DEFAULT_ELO_SCALE, DCParams
from worldcup.models.markets import score_matrix

DEFAULT_BASE_RATE = 1.3  # international goals/team/match historical mean


def elo_score_matrix(
    home_elo: float,
    away_elo: float,
    home_advantage: float,
    neutral: bool = False,
    base_rate: float = DEFAULT_BASE_RATE,
    elo_scale: float = DEFAULT_ELO_SCALE,
    max_goals: int = 10,
) -> np.ndarray:
    """Elo-only score-probability matrix (no Dixon-Coles τ)."""
    ha = 0.0 if neutral else home_advantage
    diff = home_elo - away_elo
    lh = base_rate * math.exp(elo_scale * diff + ha)
    la = base_rate * math.exp(-elo_scale * diff)
    return score_matrix(lh, la, rho=0.0, max_goals=max_goals)


def hybrid_score_matrix(
    dc: DCParams,
    elo: dict[str, float],
    home: str,
    away: str,
    dc_weight: float = 0.5,
    neutral: bool = False,
    base_rate: float = DEFAULT_BASE_RATE,
    elo_scale: float = DEFAULT_ELO_SCALE,
    max_goals: int = 10,
) -> np.ndarray:
    """Blend DC and Elo score matrices.

    `dc_weight=1.0` → pure DC. `0.0` → pure Elo. `0.5` → equal mix.
    """
    if not (0.0 <= dc_weight <= 1.0):
        raise ValueError(f"dc_weight must be in [0, 1], got {dc_weight}")

    # DC component
    lh_dc, la_dc = dc.predict_lambda(home, away, neutral=neutral)
    M_dc = score_matrix(lh_dc, la_dc, rho=dc.rho, max_goals=max_goals)

    if dc_weight >= 1.0 - 1e-9:
        return M_dc

    # Elo component
    home_elo = elo.get(home)
    away_elo = elo.get(away)
    if home_elo is None or away_elo is None:
        # Fall back to pure DC if Elo missing
        return M_dc

    M_elo = elo_score_matrix(
        home_elo,
        away_elo,
        home_advantage=dc.home_advantage,
        neutral=neutral,
        base_rate=base_rate,
        elo_scale=elo_scale,
        max_goals=max_goals,
    )

    if dc_weight <= 1e-9:
        return M_elo

    M = dc_weight * M_dc + (1.0 - dc_weight) * M_elo
    # Both matrices are normalized; convex combination preserves normalization.
    return M


def load_elo_for_dc(dc: DCParams, db_path) -> dict[str, float]:
    """Convenience: load Elo for every team in the DC fit. Returns {team: elo}."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT team_code, value FROM team_ratings "
            "WHERE rating_type='elo' AND source='eloratings.net'"
        ).fetchall()
    finally:
        conn.close()
    elo = {code: float(v) for code, v in rows}
    return {t: elo[t] for t in dc.teams if t in elo}
