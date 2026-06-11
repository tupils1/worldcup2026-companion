"""Convert Dixon-Coles goal-rate predictions into market probabilities.

Input: per-match (λ_home, λ_away, ρ) from `DCParams.predict_lambda`.
Output: probabilities for 1X2 / Asian Handicap / Over-Under / BTTS / scoreline.

All conversions go through the same intermediate: a (max_goals+1)² probability
matrix M[h, a] = P(home scores h, away scores a) under independent Poisson
with the Dixon-Coles low-score correction τ applied (then renormalized).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def score_matrix(
    lambda_h: float,
    lambda_a: float,
    rho: float = 0.0,
    max_goals: int = 10,
) -> np.ndarray:
    """P(home=i, away=j) with Dixon-Coles low-score correction.

    The τ correction is applied to the (0,0), (1,0), (0,1), (1,1) cells then
    the whole matrix is renormalized — τ doesn't preserve total mass exactly
    so a renorm is mandatory to keep things a proper distribution.
    """
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g, lambda_h)
    pa = poisson.pmf(g, lambda_a)
    M = np.outer(ph, pa)
    M[0, 0] *= 1.0 - lambda_h * lambda_a * rho
    M[1, 0] *= 1.0 + lambda_a * rho
    M[0, 1] *= 1.0 + lambda_h * rho
    M[1, 1] *= 1.0 - rho
    M /= M.sum()
    return M


def prob_1x2(M: np.ndarray) -> tuple[float, float, float]:
    """Return (home_win, draw, away_win)."""
    p_home = float(np.tril(M, k=-1).sum())
    p_draw = float(np.diag(M).sum())
    p_away = float(np.triu(M, k=1).sum())
    return p_home, p_draw, p_away


def prob_over_under(M: np.ndarray, line: float) -> tuple[float, float, float]:
    """Total-goals market. Returns (over, push, under).

    `line` = 2.5 → no push, P(total ≥ 3) vs P(total ≤ 2).
    `line` = 3 (integer) → push when total == 3.
    """
    max_g = M.shape[0]
    total = np.add.outer(np.arange(max_g), np.arange(max_g))
    if abs(line - round(line)) < 1e-9:
        ln = int(round(line))
        p_over = float(M[total > ln].sum())
        p_push = float(M[total == ln].sum())
        p_under = float(M[total < ln].sum())
        return p_over, p_push, p_under
    p_over = float(M[total > line].sum())
    return p_over, 0.0, 1.0 - p_over


def goal_margin_dist(M: np.ndarray) -> dict[int, float]:
    """P(home_goals − away_goals = k) for each k ∈ [−max_g, +max_g]."""
    max_g = M.shape[0]
    margins: dict[int, float] = {}
    for h in range(max_g):
        for a in range(max_g):
            k = h - a
            margins[k] = margins.get(k, 0.0) + float(M[h, a])
    return margins


def prob_asian_handicap(
    M: np.ndarray, home_handicap: float
) -> tuple[float, float, float]:
    """Asian Handicap. `home_handicap = -1.5` means home gives 1.5 goals.

    Returns (home_covers, push, away_covers).
    Handles integer, half-step, and quarter (0.25 / 0.75) lines by splitting
    quarter lines into two half-stake half-line bets.
    """
    q4 = home_handicap * 4
    if abs(q4 - round(q4)) > 1e-9:
        raise ValueError(f"Handicap {home_handicap} must be a multiple of 0.25")
    if int(round(q4)) % 2 != 0:
        # Quarter handicap: average of two adjacent half-step bets
        h1 = home_handicap - 0.25
        h2 = home_handicap + 0.25
        w1, p1, l1 = prob_asian_handicap(M, h1)
        w2, p2, l2 = prob_asian_handicap(M, h2)
        return ((w1 + w2) / 2, (p1 + p2) / 2, (l1 + l2) / 2)

    is_integer = abs(home_handicap - round(home_handicap)) < 1e-9
    margins = goal_margin_dist(M)
    p_win, p_push, p_lose = 0.0, 0.0, 0.0
    for k, p in margins.items():
        adj = k + home_handicap
        if is_integer:
            if adj > 0:
                p_win += p
            elif adj == 0:
                p_push += p
            else:
                p_lose += p
        else:  # half-step → no push
            if adj > 0:
                p_win += p
            else:
                p_lose += p
    return p_win, p_push, p_lose


def prob_btts(M: np.ndarray) -> float:
    """Probability both teams score at least once."""
    return float(M[1:, 1:].sum())


def most_likely_score(M: np.ndarray) -> tuple[int, int, float]:
    """Modal scoreline."""
    h, a = np.unravel_index(int(np.argmax(M)), M.shape)
    return int(h), int(a), float(M[h, a])


def fair_decimal_odds(prob: float) -> float:
    """No-vig fair price corresponding to a probability."""
    return float("inf") if prob <= 0 else 1.0 / prob
