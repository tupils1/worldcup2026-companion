"""Calibration backtest for Dixon-Coles.

Train/holdout split on historical international matches; compute log-loss,
Brier score, and top-1 accuracy on the 1X2 outcome. Used to (a) confirm
baseline calibration and (b) select `elo_prior_strength`.

Why these metrics (not raw accuracy):
    - log-loss / Brier penalize OVERCONFIDENCE — the failure mode that
      destroys Kelly-sized bankrolls.
    - A model with 65% accuracy and bad calibration is *worse* for betting
      than 55% accuracy with good calibration.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.models.dixon_coles import fit
from worldcup.models.markets import prob_1x2, score_matrix

HOSTS = {"USA", "CAN", "MEX"}


@dataclass(frozen=True)
class CalibrationResult:
    elo_prior_strength: float
    n_train: int
    n_test: int
    log_loss: float
    brier_score: float
    accuracy_top1: float
    home_advantage: float
    rho: float


def _outcome_1x2(home_score: int, away_score: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if home_score > away_score:
        return 0
    if home_score < away_score:
        return 2
    return 1


def evaluate(
    db_path: Path | str = DEFAULT_DB_PATH,
    train_since: str = "2014-01-01",
    train_until: str = "2024-12-31",
    test_until: str = "2026-05-27",
    elo_prior_strength: float = 0.0,
) -> CalibrationResult:
    """Fit on (train_since, train_until], evaluate on (train_until, test_until]."""
    params = fit(
        db_path=db_path,
        since=train_since,
        until=train_until,
        elo_prior_strength=elo_prior_strength,
        as_of=dt.date.fromisoformat(train_until),
    )

    conn = get_conn(db_path)
    try:
        test_rows = conn.execute(
            """
            SELECT home_code, away_code, home_score, away_score, neutral_venue
            FROM matches
            WHERE finished = 1
              AND home_score IS NOT NULL AND away_score IS NOT NULL
              AND match_date > ? AND match_date <= ?
            """,
            (train_until, test_until),
        ).fetchall()
    finally:
        conn.close()

    log_losses: list[float] = []
    briers: list[float] = []
    correct = 0
    n_used = 0
    for r in test_rows:
        home, away = r["home_code"], r["away_code"]
        if home not in params.attack or away not in params.attack:
            continue
        neutral = bool(r["neutral_venue"])
        if home in HOSTS:
            neutral = False
        lh, la = params.predict_lambda(home, away, neutral=neutral)
        M = score_matrix(lh, la, rho=params.rho)
        p_h, p_d, p_a = prob_1x2(M)
        probs = np.clip(np.array([p_h, p_d, p_a]), 1e-12, 1.0 - 1e-12)
        outcome = _outcome_1x2(r["home_score"], r["away_score"])
        log_losses.append(float(-np.log(probs[outcome])))
        y = np.zeros(3)
        y[outcome] = 1.0
        briers.append(float(np.sum((probs - y) ** 2)))
        if int(np.argmax(probs)) == outcome:
            correct += 1
        n_used += 1

    return CalibrationResult(
        elo_prior_strength=elo_prior_strength,
        n_train=params.n_matches,
        n_test=n_used,
        log_loss=float(np.mean(log_losses)) if n_used else float("nan"),
        brier_score=float(np.mean(briers)) if n_used else float("nan"),
        accuracy_top1=correct / n_used if n_used else float("nan"),
        home_advantage=params.home_advantage,
        rho=params.rho,
    )


def reliability_bins(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> list[tuple[float, float, float, int]]:
    """For each bin of predicted prob, return (bin_mid, mean_pred, mean_actual, n)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() > 0:
            out.append(
                (
                    float((lo + hi) / 2),
                    float(probs[mask].mean()),
                    float(outcomes[mask].mean()),
                    int(mask.sum()),
                )
            )
    return out


def main() -> None:
    """CLI: sweep prior strengths and pick the best one."""
    print("Calibration sweep: train 2014-2024, test 2025-2026")
    print(f"{'prior':>6}  {'n_tr':>5}  {'n_te':>4}  {'log_loss':>9}  {'brier':>7}  "
          f"{'top1_acc':>9}  {'h':>6}  {'ρ':>7}")
    print("-" * 78)
    best = None
    for strength in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
        r = evaluate(elo_prior_strength=strength)
        marker = ""
        if best is None or r.log_loss < best.log_loss:
            best = r
            marker = "  ← best"
        print(
            f"{r.elo_prior_strength:>6.1f}  {r.n_train:>5}  {r.n_test:>4}  "
            f"{r.log_loss:>9.4f}  {r.brier_score:>7.4f}  "
            f"{r.accuracy_top1:>9.3f}  {r.home_advantage:>+6.3f}  {r.rho:>+7.4f}{marker}"
        )
    print(
        f"\nBest by log-loss: prior_strength={best.elo_prior_strength}, "
        f"log_loss={best.log_loss:.4f}, brier={best.brier_score:.4f}, "
        f"acc={best.accuracy_top1:.3f}"
    )


if __name__ == "__main__":
    main()
