"""Value bet detection: model probability vs de-vigged bookmaker odds.

Pipeline:
    1. Fit Dixon-Coles (with calibrated Elo prior).
    2. Load latest odds snapshots from `odds` table.
    3. For each (match, bookmaker, market, line) tuple:
       a) De-vig the bookmaker's quote (Pinnacle proportional or Shin).
       b) Compute model probability for each selection.
       c) Edge = model_prob − market_implied_prob.
       d) If edge ≥ threshold, emit a ValueBet with Kelly-1/4 sizing.

Markets covered:
    - 1X2 (h2h)
    - Asian handicap (spreads) — both half and integer lines
    - Over/Under (totals)
    - Outright winner (compared to Monte Carlo champ probabilities)

Sizing: default Kelly-1/4 for survival under model uncertainty. See
[[modeling-principles]] rule 4.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.models.dixon_coles import DCParams, fit
from worldcup.models.hybrid import hybrid_score_matrix, load_elo_for_dc
from worldcup.models.markets import (
    prob_1x2,
    prob_asian_handicap,
    prob_over_under,
    score_matrix,
)

HOSTS = frozenset({"USA", "CAN", "MEX"})


# ─────────────────────── de-vigging ───────────────────────

def devig_proportional(prices: list[float]) -> list[float]:
    """Naive de-vig: P_i = (1/o_i) / Σ(1/o_j). Fast, slight bias at extremes."""
    inv = np.array([1.0 / p for p in prices], dtype=np.float64)
    return list(inv / inv.sum())


def devig_shin(prices: list[float], max_iter: int = 100, tol: float = 1e-9) -> list[float]:
    """Shin (1992) de-vig: assumes informed-trader fraction z, more accurate
    than proportional for skewed lines (favorites/longshots).

    Solves: π_i = (√(z² + 4·(1-z)·s_i²/Σs_j) − z) / (2(1-z))
            with Σπ_i = 1, where s_i = 1/o_i.
    """
    s = np.array([1.0 / p for p in prices], dtype=np.float64)
    total = s.sum()
    if total <= 1.0:
        return list(s / total)  # no overround → just normalise
    z = (total - 1.0) / len(s)  # initial guess
    for _ in range(max_iter):
        disc = z * z + 4.0 * (1.0 - z) * s * s / total
        pi = (np.sqrt(disc) - z) / (2.0 * (1.0 - z))
        new_sum = pi.sum()
        if abs(new_sum - 1.0) < tol:
            return list(pi / new_sum)
        # Pull z toward making sum=1
        z *= new_sum
        z = float(np.clip(z, 1e-8, 0.5))
    return list(pi / pi.sum())


# ─────────────────────── Kelly sizing ───────────────────────

def kelly_full(p: float, decimal_odds: float) -> float:
    """Optimal Kelly fraction. Returns 0 if no edge."""
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    f = (p * b - (1.0 - p)) / b
    return max(0.0, f)


def kelly_fraction(p: float, decimal_odds: float, scaling: float = 0.25) -> float:
    """Scaled Kelly (default 1/4) for survival-safe sizing."""
    return scaling * kelly_full(p, decimal_odds)


# ─────────────────────── ValueBet ───────────────────────

@dataclass
class ValueBet:
    match_id: int | None
    match_date: str | None
    home_code: str | None
    away_code: str | None
    market_scope: str          # 'match' | 'outright'
    bookmaker: str
    market: str                # '1X2' | 'AH' | 'OU' | 'winner'
    selection: str             # 'home' | 'away' | 'draw' | 'over' | 'under' | <team_code>
    line: float | None
    market_price: float
    model_prob: float
    market_implied_prob: float
    edge_pct: float            # model_prob − market_implied_prob
    fair_price: float          # 1 / model_prob
    kelly_frac: float          # fractional Kelly (default 1/4)
    captured_at: str           # bookmaker quote timestamp

    @property
    def expected_value(self) -> float:
        return self.model_prob * self.market_price - 1.0

    def stake(self, bankroll: float) -> float:
        return bankroll * self.kelly_frac


# ─────────────────────── Helpers ───────────────────────

def _latest_match_odds(conn: sqlite3.Connection) -> list[dict]:
    """One row per (match, bookmaker, market, line, selection): the latest snapshot."""
    cur = conn.execute(
        """
        SELECT o.* FROM odds o
        JOIN (
            SELECT match_id, bookmaker, market, COALESCE(line, -999.0) AS line_k,
                   selection, MAX(captured_at) AS mc
            FROM odds WHERE market_scope = 'match'
            GROUP BY match_id, bookmaker, market, COALESCE(line, -999.0), selection
        ) latest ON o.match_id     = latest.match_id
                AND o.bookmaker    = latest.bookmaker
                AND o.market       = latest.market
                AND COALESCE(o.line, -999.0) = latest.line_k
                AND o.selection    = latest.selection
                AND o.captured_at  = latest.mc
        WHERE o.market_scope = 'match'
        """
    )
    return [dict(r) for r in cur.fetchall()]


def _latest_outright_odds(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(
        """
        SELECT o.* FROM odds o
        JOIN (
            SELECT bookmaker, market, subject_code, MAX(captured_at) AS mc
            FROM odds WHERE market_scope = 'outright'
            GROUP BY bookmaker, market, subject_code
        ) latest ON o.bookmaker    = latest.bookmaker
                AND o.market       = latest.market
                AND o.subject_code = latest.subject_code
                AND o.captured_at  = latest.mc
        WHERE o.market_scope = 'outright'
        """
    )
    return [dict(r) for r in cur.fetchall()]


def _match_meta(conn: sqlite3.Connection, match_ids: Iterable[int]) -> dict[int, dict]:
    ids = list({i for i in match_ids if i is not None})
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    cur = conn.execute(
        f"SELECT id, home_code, away_code, match_date, neutral_venue, finished "
        f"FROM matches WHERE id IN ({placeholders})",
        ids,
    )
    return {r["id"]: dict(r) for r in cur.fetchall()}


# ─────────────────────── Main detectors ───────────────────────

def detect_match_value_bets(
    params: DCParams,
    db_path: Path | str = DEFAULT_DB_PATH,
    min_edge: float = 0.03,
    devig: str = "shin",
    kelly_scaling: float = 0.25,
    skip_finished: bool = True,
    dc_weight: float = 1.0,
    elo: dict[str, float] | None = None,
) -> list[ValueBet]:
    """Scan latest match odds; return bets with edge ≥ `min_edge`.

    `dc_weight < 1.0` blends DC with Elo-derived probabilities — useful to
    smooth weak-team overfitting (e.g. AUS-TUR baseline edge of +40%).
    Set `elo` to the per-team rating dict; if None and dc_weight<1, falls
    back to pure DC.
    """
    conn = get_conn(db_path)
    try:
        odds_rows = _latest_match_odds(conn)
        meta = _match_meta(conn, (r["match_id"] for r in odds_rows))
    finally:
        conn.close()

    # Group by (match_id, bookmaker, market, line). All selections needed
    # together for de-vigging within a market.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in odds_rows:
        key = (r["match_id"], r["bookmaker"], r["market"], r["line"])
        groups[key].append(r)

    devig_fn = devig_shin if devig == "shin" else devig_proportional
    bets: list[ValueBet] = []

    for (match_id, bookmaker, market, line), rows in groups.items():
        m = meta.get(match_id)
        if not m:
            continue
        if skip_finished and m["finished"]:
            continue
        if m["home_code"] not in params.attack or m["away_code"] not in params.attack:
            continue
        neutral = bool(m["neutral_venue"])
        if m["home_code"] in HOSTS:
            neutral = False

        lh, la = params.predict_lambda(m["home_code"], m["away_code"], neutral=neutral)
        if dc_weight < 1.0 and elo is not None:
            M = hybrid_score_matrix(
                params, elo, m["home_code"], m["away_code"],
                dc_weight=dc_weight, neutral=neutral,
            )
        else:
            M = score_matrix(lh, la, rho=params.rho)
        captured_at = rows[0]["captured_at"]

        if market == "1X2":
            by_sel = {r["selection"]: r["price"] for r in rows}
            if not all(s in by_sel for s in ("home", "draw", "away")):
                continue
            prices = [by_sel["home"], by_sel["draw"], by_sel["away"]]
            implied = devig_fn(prices)
            p_h, p_d, p_a = prob_1x2(M)
            for sel, mp, ip, price in zip(
                ("home", "draw", "away"), (p_h, p_d, p_a), implied, prices
            ):
                edge = mp - ip
                if edge < min_edge:
                    continue
                bets.append(
                    ValueBet(
                        match_id=match_id,
                        match_date=m["match_date"],
                        home_code=m["home_code"],
                        away_code=m["away_code"],
                        market_scope="match",
                        bookmaker=bookmaker,
                        market="1X2",
                        selection=sel,
                        line=None,
                        market_price=price,
                        model_prob=mp,
                        market_implied_prob=ip,
                        edge_pct=edge,
                        fair_price=1.0 / mp if mp > 0 else float("inf"),
                        kelly_frac=kelly_fraction(mp, price, kelly_scaling),
                        captured_at=captured_at,
                    )
                )

        elif market == "OU":
            by_sel = {r["selection"]: r for r in rows}
            if "over" not in by_sel or "under" not in by_sel:
                continue
            prices = [by_sel["over"]["price"], by_sel["under"]["price"]]
            implied = devig_fn(prices)
            p_over, _push, p_under = prob_over_under(M, line)
            # Strip push from model probs for fair compare on no-push books;
            # for integer lines with push handled by book, this slightly distorts.
            denom = p_over + p_under
            if denom > 0 and abs(denom - 1.0) > 1e-6:
                p_over, p_under = p_over / denom, p_under / denom
            for sel, mp, ip, price in zip(
                ("over", "under"), (p_over, p_under), implied, prices
            ):
                edge = mp - ip
                if edge < min_edge:
                    continue
                bets.append(
                    ValueBet(
                        match_id=match_id,
                        match_date=m["match_date"],
                        home_code=m["home_code"],
                        away_code=m["away_code"],
                        market_scope="match",
                        bookmaker=bookmaker,
                        market="OU",
                        selection=sel,
                        line=line,
                        market_price=price,
                        model_prob=mp,
                        market_implied_prob=ip,
                        edge_pct=edge,
                        fair_price=1.0 / mp if mp > 0 else float("inf"),
                        kelly_frac=kelly_fraction(mp, price, kelly_scaling),
                        captured_at=captured_at,
                    )
                )

        elif market == "AH":
            # Each side stored with its own "point" perspective from the API.
            # For the home selection, line is the home handicap directly.
            # For the away selection, line is the away handicap (= −home_handicap).
            by_sel = {r["selection"]: r for r in rows}
            if "home" not in by_sel or "away" not in by_sel:
                continue
            home_row = by_sel["home"]
            away_row = by_sel["away"]
            # Convert away line to home POV. Spreads books quote symmetric:
            # home -1.5 / away +1.5. Both lines should round-trip.
            home_handicap = home_row["line"]
            if home_handicap is None:
                continue
            prices = [home_row["price"], away_row["price"]]
            implied = devig_fn(prices)
            p_home_cov, p_push, p_away_cov = prob_asian_handicap(M, home_handicap)
            # Strip push (books that don't refund push); harmless for half-lines.
            denom = p_home_cov + p_away_cov
            if denom > 0:
                p_home_cov, p_away_cov = p_home_cov / denom, p_away_cov / denom
            for sel, mp, ip, price in zip(
                ("home", "away"), (p_home_cov, p_away_cov), implied, prices
            ):
                edge = mp - ip
                if edge < min_edge:
                    continue
                bets.append(
                    ValueBet(
                        match_id=match_id,
                        match_date=m["match_date"],
                        home_code=m["home_code"],
                        away_code=m["away_code"],
                        market_scope="match",
                        bookmaker=bookmaker,
                        market="AH",
                        selection=sel,
                        line=(
                            home_handicap if sel == "home" else -home_handicap
                        ),
                        market_price=price,
                        model_prob=mp,
                        market_implied_prob=ip,
                        edge_pct=edge,
                        fair_price=1.0 / mp if mp > 0 else float("inf"),
                        kelly_frac=kelly_fraction(mp, price, kelly_scaling),
                        captured_at=captured_at,
                    )
                )

    bets.sort(key=lambda v: -v.edge_pct)
    return bets


def detect_outright_value_bets(
    champion_probs: dict[str, float],
    db_path: Path | str = DEFAULT_DB_PATH,
    min_edge: float = 0.02,
    devig: str = "shin",
    kelly_scaling: float = 0.25,
) -> list[ValueBet]:
    """Compare a {team_code: P(champion)} dict (e.g. from MC) to outright odds."""
    conn = get_conn(db_path)
    try:
        rows = _latest_outright_odds(conn)
    finally:
        conn.close()

    devig_fn = devig_shin if devig == "shin" else devig_proportional
    by_book: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_book[r["bookmaker"]].append(r)

    bets: list[ValueBet] = []
    for book, entries in by_book.items():
        prices = [r["price"] for r in entries]
        teams = [r["subject_code"] for r in entries]
        implied = devig_fn(prices)
        for team, price, ip in zip(teams, prices, implied):
            mp = champion_probs.get(team, 0.0)
            if mp <= 0:
                continue
            edge = mp - ip
            if edge < min_edge:
                continue
            bets.append(
                ValueBet(
                    match_id=None,
                    match_date=None,
                    home_code=None,
                    away_code=None,
                    market_scope="outright",
                    bookmaker=book,
                    market="winner",
                    selection=team,
                    line=None,
                    market_price=price,
                    model_prob=mp,
                    market_implied_prob=ip,
                    edge_pct=edge,
                    fair_price=1.0 / mp,
                    kelly_frac=kelly_fraction(mp, price, kelly_scaling),
                    captured_at=entries[0]["captured_at"],
                )
            )
    bets.sort(key=lambda v: -v.edge_pct)
    return bets


def _print_bets(bets: list[ValueBet], bankroll: float, max_rows: int) -> None:
    if not bets:
        print("  No value bets found.")
        return
    print(
        f"  {'rk':>3}  {'date':10}  {'match':<10}  {'bookie':<14} {'mkt':<4} "
        f"{'sel':<6} {'line':>6}  {'price':>5}  {'model%':>6}  {'mkt%':>6}  "
        f"{'edge':>5}  {'EV%':>5}  {'bet $':>7}"
    )
    for i, b in enumerate(bets[:max_rows], 1):
        m_str = (
            f"{b.home_code}-{b.away_code}"
            if b.home_code
            else b.selection
        )
        line_str = f"{b.line:+.2f}" if b.line is not None else ""
        print(
            f"  {i:3d}  {b.match_date or '(outright)':<10}  {m_str:<10}  "
            f"{b.bookmaker:<14} {b.market:<4} {b.selection:<6} "
            f"{line_str:>6}  {b.market_price:>5.2f}  "
            f"{b.model_prob*100:>5.1f}%  {b.market_implied_prob*100:>5.1f}%  "
            f"{b.edge_pct*100:>+4.1f}%  {b.expected_value*100:>+4.1f}%  "
            f"${b.stake(bankroll):>6.2f}"
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prior", type=float, default=0.5, help="Elo prior strength")
    ap.add_argument("--min-edge", type=float, default=0.03, help="Min edge fraction")
    ap.add_argument("--devig", choices=("shin", "proportional"), default="shin")
    ap.add_argument("--kelly-scale", type=float, default=0.25, help="Kelly fraction multiplier")
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--mode", choices=("match", "outright", "both"), default="both"
    )
    ap.add_argument(
        "--mc-sims", type=int, default=20_000,
        help="N MC simulations for outright champ probs (only if mode includes outright)",
    )
    ap.add_argument(
        "--dc-weight", type=float, default=0.5,
        help="DC ⊕ Elo mix: 1.0=pure DC, 0.0=pure Elo, 0.5=equal blend (default; tames weak-team overfit)",
    )
    args = ap.parse_args()

    print(f"Fitting Dixon-Coles (Elo prior λ={args.prior}) ...")
    params = fit(elo_prior_strength=args.prior)
    print(
        f"  h={params.home_advantage:+.3f}, ρ={params.rho:+.3f}, "
        f"matches={params.n_matches}, log-lik={params.fit_loglik:.0f}"
    )

    elo = load_elo_for_dc(params, DEFAULT_DB_PATH) if args.dc_weight < 1.0 else None
    if args.mode in ("match", "both"):
        print(
            f"\n=== Match-level value bets (edge ≥ {args.min_edge:.0%}, "
            f"devig={args.devig}, Kelly×{args.kelly_scale}, "
            f"DC weight={args.dc_weight}) ==="
        )
        bets = detect_match_value_bets(
            params,
            min_edge=args.min_edge,
            devig=args.devig,
            kelly_scaling=args.kelly_scale,
            dc_weight=args.dc_weight,
            elo=elo,
        )
        print(f"  Found {len(bets)} value bets")
        _print_bets(bets, args.bankroll, args.top)

    if args.mode in ("outright", "both"):
        print("\n=== Outright value bets ===")
        from worldcup.simulator.monte_carlo import monte_carlo, ROUND_WIN

        print(f"  Running {args.mc_sims:,} MC sims for champ probabilities ...")
        mc = monte_carlo(
            n_sims=args.mc_sims,
            elo_prior_strength=args.prior,
            seed=42,
            verbose=False,
        )
        champ = {t: p[ROUND_WIN] for t, p in mc["probabilities"].items()}
        bets = detect_outright_value_bets(
            champ,
            min_edge=args.min_edge,
            devig=args.devig,
            kelly_scaling=args.kelly_scale,
        )
        print(f"  Found {len(bets)} outright value bets")
        _print_bets(bets, args.bankroll, args.top)


if __name__ == "__main__":
    main()
