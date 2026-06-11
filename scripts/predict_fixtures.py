#!/usr/bin/env python
"""End-to-end demo: fit Dixon-Coles + print full market probabilities
for the upcoming 2026 World Cup fixtures.

Usage:
    PYTHONPATH=src python scripts/predict_fixtures.py [--limit N] [--since YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.models.dixon_coles import fit
from worldcup.models.markets import (
    fair_decimal_odds,
    most_likely_score,
    prob_1x2,
    prob_asian_handicap,
    prob_btts,
    prob_over_under,
    score_matrix,
)

HOSTS = {"USA", "CAN", "MEX"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--since", default="2014-01-01", help="fit window start")
    ap.add_argument(
        "--prior", type=float, default=0.5,
        help="Elo prior strength (0=classic DC; 0.5=calibration-best per holdout 2025-2026)",
    )
    args = ap.parse_args()

    print(f"Fitting Dixon-Coles on matches since {args.since} (Elo prior λ={args.prior}) ...")
    params = fit(db_path=args.db, since=args.since, elo_prior_strength=args.prior)
    print(
        f"  {params.n_matches} matches, "
        f"h={params.home_advantage:+.3f}, ρ={params.rho:+.3f}, "
        f"log-lik={params.fit_loglik:.0f}\n"
    )

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    fixtures = conn.execute(
        """
        SELECT match_date, home_code, away_code, neutral_venue, venue
        FROM matches
        WHERE finished = 0 AND match_date >= '2026-06-11'
        ORDER BY match_date LIMIT ?
        """,
        (args.limit,),
    ).fetchall()

    for f in fixtures:
        home, away = f["home_code"], f["away_code"]
        if home not in params.attack or away not in params.attack:
            print(f"  SKIP {home}-{away}: not in model (no recent matches)")
            continue
        neutral = bool(f["neutral_venue"])
        if home in HOSTS:
            neutral = False

        lh, la = params.predict_lambda(home, away, neutral=neutral)
        M = score_matrix(lh, la, rho=params.rho)

        p_h, p_d, p_a = prob_1x2(M)
        ah_05 = prob_asian_handicap(M, -0.5)
        ah_15 = prob_asian_handicap(M, -1.5)
        ou25 = prob_over_under(M, 2.5)
        ou35 = prob_over_under(M, 3.5)
        p_btts = prob_btts(M)
        mh, ma, p_score = most_likely_score(M)

        flag = " (neutral)" if neutral else ""
        print(f"┌── {f['match_date'][:10]}  {home} vs {away}{flag}  @ {f['venue']}")
        print(
            f"│   λ_h = {lh:.2f}   λ_a = {la:.2f}   "
            f"modal score: {mh}-{ma} ({p_score:.1%})"
        )
        print(
            f"│   1X2:   {home} {p_h:6.1%} ({fair_decimal_odds(p_h):5.2f})   "
            f"Draw {p_d:6.1%} ({fair_decimal_odds(p_d):5.2f})   "
            f"{away} {p_a:6.1%} ({fair_decimal_odds(p_a):5.2f})"
        )
        print(
            f"│   AH -0.5:  {home} {ah_05[0]:6.1%} ({fair_decimal_odds(ah_05[0]):5.2f})   "
            f"{away}+0.5 {ah_05[2]:6.1%} ({fair_decimal_odds(ah_05[2]):5.2f})"
        )
        print(
            f"│   AH -1.5:  {home} {ah_15[0]:6.1%} ({fair_decimal_odds(ah_15[0]):5.2f})   "
            f"{away}+1.5 {ah_15[2]:6.1%} ({fair_decimal_odds(ah_15[2]):5.2f})"
        )
        print(
            f"│   O/U 2.5:  Over {ou25[0]:6.1%} ({fair_decimal_odds(ou25[0]):5.2f})   "
            f"Under {ou25[2]:6.1%} ({fair_decimal_odds(ou25[2]):5.2f})"
        )
        print(
            f"│   O/U 3.5:  Over {ou35[0]:6.1%} ({fair_decimal_odds(ou35[0]):5.2f})   "
            f"Under {ou35[2]:6.1%} ({fair_decimal_odds(ou35[2]):5.2f})"
        )
        print(f"└   BTTS:    Yes {p_btts:6.1%} ({fair_decimal_odds(p_btts):5.2f})\n")

    conn.close()


if __name__ == "__main__":
    main()
