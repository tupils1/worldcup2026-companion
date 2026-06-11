"""Correlated-prop value detector: score_matrix joint probs vs Pinnacle props.

Pulls Pinnacle's full-match team-props (Winner/Total, BTTS/Winner, BTTS/Total,
BTTS, Total Odd/Even, Win-to-Nil, etc.), computes the TRUE joint probability
from our Dixon-Coles score matrix, and compares.

Why correlated props are the model's best market:
    Books price multi-outcome props with varying sophistication. Combos like
    "Home win & Over 2.5" are POSITIVELY correlated (winning teams score more),
    so the true joint > naive product. Even Pinnacle's prop book is less sharp
    than its main lines (lower liquidity). Gaps here are more likely real than
    on the 1X2 main line.

Only full-match props (no half-time models needed). Skips "1st Half" / "HT-FT".

Run:
    PYTHONPATH=src python -m worldcup.strategy.prop_value --min-edge 0.03
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH
from worldcup.ingest.pinnacle import NAME_TO_FIFA, PinnacleClient, american_to_decimal
from worldcup.models.dixon_coles import fit
from worldcup.models.markets import score_matrix
from worldcup.strategy.value_bets import devig_shin

HOSTS = frozenset({"USA", "CAN", "MEX"})


# ─── score_matrix joint-probability helpers ───────────────────────────────

def _grid(M):
    n = M.shape[0]
    H = np.arange(n)[:, None] * np.ones(n)[None, :]
    A = np.ones(n)[:, None] * np.arange(n)[None, :]
    return H, A  # H[i,j]=i (home goals), A[i,j]=j (away goals)


def p_winner_total(M, side, line):
    """P(side wins AND total over/under line). side: home/draw/away, line e.g. ('over',2.5)."""
    H, A = _grid(M)
    ou, val = line
    win = {"home": H > A, "draw": H == A, "away": A > H}[side]
    tot = (H + A > val) if ou == "over" else (H + A < val)
    return float(M[win & tot].sum())


def p_btts_winner(M, btts_yes, side):
    H, A = _grid(M)
    bt = (H >= 1) & (A >= 1) if btts_yes else ~((H >= 1) & (A >= 1))
    win = {"home": H > A, "draw": H == A, "away": A > H}[side]
    return float(M[bt & win].sum())


def p_btts_total(M, btts_yes, line):
    H, A = _grid(M)
    ou, val = line
    bt = (H >= 1) & (A >= 1) if btts_yes else ~((H >= 1) & (A >= 1))
    tot = (H + A > val) if ou == "over" else (H + A < val)
    return float(M[bt & tot].sum())


def p_btts(M, yes=True):
    H, A = _grid(M)
    bt = (H >= 1) & (A >= 1)
    return float(M[bt].sum()) if yes else float(M[~bt].sum())


def p_total_oddeven(M, odd=True):
    H, A = _grid(M)
    tot = (H + A).astype(int)
    mask = (tot % 2 == 1) if odd else (tot % 2 == 0)
    return float(M[mask].sum())


def p_team_oddeven(M, side, odd=True):
    H, A = _grid(M)
    g = H if side == "home" else A
    mask = (g.astype(int) % 2 == 1) if odd else (g.astype(int) % 2 == 0)
    return float(M[mask].sum())


def p_win_to_nil(M, side):
    H, A = _grid(M)
    if side == "home":
        return float(M[(H > A) & (A == 0)].sum())
    return float(M[(A > H) & (H == 0)].sum())


def p_either_score(M):
    return float(1.0 - M[0, 0])


def p_double_chance(M, combo):
    H, A = _grid(M)
    if combo == "home_draw":
        return float(M[H >= A].sum())
    if combo == "away_draw":
        return float(M[A >= H].sum())
    return float(M[H != A].sum())  # home_away


# ─── Pinnacle prop parsing ─────────────────────────────────────────────────

def build_match_home_away(matchups) -> dict[int, tuple[str, str]]:
    """{regular matchup id: (home_fifa, away_fifa)} from type=matchup entries."""
    out = {}
    for m in matchups:
        if m.get("type") != "matchup" or m.get("special"):
            continue
        parts = m.get("participants", [])
        home = away = None
        for p in parts:
            algn = p.get("alignment")
            fifa = NAME_TO_FIFA.get(p.get("name", ""))
            if algn == "home":
                home = fifa
            elif algn == "away":
                away = fifa
        if home and away:
            out[m["id"]] = (home, away)
    return out


def parse_side(token: str, home_name: str, away_name: str) -> str | None:
    """Map a team-name or 'Draw' token to home/draw/away."""
    t = token.strip()
    if t.lower() == "draw":
        return "draw"
    if t == home_name:
        return "home"
    if t == away_name:
        return "away"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--prior", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    print("Fitting Dixon-Coles ...")
    params = fit(elo_prior_strength=args.prior)

    print("Pulling Pinnacle matchups + markets ...")
    client = PinnacleClient()
    matchups = client.matchups()
    markets = client.markets()
    mbm = defaultdict(list)
    for m in markets:
        mbm[m.get("matchupId")].append(m)

    # Map each prop matchup to its (home, away) via parent regular matchup
    home_away = build_match_home_away(matchups)
    # Also map by participant team names within prop (fallback)
    # Build name lookup for prop's parent
    parent_of = {}
    for m in matchups:
        if m.get("parentId"):
            parent_of[m["id"]] = m["parentId"]

    edges = []
    props = [m for m in matchups if isinstance(m.get("special"), dict)
             and m["special"].get("category") == "Team Props"]

    for prop in props:
        desc = prop["special"]["description"]
        if "1st Half" in desc or "Half-Time" in desc or "First Team" in desc:
            continue  # need HT model
        parent = parent_of.get(prop["id"])
        ha = home_away.get(parent)
        if not ha:
            continue
        home_fifa, away_fifa = ha
        if home_fifa not in params.attack or away_fifa not in params.attack:
            continue
        # Build score matrix once per match (cache by match)
        neutral = not (home_fifa in HOSTS)
        lh, la = params.predict_lambda(home_fifa, away_fifa, neutral=neutral)
        M = score_matrix(lh, la, rho=params.rho)

        # Get participant prices
        pid_name = {p["id"]: p.get("name", "") for p in prop.get("participants", [])}
        ml = next((mk for mk in mbm.get(prop["id"], []) if mk.get("type") == "moneyline"), None)
        if not ml:
            continue
        # team display names from parent participants
        home_disp = away_disp = None
        for m2 in matchups:
            if m2["id"] == parent:
                for p in m2.get("participants", []):
                    if p.get("alignment") == "home": home_disp = p.get("name")
                    elif p.get("alignment") == "away": away_disp = p.get("name")
        if not home_disp or not away_disp:
            continue

        # Collect (outcome_label, decimal_odds)
        outcomes = []
        for pr in ml.get("prices", []):
            pid = pr.get("participantId"); am = pr.get("price")
            if pid is None or am is None: continue
            outcomes.append((pid_name.get(pid, ""), american_to_decimal(am)))
        if len(outcomes) < 2:
            continue

        # De-vig the prop (all outcomes mutually exclusive within a prop)
        labels = [o[0] for o in outcomes]
        prices = [o[1] for o in outcomes]
        implied = devig_shin(prices)

        # Compute model prob per outcome based on prop type
        for (label, odds), imp in zip(outcomes, implied):
            model_p = _model_prob_for_outcome(M, desc, label, home_disp, away_disp)
            if model_p is None:
                continue
            edge = model_p - imp
            if abs(edge) >= args.min_edge:
                edges.append({
                    "match": f"{home_fifa}-{away_fifa}", "prop": desc, "outcome": label,
                    "odds": odds, "pin_impl": imp, "model": model_p, "edge": edge,
                })

    edges.sort(key=lambda x: -x["edge"])
    print(f"\n=== Correlated-prop edges (|model − Pinnacle| ≥ {args.min_edge:.0%}) ===\n")
    print(f"{'match':<9} {'prop':<28} {'outcome':<22} {'odds':>6} {'pin':>6} {'model':>6} {'edge':>7}")
    print("-" * 92)
    pos = [e for e in edges if e["edge"] > 0]
    for e in pos[:args.top]:
        print(f"{e['match']:<9} {e['prop'][:28]:<28} {e['outcome'][:22]:<22} "
              f"{e['odds']:6.2f} {e['pin_impl']*100:5.1f}% {e['model']*100:5.1f}% {e['edge']*100:+6.1f}%")
    print(f"\n  {len(pos)} positive-edge prop outcomes. ⚠️ Pinnacle props are semi-sharp;")
    print(f"  treat >10% as model bias. Real value: soft book using PRODUCT pricing on these combos.")


def _model_prob_for_outcome(M, desc, label, home_disp, away_disp):
    """Dispatch label → score_matrix joint probability based on prop type."""
    d = desc.lower()
    lab = label.strip()

    def side_of(name):
        if name.lower() == "draw": return "draw"
        if name == home_disp: return "home"
        if name == away_disp: return "away"
        return None

    # Winner/Total Goals: "Team & Over 2.5"
    if "winner/total" in d:
        m = re.match(r"(.+?) & (Over|Under) ([\d.]+)", lab)
        if not m: return None
        side = side_of(m.group(1))
        if side is None: return None
        return p_winner_total(M, side, (m.group(2).lower(), float(m.group(3))))

    # BTTS/Winner: "Yes & Team" or "No & Team"
    if "both teams to score/winner" in d:
        m = re.match(r"(Yes|No) & (.+)", lab)
        if not m: return None
        side = side_of(m.group(2))
        if side is None: return None
        return p_btts_winner(M, m.group(1) == "Yes", side)

    # BTTS/Total: "Yes & Over 2.5"
    if "both teams to score/total" in d:
        m = re.match(r"(Yes|No) & (Over|Under) ([\d.]+)", lab)
        if not m: return None
        return p_btts_total(M, m.group(1) == "Yes", (m.group(2).lower(), float(m.group(3))))

    # BTTS simple
    if d == "both teams to score?":
        return p_btts(M, yes=(lab.lower() == "yes"))

    # Either team to score
    if "either team to score" in d:
        return p_either_score(M) if lab.lower() == "yes" else 1 - p_either_score(M)

    # Total Goals Odd/Even
    if d == "total goals odd/even":
        return p_total_oddeven(M, odd=(lab.lower() == "odd"))

    # Team Goals Odd/Even: "<Team> Goals Odd/Even"
    if "goals odd/even" in d and "total" not in d:
        team = desc.split(" Goals")[0]
        side = side_of(team)
        if side is None: return None
        return p_team_oddeven(M, side, odd=(lab.lower() == "odd"))

    # To Win to Nil
    if "to win to nil" in d:
        team = desc.split(" To Win")[0]
        side = side_of(team)
        if side is None or lab.lower() != "yes":
            if lab.lower() == "no" and side: return 1 - p_win_to_nil(M, side)
            return None
        return p_win_to_nil(M, side)

    # Double Chance
    if d == "double chance":
        if "/" in lab or "or" in lab.lower():
            # e.g. "Sweden or Draw"
            pass
        return None  # parsing varies; skip for now

    return None


if __name__ == "__main__":
    main()
