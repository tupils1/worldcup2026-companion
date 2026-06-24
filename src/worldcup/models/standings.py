"""FIFA 2026 group-stage ranking — head-to-head FIRST (the 2026 rule change).

The 2026 World Cup reordered the group tiebreakers: when teams are level on points,
the FIRST cut is now the head-to-head mini-table among ONLY the tied teams (points,
then GD, then goals between them), and overall goal difference drops to the SECOND
stage. Previous editions (and most of this codebase, written from memory) used overall
GD first — which is wrong for 2026 and changes who tops/advances in tie scenarios.

Official order (https://www.fifa.com/.../groups-how-teams-qualify-tie-breakers):
  0. points (all group matches)
  1a-c. HEAD-TO-HEAD among the tied teams: points, GD, goals (between them only)
        — reapplied recursively to any still-tied subset
  2a-c. overall GD, overall goals, team conduct (cards — not modelled; skipped)
  3.    FIFA ranking  (we use Elo as the proxy; no FIFA ranking in the DB)

Third-place teams come from different groups, so head-to-head can't apply — rank them
with rank_thirds() (points, GD, goals, then Elo).
"""
from __future__ import annotations

from itertools import groupby

Match = tuple[str, str, int, int]   # (home, away, home_goals, away_goals)


def _table(matches: list[Match], teams) -> dict[str, dict]:
    """Aggregate pts/gd/gf over the given matches, restricted to the given teams."""
    ts = set(teams)
    tbl = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    for h, a, hg, ag in matches:
        if h not in ts or a not in ts:
            continue
        tbl[h]["gf"] += hg; tbl[a]["gf"] += ag
        tbl[h]["gd"] += hg - ag; tbl[a]["gd"] += ag - hg
        if hg > ag:
            tbl[h]["pts"] += 3
        elif hg < ag:
            tbl[a]["pts"] += 3
        else:
            tbl[h]["pts"] += 1; tbl[a]["pts"] += 1
    return tbl


def _break_tie(tied: list[str], matches: list[Match], overall: dict, elo: dict) -> list[str]:
    """Order teams level on POINTS, per FIFA 2026: head-to-head first (recursively),
    then overall GD/goals, then Elo (FIFA-ranking proxy)."""
    if len(tied) == 1:
        return list(tied)
    h2h = _table(matches, tied)   # mini-table among ONLY the tied teams
    ordered = sorted(tied, key=lambda t: (-h2h[t]["pts"], -h2h[t]["gd"], -h2h[t]["gf"]))
    out: list[str] = []
    for _key, grp in groupby(ordered, key=lambda t: (h2h[t]["pts"], h2h[t]["gd"], h2h[t]["gf"])):
        grp = list(grp)
        if len(grp) == 1:
            out.extend(grp)
        elif len(grp) == len(tied):
            # head-to-head separated nothing → fall through to overall criteria
            out.extend(sorted(grp, key=lambda t: (
                -overall[t]["gd"], -overall[t]["gf"], -elo.get(t, 1500.0))))
        else:
            # a proper subset is still tied → reapply head-to-head within just that subset
            out.extend(_break_tie(grp, matches, overall, elo))
    return out


def rank_teams(teams, matches: list[Match], elo: dict | None = None) -> list[str]:
    """Return `teams` in final group order under the FIFA 2026 tiebreakers."""
    elo = elo or {}
    overall = _table(matches, teams)
    out: list[str] = []
    by_pts = sorted(teams, key=lambda t: -overall[t]["pts"])
    for _pts, grp in groupby(by_pts, key=lambda t: overall[t]["pts"]):
        grp = list(grp)
        out.extend(grp if len(grp) == 1 else _break_tie(grp, matches, overall, elo))
    return out


def third_place_key(team: str, matches: list[Match], teams, elo: dict | None = None):
    """Sort key for cross-group third-place ranking (no head-to-head): points, GD,
    goals, then Elo. Higher is better → use as reverse-sort or negate."""
    elo = elo or {}
    t = _table(matches, teams)[team]
    return (t["pts"], t["gd"], t["gf"], elo.get(team, 1500.0))
