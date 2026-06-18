"""出线形势 + 球队出线动机 — conditional qualification outlook after group games.

Once round 1 is played, projecting qualification from a blank slate throws away
what already happened (Saudi already drew Uruguay; Germany already hammered
Curaçao). This conditions the Monte Carlo on the REAL results: seed each group's
table from finished matches, simulate only the remaining fixtures, and tally
P(win group / 1st / 2nd / 3rd / advance) per team — advance = top-2 OR one of the
best-8 third-placers (the 2026 48-team rule), computed across all groups jointly.

It then derives each team's INCENTIVE for their next match from the projection:
  锁定 (already mathematically through) · 主动权在手 (a win ≈ secures) ·
  生死 (must win / win-and-pray) · 出局 (eliminated) · 混战 (still open).

Robust to the duplicate twin rows (counts each unordered pair once) so standings
are correct even before scripts/dedup_twins.py is run.

    PYTHONPATH=src python -m worldcup.strategy.qualification            # full board
    PYTHONPATH=src python -m worldcup.strategy.qualification --group A  # one group
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.models.dixon_coles import fit
from worldcup.simulator.monte_carlo import HOSTS, GroupStanding, sampler_from_params
from worldcup.teams_zh import CODE_ZH

THIRD_PLACE_SLOTS = 8  # 2026: best 8 of 12 third-placers advance


@dataclass
class Seed:
    """A group's state: starting table from played games + remaining fixtures."""
    teams: list[str]
    pts: dict[str, int]
    gf: dict[str, int]
    ga: dict[str, int]
    played: dict[str, int]
    remaining: list[tuple[str, str]]  # (home, away) not-yet-played, home non-neutral if host


def load_groups(conn: sqlite3.Connection) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT group_letter g, code FROM teams WHERE in_worldcup_2026=1 AND group_letter IS NOT NULL"
    ).fetchall()
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        groups[r["g"]].append(r["code"])
    return {g: sorted(ts) for g, ts in sorted(groups.items())}


def _played_pairs(conn) -> dict[frozenset, sqlite3.Row]:
    """Finished group-stage matches keyed by unordered pair — dedup-robust: if twin
    rows exist for a pair, keep the richest (has xG/afid), so points count once."""
    rows = conn.execute("""
        SELECT home_code h, away_code a, home_score hs, away_score as_,
               (home_xg IS NOT NULL) rich, api_football_id afid
        FROM matches
        WHERE finished=1 AND home_score IS NOT NULL
          AND match_date>='2026-06-11' AND stage LIKE 'Group Stage%'""").fetchall()
    best: dict[frozenset, sqlite3.Row] = {}
    for r in rows:
        key = frozenset((r["h"], r["a"]))
        cur = best.get(key)
        if cur is None or (r["rich"], r["afid"] is not None) > (cur["rich"], cur["afid"] is not None):
            best[key] = r
    return best


def _remaining_pairs(conn, teams: list[str], played: set[frozenset]) -> list[tuple[str, str]]:
    """Group-stage fixtures for these teams not yet played, oriented home/away as scheduled."""
    ph = ",".join("?" * len(teams))
    sql = ("SELECT home_code h, away_code a FROM matches "
           "WHERE finished=0 AND match_date>='2026-06-11' AND stage LIKE 'Group Stage' || '%' "
           f"AND home_code IN ({ph}) AND away_code IN ({ph})")
    rows = conn.execute(sql, teams + teams).fetchall()
    out, seen = [], set()
    for r in rows:
        key = frozenset((r["h"], r["a"]))
        if key in played or key in seen:
            continue
        seen.add(key)
        out.append((r["h"], r["a"]))
    return out


def build_seeds(conn) -> dict[str, Seed]:
    groups = load_groups(conn)
    played = _played_pairs(conn)
    seeds: dict[str, Seed] = {}
    for g, teams in groups.items():
        pts = {t: 0 for t in teams}
        gf = {t: 0 for t in teams}
        ga = {t: 0 for t in teams}
        pl = {t: 0 for t in teams}
        done: set[frozenset] = set()
        for key, r in played.items():
            if not key <= set(teams):
                continue
            done.add(key)
            h, a, hs, as_ = r["h"], r["a"], r["hs"], r["as_"]
            gf[h] += hs; ga[h] += as_; gf[a] += as_; ga[a] += hs
            pl[h] += 1; pl[a] += 1
            if hs > as_:
                pts[h] += 3
            elif hs < as_:
                pts[a] += 3
            else:
                pts[h] += 1; pts[a] += 1
        remaining = _remaining_pairs(conn, teams, done)
        seeds[g] = Seed(teams, pts, gf, ga, pl, remaining)
    return seeds


def _rank(seed: Seed, pts, gf, ga, elo) -> list[str]:
    return sorted(seed.teams, key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t], -elo.get(t, 1500.0)))


def project(conn, n_sims: int = 20000, seed_rng: int = 12) -> dict:
    """Monte-Carlo qualification probabilities conditioned on results so far."""
    params = fit(elo_prior_strength=0.5)
    sampler = sampler_from_params(params, calibrated=True)
    elo = {t: 1500.0 for t in params.teams}
    try:
        for r in conn.execute("SELECT team_code, value FROM team_ratings WHERE rating_type='elo'"):
            elo[r["team_code"]] = float(r["value"])
    except Exception:
        pass
    seeds = build_seeds(conn)
    rng = np.random.default_rng(seed_rng)

    groups_sorted = sorted(seeds)
    # tallies
    place = {t: np.zeros(4) for g in seeds for t in seeds[g].teams}   # 1st..4th counts
    win_grp = defaultdict(int)
    advance = defaultdict(int)

    for _ in range(n_sims):
        thirds: list[tuple] = []   # (pts, gd, gf, elo, team)
        group_rank: dict[str, list[str]] = {}
        for g in groups_sorted:
            s = seeds[g]
            pts = dict(s.pts); gf = dict(s.gf); ga = dict(s.ga)
            for h, a in s.remaining:
                neutral = h not in HOSTS
                hg, ag = sampler.sample_match(h, a, neutral, rng)
                gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
                if hg > ag:
                    pts[h] += 3
                elif hg < ag:
                    pts[a] += 3
                else:
                    pts[h] += 1; pts[a] += 1
            ranked = _rank(s, pts, gf, ga, elo)
            group_rank[g] = ranked
            for i, t in enumerate(ranked):
                place[t][i] += 1
            win_grp[ranked[0]] += 1
            advance[ranked[0]] += 1
            advance[ranked[1]] += 1
            t3 = ranked[2]
            thirds.append((pts[t3], gf[t3] - ga[t3], gf[t3], elo.get(t3, 1500.0), t3))
        best8 = sorted(thirds, reverse=True)[:THIRD_PLACE_SLOTS]
        for *_rest, t in best8:
            advance[t] += 1

    out = {"n_sims": n_sims, "groups": {}}
    for g in groups_sorted:
        s = seeds[g]
        rows = []
        for t in s.teams:
            rows.append({
                "team": t, "pts": s.pts[t], "gd": s.gf[t] - s.ga[t], "gf": s.gf[t],
                "played": s.played[t],
                "p_win": win_grp[t] / n_sims,
                "p_1st": place[t][0] / n_sims, "p_2nd": place[t][1] / n_sims,
                "p_3rd": place[t][2] / n_sims, "p_adv": advance[t] / n_sims,
            })
        rows.sort(key=lambda r: -r["p_adv"])
        out["groups"][g] = {"rows": rows, "remaining": s.remaining,
                            "all_played": not s.remaining}
    return out


# ── incentive derivation ─────────────────────────────────────────────────────
def incentive(row: dict, all_played: bool) -> tuple[str, str]:
    """A team's qualification incentive label + one-line read, from its P(advance)."""
    p = row["p_adv"]
    if all_played:
        return ("已出线", "小组赛收官,已晋级") if p >= 0.999 else (
            ("已出局", "小组赛收官,已淘汰") if p <= 0.001 else ("待定", "等其他组末轮定最佳第三"))
    if p >= 0.985:
        return "基本锁定", "出线概率极高,末轮多为练兵/保人"
    if p <= 0.03:
        return "命悬一线", "出线概率渺茫,需大胜+多组结果配合"
    if p <= 0.15:
        return "生死战", "必须拿分(多半是赢),否则大概率出局"
    if p >= 0.75:
        return "主动权在手", "一场不败≈出线,赢则提前锁定"
    return "混战", "出线悬而未决,下一场分量很重"


def board(conn, n_sims: int = 20000) -> dict:
    proj = project(conn, n_sims=n_sims)
    for g, gd in proj["groups"].items():
        for row in gd["rows"]:
            label, note = incentive(row, gd["all_played"])
            row["incentive"], row["incentive_note"] = label, note
    return proj


def zh(code: str) -> str:
    return CODE_ZH.get(code, code)


def _fmt_group(g: str, gd: dict) -> list[str]:
    out = [f"【{g}组】" + ("(小组赛已收官)" if gd["all_played"] else
                          f"(剩{len(gd['remaining'])}场)")]
    out.append(f"  {'队':<6}{'积分':>4}{'净':>4}{'进':>4}  {'出线率':>6}  {'夺头名':>6}  动机")
    for r in gd["rows"]:
        out.append(f"  {zh(r['team']):<6}{r['pts']:>4}{r['gd']:>+4}{r['gf']:>4}  "
                   f"{r['p_adv']*100:>5.0f}%  {r['p_win']*100:>5.0f}%  "
                   f"{r['incentive']}·{r['incentive_note']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", help="only this group (A-L)")
    ap.add_argument("--n-sims", type=int, default=20000)
    args = ap.parse_args()
    conn = get_conn(DEFAULT_DB_PATH)
    try:
        b = board(conn, n_sims=args.n_sims)
    finally:
        conn.close()
    print("=" * 70)
    print(f"出线形势 + 出线动机 (条件化蒙特卡洛, {b['n_sims']:,} 次, 基于已踢结果)")
    print("=" * 70)
    for g, gd in b["groups"].items():
        if args.group and g != args.group.upper():
            continue
        print("\n".join(_fmt_group(g, gd)))
        print()


if __name__ == "__main__":
    main()
