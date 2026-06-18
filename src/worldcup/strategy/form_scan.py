"""本届状态扫描 — objective xG-vs-result read for EVERY team, watched or not.

You watched the strong teams; for the rest, xG is the eye-test that doesn't need a
TV (or a noisy social-media scrape that's mostly already priced in). For each team
this scans the gap between what they DID (goals) and what they DESERVED (xG), both
ways:
  攻 delta = 进球 − xG_for   (>0 clinical/lucky finishing; <0 wasteful — Spain 0g/2.8xG)
  防 delta = xG_against − 失球 (>0 conceded fewer than chances faced; lucky/great GK)

Why it matters for round 2 betting: finishing and goalkeeping variance regress hard.
A team riding +finishing (Sweden 5g from 1.5xG) is a FADE candidate — results will
cool. A team with −finishing but high creation (Spain) is a BACK candidate — the
goals are coming. The market prices results faster than xG, so the regression gap is
where a soft edge can hide. This is a HYPOTHESIS generator, not a green light — still
run it through the gates.

    PYTHONPATH=src python -m worldcup.strategy.form_scan
    PYTHONPATH=src python -m worldcup.strategy.form_scan --team ESP
"""
from __future__ import annotations

import argparse

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.teams_zh import CODE_ZH

WC_START = "2026-06-11"
HOT = 1.0    # |delta| above this per-tournament = a real regression flag
WARM = 0.5


def team_form(conn, code: str) -> dict | None:
    rows = conn.execute("""
        SELECT home_code h, away_code a, home_score hs, away_score as_,
               home_xg hx, away_xg ax
        FROM matches WHERE finished=1 AND match_date>=? AND home_xg IS NOT NULL
          AND (home_code=? OR away_code=?)""", (WC_START, code, code)).fetchall()
    if not rows:
        return None
    gf = ga = xgf = xga = 0.0
    for r in rows:
        home = r["h"] == code
        gf += r["hs"] if home else r["as_"]
        ga += r["as_"] if home else r["hs"]
        xgf += (r["hx"] if home else r["ax"]) or 0.0
        xga += (r["ax"] if home else r["hx"]) or 0.0
    return {"code": code, "n": len(rows), "gf": gf, "ga": ga, "xgf": xgf, "xga": xga,
            "att": gf - xgf, "deff": xga - ga}


def _label(att: float, deff: float) -> tuple[str, str]:
    """Regression read for the team's NEXT match."""
    luck = att + deff   # total over-performance vs xG, both ends
    if att <= -HOT:
        return "⚑被低估", "创造多却没进球(终结差)→ 进球该回归向上,可背其大球/正名"
    if luck >= 2 * HOT:
        return "⚑虚高", "进球+防守都远好于xG → 强运气,结果大概率回落,可fade"
    if att >= HOT:
        return "偏运气", "进球高于创造 → 终结过热,留意回归"
    if deff >= HOT:
        return "门将/防运气", "失球少于面对的xG → 防守被美化,留意回归"
    if abs(luck) <= WARM:
        return "名副其实", "结果≈xG,状态实打实"
    return "略偏", ""


def scan(conn) -> list[dict]:
    codes = [r["code"] for r in conn.execute(
        "SELECT code FROM teams WHERE in_worldcup_2026=1").fetchall()]
    out = []
    for c in codes:
        f = team_form(conn, c)
        if f:
            f["label"], f["why"] = _label(f["att"], f["deff"])
            out.append(f)
    # most "off" first: biggest absolute total luck
    out.sort(key=lambda f: -abs(f["att"] + f["deff"]))
    return out


def digest_lines(conn, top: int = 8) -> list[str]:
    """Compact regression watchlist for the digest — only the teams xG flags hardest.
    Leads with the signal's own track record (once round 2 lets it be graded)."""
    rows = [f for f in scan(conn) if f["label"].startswith("⚑")][:top]
    if not rows:
        return []
    out = []
    try:
        from worldcup.eval.form_review import digest_line as _track
        track = _track(conn)
        if track:
            out.append(track)  # e.g. "回归信号命中: 7/10 (虚高 4/5 · 被低估 3/5)"
    except Exception:
        pass
    for f in rows:
        z = CODE_ZH.get(f["code"], f["code"])
        out.append(f"{z}: 进{f['gf']:.0f}/xG{f['xgf']:.1f} 失{f['ga']:.0f}/被xG{f['xga']:.1f}"
                   f" → {f['label']}")
        out.append(f"  {f['why']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team")
    ap.add_argument("--all", action="store_true", help="show every team, not just flagged")
    args = ap.parse_args()
    conn = get_conn(DEFAULT_DB_PATH)
    try:
        rows = scan(conn)
    finally:
        conn.close()
    if args.team:
        rows = [f for f in rows if f["code"] == args.team.upper()]
    elif not args.all:
        rows = [f for f in rows if f["label"].startswith("⚑")]
    print("本届状态扫描 (进球 vs xG;回归=第二轮软盘信号)")
    print(f"{'队':<8}{'场':>2}  {'进/xG攻':>10}  {'失/被xG':>10}  {'攻Δ':>6}{'防Δ':>6}  读")
    for f in rows:
        z = CODE_ZH.get(f["code"], f["code"])
        print(f"{z:<8}{f['n']:>2}  {f['gf']:.0f}/{f['xgf']:>5.1f}  {f['ga']:.0f}/{f['xga']:>5.1f}"
              f"  {f['att']:>+5.1f}{f['deff']:>+6.1f}  {f['label']}{(' '+f['why']) if f['why'] else ''}")
    print("\n纪律: 这是回归假设,不是绿灯。市场常已部分定价 → 仍要过门禁、对 CLV。")


if __name__ == "__main__":
    main()
