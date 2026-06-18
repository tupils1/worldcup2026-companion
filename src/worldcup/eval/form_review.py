"""回归信号对答案 — does the xG over/under-performance flag actually predict?

form_scan flags teams as ⚑虚高 (scored above xG → "results will cool, fade") or
⚑被低估 (created but didn't score → "goals are coming, back"). That's a regression-
to-the-mean CLAIM. Like 对答案 grades the tactical leans, this grades the regression
flags against what teams actually did next — so the signal earns trust (or gets
killed) on real hit-rate instead of plausibility.

Method: walk each team's WC matches in order. From their 2nd match on, take their
prior finishing rate (avg goals−xG over earlier games). If that prior is clearly
flagged (|avg| > HOT), predict this match regresses toward the mean:
  虚高 (prior avg > +HOT)  → this game's (goals−xG) should drop below the prior avg
  被低估 (prior avg < −HOT) → this game's (goals−xG) should rise above the prior avg
HIT/MISS on that. The scoreboard is the honest answer to "is this signal real?".

    PYTHONPATH=src python -m worldcup.eval.form_review
"""
from __future__ import annotations

import sqlite3

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.strategy.form_scan import HOT, WC_START
from worldcup.teams_zh import CODE_ZH


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS form_review (
            team TEXT, match_date TEXT,
            flag TEXT,                 -- 'over' (虚高) / 'under' (被低估)
            prior_avg REAL,            -- finishing delta/game BEFORE this match
            match_delta REAL,          -- this match's goals - xG
            hit INTEGER,               -- regressed toward the mean as predicted?
            reviewed_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (team, match_date)
        )""")
    conn.commit()


def _team_matches(conn, code: str) -> list[dict]:
    """This team's finished WC matches in date order: (date, goals_for, xg_for)."""
    rows = conn.execute("""
        SELECT match_date d, home_code h, away_code a, home_score hs, away_score as_,
               home_xg hx, away_xg ax
        FROM matches WHERE finished=1 AND match_date>=? AND home_xg IS NOT NULL
          AND (home_code=? OR away_code=?)
        ORDER BY match_date""", (WC_START, code, code)).fetchall()
    out = []
    for r in rows:
        home = r["h"] == code
        gf = r["hs"] if home else r["as_"]
        xgf = (r["hx"] if home else r["ax"]) or 0.0
        out.append({"date": r["d"], "gf": gf, "xgf": xgf, "delta": gf - xgf})
    return out


def review(conn) -> int:
    ensure_table(conn)
    codes = [r["code"] for r in conn.execute(
        "SELECT code FROM teams WHERE in_worldcup_2026=1").fetchall()]
    n = 0
    for c in codes:
        seq = _team_matches(conn, c)
        for k in range(1, len(seq)):          # from the 2nd match on
            prior = seq[:k]
            prior_avg = sum(m["delta"] for m in prior) / len(prior)
            if abs(prior_avg) <= HOT:
                continue                       # not flagged → nothing to predict
            flag = "over" if prior_avg > 0 else "under"
            this = seq[k]
            # regressed toward the mean = finishing moved opposite to the flag
            hit = (this["delta"] < prior_avg) if flag == "over" else (this["delta"] > prior_avg)
            if conn.execute("SELECT 1 FROM form_review WHERE team=? AND match_date=?",
                            (c, this["date"])).fetchone():
                continue
            conn.execute("""INSERT INTO form_review
                (team, match_date, flag, prior_avg, match_delta, hit, reviewed_at)
                VALUES (?,?,?,?,?,?,datetime('now'))""",
                (c, this["date"], flag, round(prior_avg, 2), round(this["delta"], 2), int(hit)))
            n += 1
    conn.commit()
    return n


def scoreboard(conn) -> dict:
    try:
        rows = conn.execute("SELECT flag, hit FROM form_review").fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, list[int]] = {"虚高": [], "被低估": [], "合计": []}
    for r in rows:
        k = "虚高" if r["flag"] == "over" else "被低估"
        out[k].append(r["hit"]); out["合计"].append(r["hit"])
    return {k: (sum(v), len(v)) for k, v in out.items() if v}


def digest_line(conn) -> str:
    """One-line cumulative accuracy for the digest's state-scan header, or '' if none."""
    sb = scoreboard(conn)
    tot = sb.get("合计")
    if not tot or not tot[1]:
        return ""
    bits = " · ".join(f"{k} {h}/{g}" for k, (h, g) in sb.items() if k != "合计" and g)
    return f"回归信号命中: {tot[0]}/{tot[1]} ({bits})"


def main() -> None:
    conn = get_conn(DEFAULT_DB_PATH)
    try:
        n = review(conn)
        print(f"回归信号对答案: 本次新评 {n} 条")
        for r in conn.execute("SELECT * FROM form_review ORDER BY match_date DESC, team LIMIT 16"):
            z = CODE_ZH.get(r["team"], r["team"])
            tag = "虚高" if r["flag"] == "over" else "被低估"
            mk = "✓回归" if r["hit"] else "✗未回归"
            print(f"  {r['match_date']} {z}: 赛前{tag}(终结{r['prior_avg']:+.1f}/场) → "
                  f"本场终结{r['match_delta']:+.1f} {mk}")
        sb = scoreboard(conn)
        if sb:
            print("累计: " + " · ".join(f"{k} {h}/{g}" for k, (h, g) in sb.items()))
        else:
            print("(还没有可评的:每队需打满2场才能验证上一场的回归预测——第二轮后激活)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
