"""赛后「对答案」— grade yesterday's 推演 against what actually happened.

For every finished WC fixture that has a daily_tactics row, grade each of the
three total-goals opinions (tactical LLM / DC model / market) plus the tactical
corners & cards leans against the real numbers, and persist to `tactics_review`.
The digest's 「昨日对答案」 block reads this table; the cumulative scoreboard is
the first hard evidence of whose opinion is worth listening to.

Judgment can only have value if it is graded — without this loop, template prose
never pays a price for being wrong.

Lines used for grading (fixed, documented, deliberately simple):
  total goals  : over ↔ ≥3, under ↔ ≤2          (the OU 2.5 the leans were about)
  corners      : high ↔ ≥10, low ↔ ≤9           (WC corner medians sit ~9-10)
  cards        : high ↔ ≥5,  low ↔ ≤4           (yellows+reds, both teams)

Run (daily, after the stats ingest):
    PYTHONPATH=src python -m worldcup.eval.tactics_review
"""
from __future__ import annotations

import sqlite3

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

CORNERS_LINE = 9.5
CARDS_LINE = 4.5
WC_START = "2026-06-11"


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tactics_review (
            home TEXT, away TEXT, match_date TEXT,
            home_score INTEGER, away_score INTEGER,
            total_goals INTEGER, total_corners INTEGER, total_cards INTEGER,
            t_tg TEXT, q_lean TEXT, m_tg TEXT,
            hit_t INTEGER, hit_q INTEGER, hit_m INTEGER,
            corners_lean TEXT, hit_corners INTEGER,
            cards_lean TEXT, hit_cards INTEGER,
            conf REAL,
            reviewed_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (home, away, match_date)
        )""")
    conn.commit()


def _hit_ou(lean: str | None, total: int | None, line: float) -> int | None:
    """1=hit, 0=miss, None=no opinion (neutral/missing) or no actual data."""
    if total is None or lean in (None, "", "neutral"):
        return None
    if lean in ("over", "high"):
        return 1 if total > line else 0
    if lean in ("under", "low"):
        return 1 if total < line else 0
    return None


def review_finished(conn: sqlite3.Connection) -> int:
    """Grade every finished, not-yet-reviewed WC fixture with a daily_tactics row."""
    ensure_table(conn)
    rows = conn.execute("""
        SELECT m.home_code h, m.away_code a, m.match_date d,
               m.home_score hs, m.away_score as_,
               m.home_corners hc, m.away_corners ac,
               m.home_yellows hy, m.away_yellows ay,
               m.home_reds hr, m.away_reds ar,
               t.t_tg, t.q_lean, t.m_tg, t.corners, t.cards, t.conf
        FROM matches m JOIN daily_tactics t
          ON t.home = m.home_code AND t.away = m.away_code
        WHERE m.finished = 1 AND m.match_date >= ?
          AND m.home_score IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM tactics_review r
                          WHERE r.home = m.home_code AND r.away = m.away_code
                            AND r.match_date = m.match_date)
        ORDER BY m.match_date""", (WC_START,)).fetchall()
    n = 0
    for r in rows:
        goals = r["hs"] + r["as_"]
        corners = (r["hc"] + r["ac"]) if (r["hc"] is not None and r["ac"] is not None) else None
        cards = None
        if r["hy"] is not None and r["ay"] is not None:
            cards = r["hy"] + r["ay"] + (r["hr"] or 0) + (r["ar"] or 0)
        conn.execute("""INSERT OR REPLACE INTO tactics_review
            (home, away, match_date, home_score, away_score,
             total_goals, total_corners, total_cards,
             t_tg, q_lean, m_tg, hit_t, hit_q, hit_m,
             corners_lean, hit_corners, cards_lean, hit_cards, conf, reviewed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
            (r["h"], r["a"], r["d"], r["hs"], r["as_"],
             goals, corners, cards,
             r["t_tg"], r["q_lean"], r["m_tg"],
             _hit_ou(r["t_tg"], goals, 2.5),
             _hit_ou(r["q_lean"], goals, 2.5),
             _hit_ou(r["m_tg"], goals, 2.5),
             r["corners"], _hit_ou(r["corners"], corners, CORNERS_LINE),
             r["cards"], _hit_ou(r["cards"], cards, CARDS_LINE),
             r["conf"]))
        n += 1
    conn.commit()
    return n


def scoreboard(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """Cumulative (hits, graded) per opinion source since WC start."""
    out: dict[str, tuple[int, int]] = {}
    for label, col in (("战术", "hit_t"), ("模型", "hit_q"), ("市场", "hit_m"),
                       ("角球", "hit_corners"), ("牌", "hit_cards")):
        r = conn.execute(f"SELECT SUM({col}), COUNT({col}) FROM tactics_review").fetchone()
        out[label] = (r[0] or 0, r[1] or 0)
    return out


def main() -> None:
    conn = get_conn(DEFAULT_DB_PATH)
    try:
        n = review_finished(conn)
        print(f"对答案: 本次新评 {n} 场")
        for r in conn.execute("""SELECT * FROM tactics_review ORDER BY match_date DESC LIMIT 8"""):
            def mk(hit):
                return {1: "✓", 0: "✗"}.get(hit, "—")
            print(f"  {r['match_date']} {r['home']} {r['home_score']}-{r['away_score']} {r['away']}"
                  f" │ 总进球{r['total_goals']}: 战术{mk(r['hit_t'])} 模型{mk(r['hit_q'])} 市场{mk(r['hit_m'])}"
                  f" · 角球{r['total_corners'] if r['total_corners'] is not None else '?'}{mk(r['hit_corners'])}"
                  f" · 牌{r['total_cards'] if r['total_cards'] is not None else '?'}{mk(r['hit_cards'])}")
        sb = scoreboard(conn)
        print("累计: " + " · ".join(f"{k} {h}/{g}" for k, (h, g) in sb.items() if g))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
