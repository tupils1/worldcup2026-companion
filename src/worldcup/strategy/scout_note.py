"""手动球探观察 — your eye-test, given a home the system actually consumes.

A 15-year watcher's subjective reads ("Spain wastes chances", "Argentina leans too
hard on Messi") catch exactly what the quant model is blind to: process vs result,
dependency structure, and finishing quality the goals-only DC model can't see. This
lets you log per-team observations that then:
  1. feed the tactical 推演 LLM prompt as an extra anchor (a HUMAN read, weighed but
     not blindly trusted — adding a note re-generates that team's briefs);
  2. get shown against an objective anchor — the team's this-tournament goals-minus-xG
     (finishing luck) — so "把握机会差" can be checked: did they create and not score?

Discipline (the project's "seductive narrative" guard): a note is a HYPOTHESIS. It's
worth most when it AGREES with xG against the model's face-value read (e.g. the model
under-rates Spain off 0 goals; you + xG say they created plenty), or flags a structural
factor no box score holds. It does NOT gate bets — it informs the read.

    PYTHONPATH=src python -m worldcup.strategy.scout_note --add ESP "把握机会差,但创造力强,进球会来"
    PYTHONPATH=src python -m worldcup.strategy.scout_note --add ARG "过度依赖梅西,梅西被盯死则进攻熄火"
    PYTHONPATH=src python -m worldcup.strategy.scout_note --list           # all, with xG cross-check
    PYTHONPATH=src python -m worldcup.strategy.scout_note --list ESP
    PYTHONPATH=src python -m worldcup.strategy.scout_note --rm 3
"""
from __future__ import annotations

import argparse
import sqlite3

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.teams_zh import CODE_ZH

WC_START = "2026-06-11"
NOTE_WINDOW_DAYS = 21  # a subjective read has a shelf life; older notes drop out of the prompt


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scout_notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            team_code  TEXT NOT NULL,
            note       TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""")
    conn.commit()


def add_note(conn, team: str, note: str) -> int:
    ensure_table(conn)
    cur = conn.execute("INSERT INTO scout_notes (team_code, note) VALUES (?, ?)",
                       (team.upper(), note.strip()))
    conn.commit()
    return cur.lastrowid


def remove_note(conn, note_id: int) -> bool:
    ensure_table(conn)
    cur = conn.execute("DELETE FROM scout_notes WHERE id=?", (note_id,))
    conn.commit()
    return cur.rowcount > 0


def team_notes(conn, code: str, days: int = NOTE_WINDOW_DAYS) -> list[str]:
    """Active (recent) observation strings for a team — what the prompt sees."""
    try:
        rows = conn.execute(
            "SELECT note FROM scout_notes WHERE team_code=? "
            "AND created_at >= datetime('now', ?) ORDER BY id DESC",
            (code.upper(), f"-{days} day")).fetchall()
    except sqlite3.OperationalError:
        return []  # table not created yet
    return [r["note"] for r in rows]


def finishing_delta(conn, code: str) -> dict | None:
    """Objective cross-check anchor: this tournament's goals_for, xG_for, and the
    finishing delta (goals − xG). delta << 0 = created but didn't score (wasteful);
    delta >> 0 = scored above the chances (clinical / lucky)."""
    rows = conn.execute("""
        SELECT home_code h, away_code a, home_score hs, away_score as_, home_xg hx, away_xg ax
        FROM matches WHERE finished=1 AND match_date>=? AND home_xg IS NOT NULL
          AND (home_code=? OR away_code=?)""", (WC_START, code.upper(), code.upper())).fetchall()
    if not rows:
        return None
    gf = xf = 0.0
    n = 0
    for r in rows:
        home = r["h"] == code.upper()
        gf += r["hs"] if home else r["as_"]
        xf += (r["hx"] if home else r["ax"]) or 0.0
        n += 1
    return {"matches": n, "gf": gf, "xg": xf, "delta": gf - xf}


def notes_signature(conn, home: str, away: str) -> str:
    """Cache key fragment — changes when either team's active notes change, so adding
    a note invalidates the cached matchup brief and it re-generates with the note."""
    ids = conn.execute(
        "SELECT id FROM scout_notes WHERE team_code IN (?, ?) "
        "AND created_at >= datetime('now', ?) ORDER BY id",
        (home.upper(), away.upper(), f"-{NOTE_WINDOW_DAYS} day")).fetchall()
    return ",".join(str(r["id"]) for r in ids)


def _zh(code: str) -> str:
    return CODE_ZH.get(code.upper(), code.upper())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--add", nargs=2, metavar=("TEAM", "NOTE"), help="add an observation")
    ap.add_argument("--list", nargs="?", const="*", metavar="TEAM", help="list notes (all, or one team)")
    ap.add_argument("--rm", type=int, metavar="ID", help="remove a note by id")
    args = ap.parse_args()
    conn = get_conn(DEFAULT_DB_PATH)
    ensure_table(conn)
    try:
        if args.add:
            nid = add_note(conn, args.add[0], args.add[1])
            print(f"已记录 #{nid}: {_zh(args.add[0])} — {args.add[1]}")
            print("（下次跑 scout --daily 时,该队的推演会带上这条观察并对照 xG）")
            return
        if args.rm is not None:
            print("已删除" if remove_note(conn, args.rm) else f"无此条 #{args.rm}")
            return
        # list (default)
        where = "" if args.list in (None, "*") else " WHERE team_code=?"
        params = () if args.list in (None, "*") else (args.list.upper(),)
        rows = conn.execute(
            f"SELECT id, team_code, note, substr(created_at,1,10) d FROM scout_notes{where} "
            "ORDER BY team_code, id DESC", params).fetchall()
        if not rows:
            print("暂无观察记录。用 --add TEAM \"...\" 添加。")
            return
        print("你的现场观察 (对照本届 xG):\n")
        last_team = None
        for r in rows:
            if r["team_code"] != last_team:
                fd = finishing_delta(conn, r["team_code"])
                if fd:
                    sign = "高效/运气" if fd["delta"] > 0.5 else ("浪费机会" if fd["delta"] < -0.5 else "中性")
                    anchor = (f"  [本届{fd['matches']}场: 进{fd['gf']:.0f}球 / xG{fd['xg']:.1f} "
                              f"→ 终结{fd['delta']:+.1f} {sign}]")
                else:
                    anchor = "  [本届无 xG 数据]"
                print(f"{_zh(r['team_code'])} ({r['team_code']}){anchor}")
                last_team = r["team_code"]
            print(f"    #{r['id']} ({r['d']}) {r['note']}")
        print("\n纪律: 观察是假设,不是定论。与 xG 一致=确认;与模型相反=看点。不直接进门禁。")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
