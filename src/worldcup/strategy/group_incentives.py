"""末轮出线利益 / 死亡之组探测器 — 2026 赛制特有的"软盘错价"角度。

48 队、12 组、**小组前2 + 8 个最好第3名**晋级。第3轮(末轮)很多场的利益是扭曲的:
"一平皆大欢喜""已锁出线轮换""已出局摆烂"。而"最好第3名"的排列极复杂,休闲玩家算不过来 →
这些末轮场的**总进球/小球被系统性错价**。本模块在小组前两轮打完后,给每场末轮赛标出利益状态
+ 对总进球的方向(默契平→小球、生死对攻→大球、走过场→小球)。

逻辑(组内**精确枚举**,不靠近似):末轮一组两场(A-B、C-D)共 3×3=9 种结果 → 每种算最终排名
(积分→净胜球→进球)→ 判每队在 W/D/L 下能否进前2;第3名出线另用阈值(跨组,近似)。

赛制说明:小组前两轮未打完 → 无积分 → 输出"待前两轮"。开赛后自动激活。
Run: PYTHONPATH=src python -m worldcup.strategy.group_incentives
"""
from __future__ import annotations
import sqlite3

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

# 第3名出线点数(跨组近似):48队制 8/12 第3名晋级 → 门槛低。≥4 多半安全,3 边缘,≤2 基本出局。
THIRD_SAFE = 4
THIRD_ALIVE = 3

# 一个比赛结果对(主,客)的 (积分, 净胜球, 进球) 增量(净胜/进球用作抢断同分的近似)
DELTA = {
    "H": ((3, 1, 1), (0, -1, 0)),   # 主胜
    "D": ((1, 0, 0), (1, 0, 0)),    # 平
    "A": ((0, -1, 0), (3, 1, 1)),   # 客胜
}


def group_standings(conn, group_letter: str) -> dict[str, dict]:
    """A group's table from FINISHED group-stage matches (pts/gd/gf/played)."""
    teams = [r["code"] for r in conn.execute(
        "SELECT code FROM teams WHERE group_letter=? AND in_worldcup_2026=1", (group_letter,))]
    tbl = {t: {"pts": 0, "gd": 0, "gf": 0, "played": 0} for t in teams}
    for m in conn.execute("""SELECT home_code h, away_code a, home_score hs, away_score as_
                             FROM matches WHERE finished=1 AND home_score IS NOT NULL
                               AND stage LIKE 'Group Stage%'
                               AND home_code IN (SELECT code FROM teams WHERE group_letter=?)""",
                          (group_letter,)):
        h, a, hs, as_ = m["h"], m["a"], m["hs"], m["as_"]
        if h not in tbl or a not in tbl:
            continue
        tbl[h]["gf"] += hs; tbl[a]["gf"] += as_
        tbl[h]["gd"] += hs - as_; tbl[a]["gd"] += as_ - hs
        tbl[h]["played"] += 1; tbl[a]["played"] += 1
        if hs > as_:   tbl[h]["pts"] += 3
        elif hs < as_: tbl[a]["pts"] += 3
        else:          tbl[h]["pts"] += 1; tbl[a]["pts"] += 1
    return tbl


def _final_rank(base: dict, results: list[tuple]) -> list[str]:
    tbl = {c: dict(v) for c, v in base.items()}
    for h, a, o in results:
        hd, ad = DELTA[o]
        for code, (dp, dg, df) in ((h, hd), (a, ad)):
            tbl[code]["pts"] += dp; tbl[code]["gd"] += dg; tbl[code]["gf"] += df
    return sorted(tbl, key=lambda c: (-tbl[c]["pts"], -tbl[c]["gd"], -tbl[c]["gf"]))


def _team_state(base, fh, fa, oh, oa, team) -> str:
    """secured / draw_ok / must_win / dead / live, by enumerating the 9 MD3 outcome combos."""
    by_res = {"win": [], "draw": [], "loss": []}
    for fo in "HDA":
        tr = ("win" if (fo == "H" and team == fh) or (fo == "A" and team == fa)
              else "loss" if (fo == "H" and team == fa) or (fo == "A" and team == fh)
              else "draw")
        for oo in "HDA":
            top2 = team in _final_rank(base, [(fh, fa, fo), (oh, oa, oo)])[:2]
            by_res[tr].append(top2)
    allres = by_res["win"] + by_res["draw"] + by_res["loss"]
    if all(allres):
        return "secured"            # 进前2 无论如何
    if not any(allres):             # 任何情况都进不了前2 → 看第3名
        proj = base[team]["pts"] + 3   # 全力赢能到的点数
        return "must_win" if proj >= THIRD_ALIVE else "dead"
    if all(by_res["draw"]):
        return "draw_ok"            # 平局通常足够进前2
    if all(by_res["win"]) and sum(by_res["draw"]) <= 1:
        return "must_win"           # 赢能进、平基本不够 → 必须赢
    return "live"


# (stateA, stateB)-set → (标签, 总进球倾向, 说明). 顺序敏感的用 frozenset 对称匹配。
def _fixture_call(sa: str, sb: str) -> tuple[str, str, str]:
    pair = frozenset((sa, sb))
    if sa == sb == "secured":
        return ("走过场", "under", "双方已锁前2 → 强度低、轮换,偏小球")
    if pair == frozenset(("dead",)) or (sa == "dead" and sb == "dead"):
        return ("死亡过场", "under", "双方已出局 → 轮换、低强度,偏小球")
    if sa == sb == "draw_ok":
        return ("默契平", "under", "一平皆大欢喜 → 可能默契/保守,偏小球")
    if sa == sb == "must_win":
        return ("生死对攻", "over", "双方必须赢 → 开放对攻,偏大球")
    if "secured" in pair and "must_win" in pair:
        return ("一安一拼", "neutral", "一方已出线(可能轮换)、一方拼命 → 方向不定,偏看需求方")
    if "dead" in pair and "must_win" in pair:
        return ("一死一拼", "neutral", "一方出局(轮换)、一方拼命 → 需求方进攻 vs 对方放养")
    if "draw_ok" in pair and "secured" in pair:
        return ("低压", "under", "双方都安全(锁定/平即可)→ 强度低,偏小球")
    return ("常规", "neutral", "双方有正常竞争 → 中性")


def md3_board(conn) -> list[dict]:
    """每场末轮赛的利益状态 + 总进球倾向. 只在该组前两轮都打完后才判定。"""
    out = []
    md3 = conn.execute("""SELECT m.home_code h, m.away_code a, m.match_date d, t.group_letter g
        FROM matches m JOIN teams t ON t.code=m.home_code
        WHERE m.finished=0 AND m.stage='Group Stage - 3' ORDER BY t.group_letter, m.match_date""").fetchall()
    for r in md3:
        g = r["g"]
        st = group_standings(conn, g)
        # 该组的两场末轮赛
        gm = [(x["home_code"], x["away_code"]) for x in conn.execute(
            """SELECT home_code, away_code FROM matches m JOIN teams t ON t.code=m.home_code
               WHERE m.stage='Group Stage - 3' AND t.group_letter=?""", (g,))]
        if len(st) != 4 or any(v["played"] < 2 for v in st.values()):
            out.append({"group": g, "home": r["h"], "away": r["a"], "date": r["d"],
                        "state": "待前两轮", "lean": "—", "note": "小组前两轮未打完,无法判定"})
            continue
        fh, fa = r["h"], r["a"]
        other = [p for p in gm if set(p) != {fh, fa}]
        if not other:
            continue
        oh, oa = other[0]
        sa = _team_state(st, fh, fa, oh, oa, fh)
        sb = _team_state(st, fh, fa, oh, oa, fa)
        label, lean, note = _fixture_call(sa, sb)
        out.append({"group": g, "home": fh, "away": fa, "date": r["d"],
                    "state": label, "lean": lean, "note": note, "sa": sa, "sb": sb})
    return out


def main():
    conn = get_conn(DEFAULT_DB_PATH)
    board = md3_board(conn)
    conn.close()
    print("=" * 76)
    print("末轮出线利益探测器 (Group Stage - 3) — 默契平/生死对攻/走过场 → 总进球方向")
    print("=" * 76)
    if not board:
        print("  无末轮赛程。")
        return
    pending = [b for b in board if b["state"] == "待前两轮"]
    live = [b for b in board if b["state"] != "待前两轮"]
    for b in live:
        tag = {"under": "↓小球", "over": "↑大球", "neutral": "中性"}.get(b["lean"], "")
        print(f"  [{b['group']}] {b['date'][5:]} {b['home']}-{b['away']:<4} "
              f"{b['state']:<6} {tag}  ({b.get('sa','?')}/{b.get('sb','?')})")
        print(f"        {b['note']}")
    if pending:
        print(f"\n  待前两轮打完才能判定的末轮场: {len(pending)} (开赛后自动激活)")
    print("\n用法:默契平/走过场/死亡过场 → 看软盘是否漏了小球;生死对攻 → 大球。竞彩/Polymarket 总进球对账,")
    print("市场没把利益结构算进去才算 edge(48 队第3名规则复杂,大众常算不清 → 这正是错价来源)。")


if __name__ == "__main__":
    main()
