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
from worldcup.models.standings import rank_teams

# 第3名出线点数(跨组近似):48队制 8/12 第3名晋级 → 门槛低。≥4 多半安全,3 边缘,≤2 基本出局。
THIRD_SAFE = 4
THIRD_ALIVE = 3
DEAD_GD = -4  # 0分且净胜球已极差 → 即便末轮全胜到3分也基本排不进最佳8个第3
# 'Group Stage - 3' 这个 stage 串历届世界杯都用(2022/2023的同名比赛也在库里),
# 必须按本届日期收口,否则会把往届比赛当成本届末轮(曾导致跨组 KeyError 崩溃)。
WC2026_FROM = "2026-06-01"


def group_standings(conn, group_letter: str) -> dict[str, dict]:
    """A group's table from FINISHED group-stage matches (pts/gd/gf/played)."""
    teams = [r["code"] for r in conn.execute(
        "SELECT code FROM teams WHERE group_letter=? AND in_worldcup_2026=1", (group_letter,))]
    tbl = {t: {"pts": 0, "gd": 0, "gf": 0, "played": 0} for t in teams}
    for m in conn.execute("""SELECT home_code h, away_code a, home_score hs, away_score as_
                             FROM matches WHERE finished=1 AND home_score IS NOT NULL
                               AND stage LIKE 'Group Stage%' AND match_date >= ?
                               AND home_code IN (SELECT code FROM teams WHERE group_letter=?)""",
                          (WC2026_FROM, group_letter)):
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


def group_played(conn, group_letter: str) -> list[tuple]:
    """This group's FINISHED 2026 matches as (home, away, hg, ag) — the real pairwise
    results head-to-head tiebreaking needs."""
    teams = {r["code"] for r in conn.execute(
        "SELECT code FROM teams WHERE group_letter=? AND in_worldcup_2026=1", (group_letter,))}
    out = []
    for m in conn.execute("""SELECT home_code h, away_code a, home_score hs, away_score as_
                             FROM matches WHERE finished=1 AND home_score IS NOT NULL
                               AND stage LIKE 'Group Stage%' AND match_date >= ?""", (WC2026_FROM,)):
        if m["h"] in teams and m["a"] in teams:
            out.append((m["h"], m["a"], m["hs"], m["as_"]))
    return out


# MD3 outcome → a representative scoreline (±1 goal), so head-to-head among teams that
# end level can be evaluated consistently with the real played results.
_MD3_SCORE = {"H": (1, 0), "D": (0, 0), "A": (0, 1)}


def _final_rank(teams, played, md3_ops, elo=None) -> list[str]:
    """Rank the group under FIFA 2026 (head-to-head first) given real played matches
    plus the hypothetical MD3 results (as ±1 scorelines)."""
    results = list(played)
    for fh, fa, o in md3_ops:
        hg, ag = _MD3_SCORE[o]
        results.append((fh, fa, hg, ag))
    return rank_teams(teams, results, elo or {})


def _seeding_live(teams, played, fh, fa, oh, oa, team, elo=None) -> bool:
    """Does THIS match's result still change `team`'s final group position? Holding the
    other MD3 match fixed, if varying fh-fa's result moves team's rank, its seeding is
    live — two already-qualified teams playing each other for top spot (→ better R32
    draw) are NOT a walkover, even though both are 'secured' for top-2."""
    for oo in "HDA":
        ranks = {_final_rank(teams, played, [(fh, fa, fo), (oh, oa, oo)], elo).index(team)
                 for fo in "HDA"}
        if len(ranks) > 1:
            return True
    return False


def _team_state(base, teams, played, fh, fa, oh, oa, team, elo=None) -> str:
    """secured / third_safe / draw_ok / must_win / dead / live, by enumerating the 9
    MD3 outcome combos for top-2, then third-place math for teams that can't make top-2.
    Ranking inside the enumeration uses FIFA 2026 head-to-head (via _final_rank)."""
    by_res = {"win": [], "draw": [], "loss": []}
    for fo in "HDA":
        tr = ("win" if (fo == "H" and team == fh) or (fo == "A" and team == fa)
              else "loss" if (fo == "H" and team == fa) or (fo == "A" and team == fh)
              else "draw")
        for oo in "HDA":
            top2 = team in _final_rank(teams, played, [(fh, fa, fo), (oh, oa, oo)], elo)[:2]
            by_res[tr].append(top2)
    allres = by_res["win"] + by_res["draw"] + by_res["loss"]
    if all(allres):
        return "secured"            # 进前2 无论如何
    if not any(allres):             # 任何情况都进不了前2 → 走第3名出线逻辑
        pts, gd = base[team]["pts"], base[team]["gd"]
        # 旧版 proj=pts+3>=3 恒真 → dead 永不可达,把已4分稳出线的队误判 must_win。
        if pts >= THIRD_SAFE:
            return "third_safe"     # 已≥4分,基本锁定最佳第3 → 末轮无压力(走过场)
        if pts == 0 and gd <= DEAD_GD:
            return "dead"           # 0分且净胜球极差,末轮全胜也基本排不进8个最佳第3
        return "must_win"           # 需要赢(或拿分)才有第3名出线的实际可能
    if all(by_res["draw"]):
        return "draw_ok"            # 平局通常足够进前2
    if all(by_res["win"]) and sum(by_res["draw"]) <= 1:
        return "must_win"           # 赢能进、平基本不够 → 必须赢
    return "live"


# (stateA, stateB) → (标签, 总进球倾向, 说明). third_safe(已稳进最佳第3)在下注语义上
# 等同 secured(无压力),归一到 SAFE 一并处理。
SAFE = {"secured", "third_safe"}


def _fixture_call(sa: str, sb: str) -> tuple[str, str, str]:
    pair = {sa, sb}
    if sa in SAFE and sb in SAFE:
        return ("走过场", "under", "双方已出线(锁前2或稳最佳第3)→ 强度低、轮换,偏小球")
    if sa == sb == "dead":
        return ("死亡过场", "under", "双方已出局 → 轮换、低强度,偏小球")
    if sa == sb == "draw_ok":
        return ("默契平", "under", "一平皆大欢喜 → 可能默契/保守,偏小球")
    if sa == sb == "must_win":
        return ("生死对攻", "over", "双方必须赢 → 开放对攻,偏大球")
    if pair & SAFE and "must_win" in pair:
        return ("一安一拼", "neutral", "一方已出线(可能轮换)、一方拼命 → 方向不定,偏看需求方")
    if "dead" in pair and "must_win" in pair:
        return ("一死一拼", "neutral", "一方出局(轮换)、一方拼命 → 需求方进攻 vs 对方放养")
    if "draw_ok" in pair and pair & SAFE:
        return ("低压", "under", "双方都安全(锁定/平即可)→ 强度低,偏小球")
    if "dead" in pair and pair & SAFE:
        return ("一安一死", "under", "一方已出线、一方出局 → 都无压力,偏小球")
    return ("常规", "neutral", "双方有正常竞争 → 中性")


def _load_elo(conn) -> dict:
    try:
        return {r["team_code"]: float(r["value"]) for r in conn.execute(
            "SELECT team_code, value FROM team_ratings WHERE rating_type='elo'")}
    except Exception:
        return {}


def md3_board(conn) -> list[dict]:
    """每场末轮赛的利益状态 + 总进球倾向. 只在该组前两轮都打完后才判定。"""
    out = []
    elo = _load_elo(conn)
    md3 = conn.execute("""SELECT m.home_code h, m.away_code a, m.match_date d, t.group_letter g
        FROM matches m JOIN teams t ON t.code=m.home_code
        WHERE m.finished=0 AND m.stage='Group Stage - 3' AND m.match_date >= ?
        ORDER BY t.group_letter, m.match_date""", (WC2026_FROM,)).fetchall()
    for r in md3:
        g = r["g"]
        st = group_standings(conn, g)
        teams = list(st)
        played = group_played(conn, g)
        # 该组的两场末轮赛 — 限本届 + 同组(防往届同名 stage 行混入,曾致跨组崩溃)
        gm = [(x["home_code"], x["away_code"]) for x in conn.execute(
            """SELECT m.home_code, m.away_code FROM matches m
               JOIN teams th ON th.code=m.home_code
               JOIN teams ta ON ta.code=m.away_code
               WHERE m.stage='Group Stage - 3' AND m.match_date >= ?
                 AND th.group_letter=? AND ta.group_letter=?""", (WC2026_FROM, g, g))]
        if len(st) != 4 or any(v["played"] < 2 for v in st.values()):
            out.append({"group": g, "home": r["h"], "away": r["a"], "date": r["d"],
                        "state": "待前两轮", "lean": "—", "note": "小组前两轮未打完,无法判定"})
            continue
        fh, fa = r["h"], r["a"]
        # defensive: only same-group fixtures whose four teams are all in the table
        other = [p for p in gm if set(p) != {fh, fa} and p[0] in st and p[1] in st]
        if not other or fh not in st or fa not in st:
            continue
        oh, oa = other[0]
        sa = _team_state(st, teams, played, fh, fa, oh, oa, fh, elo)
        sb = _team_state(st, teams, played, fh, fa, oh, oa, fa, elo)
        # 两队都已锁前2,但若小组头名仍在这场之间未决(影响R32签位)→ 争头名,非走过场
        if sa == "secured" and sb == "secured" and (
                _seeding_live(teams, played, fh, fa, oh, oa, fh, elo)
                or _seeding_live(teams, played, fh, fa, oh, oa, fa, elo)):
            label, lean, note = ("争头名", "neutral",
                "双方已出线但仍争小组头名(头名→R32签位更优)→ 都有动力赢,非走过场")
        else:
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
