"""FM-style scout layer — per-team tactical/player profile + match brief (vertical slice).

WHY (user, 2026-06-01): add qualitative tactical/player reads on top of the pure-quant
model. HONEST SCOPE: our DB has almost no structured tactics (market values empty,
positions mostly null, no formations), so a profile is ~80% LLM-generated, anchored only by
season-stat form + Elo. That makes it the seductive-narrative risk zone — so this is built
to FEED the two valid use cases, NOT to predict the main line:
  1. soft-prop angles (total goals / corners / cards — markets that price tactics lazily);
  2. injury-LAG sizing (is the out player system-critical or replaceable?).
Every profile is tagged with a confidence; key players are anchored to our in-form list and
the LLM is told NOT to invent players. The brief shows the quant total-goals vs the tactical
lean and a market-check reminder — a tactical view that AGREES with the market is not an edge.

CLI:
  PYTHONPATH=src python -m worldcup.strategy.scout --brief GER CUW
  PYTHONPATH=src python -m worldcup.strategy.scout --team BRA        # one profile
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import sys

import httpx

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.ingest.llm_news_scorer import DEEPSEEK_BASE, load_api_config
from worldcup.models.dixon_coles import fit
from worldcup.models.markets import prob_over_under, score_matrix
from worldcup.teams_zh import CODE_ZH

# Co-host home games are NOT neutral (martj42 marks all WC games neutral, but
# USA/CAN/MEX in their own country get real home advantage) — same convention
# as eval/calibration.py and simulator/monte_carlo.py.
HOSTS = {"USA", "CAN", "MEX"}

SYS = """你是一名职业足球球探,服务于一个软盘 prop 下注系统(总进球/角球/牌/比分)。给你一支国家队 + 我方数据库里该队近期有出场记录的球员清单(锚定用,可能不全)。输出严格 JSON 描述其战术与关键球员。

铁律(违反则此条无用):
1. 诚实第一。小国/冷门队你若没把握,给 **低 confidence**(≤0.4),style 用中性值,别硬编。
2. **绝不编造球员**:key_players 只能用清单里的人,或你**确有把握**的现役国脚;拿不准就少列或留空。
3. 这是判断不是真理——用于生成 prop 角度 + 伤病冲击评估,不是预测胜负。

输出 JSON:
{
 "formation": "<如 4-3-3,不确定写 unknown>",
 "style": {
   "tempo": {"score": 1-5, "note": "<中文≤12字>"},
   "width": {"score": 1-5, "note": "..."},
   "press": {"score": 1-5, "note": "..."},
   "directness": {"score": 1-5, "note": "..."},
   "defensive_line": {"score": 1-5, "note": "..."},
   "setpiece_threat": {"score": 1-5, "note": "..."}
 },
 "key_players": [{"name": "...", "role": "<中文角色>", "why_key": "<中文≤20字>", "replaceable": true/false}],
 "summary_zh": "<≤60字总体打法>",
 "goals_lean": {"dir": "over"/"under"/"neutral", "note": "<中文≤20字,这队倾向制造多/少进球的理由>"},
 "corners_cards_note": "<中文≤30字:压两翼→角球多 / 高位逼抢→牌多 之类,无把握写'不确定'>",
 "confidence": 0.0-1.0
}
score 含义:tempo 高=快节奏;width 高=多走边路传中;press 高=高位逼抢;directness 高=长传冲吊;defensive_line 高=防线压高;setpiece_threat 高=定位球威胁大。"""


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_profiles (
            team_code   TEXT PRIMARY KEY,
            payload     TEXT NOT NULL,
            confidence  REAL,
            source      TEXT,
            generated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )""")
    conn.commit()


def team_anchors(conn: sqlite3.Connection, code: str) -> tuple[str, list[str]]:
    """Elo + in-form players (our only structured grounding)."""
    elo = conn.execute("SELECT value FROM team_ratings WHERE team_code=? AND rating_type='elo'",
                       (code,)).fetchone()
    elo_s = f"{elo['value']:.0f}" if elo else "?"
    rows = conn.execute("""
        SELECT p.name, p.position, s.goals, s.assists, s.minutes, s.rating, s.season
        FROM players p JOIN player_season_stats s ON s.player_id = p.id
        WHERE p.nationality_code = ?
        ORDER BY s.season DESC, s.goals DESC, COALESCE(s.rating,0) DESC LIMIT 8
    """, (code,)).fetchall()
    anchors = [f"{r['name']}({r['position'] or '?'}, {r['season']}: {r['goals']}球{r['assists']}助"
               f"{(', 评分'+format(r['rating'],'.2f')) if r['rating'] else ''})" for r in rows]
    return elo_s, anchors


def generate_profile(conn: sqlite3.Connection, code: str, key: str, model: str,
                     force: bool = False) -> dict | None:
    if not force:
        cur = conn.execute("SELECT payload FROM team_profiles WHERE team_code=?", (code,)).fetchone()
        if cur:
            return json.loads(cur["payload"])
    elo_s, anchors = team_anchors(conn, code)
    anchor_txt = "; ".join(anchors) if anchors else "（数据库无近期出场记录）"
    user = (f"国家队代码 {code} = {CODE_ZH.get(code, code)}。Elo≈{elo_s}。"
            f"我方数据库近期出场球员(锚定,可能不全):\n{anchor_txt}")
    payload = {"model": model, "temperature": 0.2, "max_tokens": 900,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": user}]}
    try:
        r = httpx.post(f"{DEEPSEEK_BASE}/chat/completions",
                       headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                       json=payload, timeout=60)
        if r.status_code != 200:
            print(f"  [{code}] LLM HTTP {r.status_code}: {r.text[:150]}")
            return None
        content = r.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1].lstrip("json").strip()
        prof = json.loads(content)
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        print(f"  [{code}] LLM error: {type(e).__name__}: {e}")
        return None
    conn.execute("INSERT OR REPLACE INTO team_profiles(team_code,payload,confidence,source,generated_at)"
                 " VALUES (?,?,?,?,datetime('now'))",
                 (code, json.dumps(prof, ensure_ascii=False), prof.get("confidence"), f"deepseek:{model}"))
    conn.commit()
    return prof


MATCHUP_SYS = """你是足球战术分析师。给你两支队的风格档案 + **当前缺阵** + **近期状态**,推演这场对阵的战术博弈。

任务:
1. 每队一个**打法标签**(传控渗透 / 高位逼抢 / 边路冲吊 / 防守反击(防反) / 摆大巴铁桶 / 高强度对攻 / 长传冲吊 / 均衡 之一或组合)。
2. 推演两种打法**相撞**:谁控球、空间在哪(肋部/边路/身后)、节奏高低、谁压谁。
3. **结合"当前缺阵"**:缺的若是关键人,剧本必须相应调整(核心中卫缺→防线松、易丢球;前场核心缺→进攻乏力、总进球下调)。`swing_factor_zh` 要**基于真实缺阵**;若无缺阵则写最可能的临场变量。
4. **结合"近期状态"**:近期进/失球反映当前手感,别只凭旧印象(状态火热→上调进攻,连续零封→上调防守)。
5. **关键个人对决** `key_battles_zh`:2-3 个决定比赛的球员对位(谁打谁的区域、谁克谁),这是看点也是 prop 线索。
   人名铁律:对位球员**只能逐字使用**输入中 key_players / 锚定球员名单的原文名字——不得展开缩写、不得补全名
   (名单写 R. Freuler 就原样写 R. Freuler);名单外的人宁可写位置(如「右后卫位」),**绝不编名**。
6. 比分领先后的变化(强队领先会收→窄胜;弱队领先会龟缩→低进球)。
6b. **结合场地/天气(若给出)**:露天**高温/高湿**抑制**高位逼抢/高节奏**那一方(体能掉得快→节奏↓→易被拖入低分);**带顶恒温场=无天气影响,别提天气**。体现在 clash 与总进球(露天酷热通常偏小球)。
7. 对**软盘 prop** 的影响(目的,**不预测胜负**):总进球/角球/牌 各给倾向 + 战术理由。

铁律:战术判断,服务 prop 角度 + 伤病评估,不预测胜负。把握低就给低 confidence。中文简洁。
队名以输入首行给出的「代码=中文名」为准,全文只许出现这两支球队的名字,别写成其他国家队。
style 分数含义(1-5):tempo 高=快节奏;width 高=多走边路传中;press 高=高位逼抢;directness 高=长传冲吊;
defensive_line 高=防线压得高(不是防得好);setpiece_threat 高=定位球威胁大。

输出 JSON:
{
 "archetype": {"home": "<标签>", "away": "<标签>"},
 "clash_zh": "<≤60字:两种打法相撞会怎样>",
 "script_zh": "<≤50字:最可能的比赛剧本>",
 "key_battles_zh": ["<球员A vs 球员B:谁克谁/争夺哪块区域 ≤22字>", "..."],
 "game_states": {"home_first_zh": "<≤30字>", "away_first_zh": "<≤30字>"},
 "betting": {
   "total_goals": {"lean": "over"/"under"/"neutral", "why_zh": "<≤25字>"},
   "corners": {"lean": "high"/"low"/"neutral", "why_zh": "<≤25字>"},
   "cards": {"lean": "high"/"low"/"neutral", "why_zh": "<≤25字>"},
   "swing_factor_zh": "<≤30字:基于真实缺阵或最大临场变量,剧本怎么变>"
 },
 "confidence": 0.0-1.0
}"""


def ensure_matchup_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS match_tactics (
            home TEXT, away TEXT, payload TEXT NOT NULL, confidence REAL, absences_sig TEXT,
            generated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (home, away)
        )""")
    try:  # migrate older tables
        conn.execute("ALTER TABLE match_tactics ADD COLUMN absences_sig TEXT")
    except Exception:
        pass
    conn.commit()


def ensure_daily_table(conn: sqlite3.Connection) -> None:
    """Compact per-fixture tactical summary the daily digest reads (no LLM at read time)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_tactics (
            home TEXT, away TEXT, match_date TEXT,
            arch_h TEXT, arch_a TEXT, q_lean TEXT, t_tg TEXT, agree INTEGER,
            corners TEXT, cards TEXT, conf REAL, m_tg TEXT, m_pover REAL,
            generated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (home, away)
        )""")
    for col, typ in (("m_tg", "TEXT"), ("m_pover", "REAL"),
                     ("top_scores", "TEXT")):  # migrate older tables
        try:
            conn.execute(f"ALTER TABLE daily_tactics ADD COLUMN {col} {typ}")
        except Exception:
            pass
    conn.commit()


def _compact(prof: dict) -> dict:
    """Trim a team profile to what the matchup reasoner needs."""
    st = prof.get("style", {})
    return {"formation": prof.get("formation"),
            "style": {k: (st.get(k) or {}).get("score") for k in st},
            "summary": prof.get("summary_zh"),
            "key_players": [{"name": k.get("name"), "role": k.get("role"),
                             "replaceable": k.get("replaceable")} for k in prof.get("key_players", [])[:4]],
            "goals_lean": prof.get("goals_lean", {}).get("dir")}


def _team_absences(conn, code: str, limit: int = 4) -> list[str]:
    """Who's ACTUALLY out, deduped — UNION of two paid/structured sources:
      1) API-Football /injuries (authoritative, per-fixture; populates during the tournament),
      2) RSS+LLM news feed (pre-tournament + headline injuries; carries the Chinese gloss).
    Pre-tournament the news source dominates; once matches start, the structured feed adds in.

    Freshness rules (an absence claim is a fact with a shelf life):
      - news older than 21 days is dropped (a November red card is not a June absence);
      - a NEWER recovery row for the same player cancels the absence."""
    names: list[str] = []
    try:  # 1) API-Football structured injuries — latest snapshot only (each run re-lists)
        for r in conn.execute("""SELECT COALESCE(p.name, '伤员') nm FROM injuries i
                LEFT JOIN players p ON p.id = i.player_id
                WHERE i.team_code = ?
                  AND i.captured_at = (SELECT MAX(captured_at) FROM injuries WHERE team_code = ?)
                ORDER BY i.id DESC""", (code, code)).fetchall():
            names.append(r["nm"])  # structured feed: no news recovery row can cancel it
    except Exception:
        pass
    try:  # 2) RSS+LLM news feed (21-day window), with recovery cancellation
        recov: list[tuple[str, str, int]] = [
            (r["pz"] or "", r["pl"] or "", r["id"]) for r in conn.execute(
                """SELECT llm_player_zh pz, llm_player pl, id FROM news_alerts
                   WHERE llm_team = ? AND llm_impact_type = 'recovery'
                     AND COALESCE(llm_player_zh, llm_player) IS NOT NULL""", (code,)).fetchall()]

        def _recovered_after(pz: str, pl: str, news_id: int) -> bool:
            for rz, rl, rid in recov:
                if rid <= news_id:
                    continue  # recovery older than the absence news → doesn't cancel
                for a in (pz, pl):
                    for b in (rz, rl):
                        if a and b and (a in b or b in a):
                            return True
            return False

        for r in conn.execute("""SELECT llm_player_zh pz, llm_player pl, id FROM news_alerts
                WHERE llm_team = ? AND COALESCE(llm_severity, severity) >= 4
                  AND COALESCE(llm_impact_type, '') IN ('injury', 'suspension', 'squad_change')
                  AND COALESCE(llm_player_zh, llm_player) IS NOT NULL
                  AND COALESCE(published, captured_at) >= datetime('now', '-21 day')
                ORDER BY id DESC""", (code,)).fetchall():
            pz, pl = r["pz"] or "", r["pl"] or ""
            if _recovered_after(pz, pl, r["id"]):
                continue
            names.append(pz or pl)
    except Exception:
        pass
    out: list[str] = []
    for nm in names:
        nm = (nm or "").strip()
        # substring-aware dedup: collapses 弗洛雷斯⊂马塞洛·弗洛雷斯 and Neymar⊂Neymar Junior
        if not nm or any(nm in k or k in nm for k in out):
            continue
        out.append(nm)
        if len(out) >= limit:
            break
    return out


def _team_form(conn, code: str, n: int = 5) -> str:
    """Last n finished results from this team's view: '近5场 胜平负进失'."""
    try:
        rows = conn.execute("""
            SELECT home_code h, away_code a, home_score hs, away_score as_
            FROM matches WHERE finished = 1 AND home_score IS NOT NULL
              AND (home_code = ? OR away_code = ?)
            ORDER BY match_date DESC LIMIT ?""", (code, code, n)).fetchall()
    except Exception:
        return ""
    wdl, gf, ga = [], 0, 0
    for r in rows:
        f, ag = (r["hs"], r["as_"]) if r["h"] == code else (r["as_"], r["hs"])
        gf += f; ga += ag
        wdl.append("胜" if f > ag else ("平" if f == ag else "负"))
    return f"近{len(wdl)}场 {''.join(wdl)} 进{gf}失{ga}" if wdl else ""


def _match_id(conn, home: str, away: str):
    r = conn.execute("""SELECT id FROM matches WHERE home_code=? AND away_code=? AND finished=0
                        AND match_date>='2026-06-01' ORDER BY match_date LIMIT 1""",
                     (home, away)).fetchone()
    return r["id"] if r else None


def _fixture_weather(conn, home: str, away: str) -> str:
    """Venue + weather note for the fixture (roof-aware, from the fixed weather layer)."""
    r = conn.execute("""SELECT venue, weather_notes wn FROM matches
                        WHERE home_code=? AND away_code=? AND finished=0 AND match_date>='2026-06-01'
                        ORDER BY match_date LIMIT 1""", (home, away)).fetchone()
    if not r or not (r["venue"] or r["wn"]):
        return ""
    return f"{r['venue'] or '?'} — {r['wn'] or '天气未知'}"


def market_total_lean(conn, home: str, away: str) -> dict | None:
    """THE sharpest cross-check: de-vigged market P(over 2.5) from the paid Odds API OU lines
    (over/under @2.5, averaged across books, Shin de-vig). 'Market is sharper than our model'
    → this is the right baseline for the tactical total-goals call."""
    from worldcup.strategy.value_bets import devig_shin
    mid = _match_id(conn, home, away)
    if not mid:
        return None
    rows = conn.execute("""SELECT bookmaker bk, selection sel, price FROM odds
        WHERE match_id=? AND market='OU' AND line=2.5
          AND captured_at=(SELECT MAX(captured_at) FROM odds WHERE match_id=? AND market='OU' AND line=2.5)""",
        (mid, mid)).fetchall()
    books: dict = {}
    for r in rows:
        books.setdefault(r["bk"], {})[r["sel"]] = r["price"]
    povs = []
    for d in books.values():
        if "over" in d and "under" in d:
            try:
                p_over = devig_shin([d["over"], d["under"]])[0]
                povs.append(p_over)
            except Exception:
                pass
    if not povs:
        return None
    p = sum(povs) / len(povs)
    return {"p_over": p, "n_books": len(povs),
            "lean": "over" if p >= 0.55 else ("under" if p <= 0.45 else "neutral")}


def _roster_whitelist(conn, code: str, prof: dict | None) -> str:
    """Verbatim player names the matchup LLM may use for this team:
    DB season-stat anchors + the profile's key_players (already anchored at profile time)."""
    _, anchors = team_anchors(conn, code)
    names = [a.split("(")[0] for a in anchors]
    names += [(k.get("name") or "") for k in (prof or {}).get("key_players", [])]
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        n = n.strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return "、".join(out[:30])


_LATIN_NAME = re.compile(r"[A-ZÀ-Þ][\w.'\-]+(?:\s+[A-ZÀ-Þ][\w.'\-]+)*")


def _validate_matchup(mu: dict, home: str, away: str, roster_txt: str) -> list[str]:
    """Post-generation checks against the two LLM failure modes we've actually seen:
      1) prose mentioning a DIFFERENT country (POR-COD推演 once talked about 科特迪瓦);
      2) key_battles inventing/expanding names (R. Freuler → 'Ricardo Freuler').
    Wrong-country → returned as a problem (caller retries once).
    Off-whitelist name → the battle line is marked (待核) in place, not retried."""
    problems: list[str] = []
    pair_zh = {CODE_ZH.get(home, home), CODE_ZH.get(away, away)}
    prose = " ".join([mu.get("clash_zh") or "", mu.get("script_zh") or "",
                      json.dumps(mu.get("game_states") or {}, ensure_ascii=False),
                      json.dumps(mu.get("betting") or {}, ensure_ascii=False)])
    hits = sorted({z for c, z in CODE_ZH.items()
                   if z not in pair_zh and c not in (home, away) and z in prose})
    if hits:
        problems.append("误写他队队名: " + "、".join(hits))
    kbs = mu.get("key_battles_zh") or []
    for i, kb in enumerate(kbs):
        for m in _LATIN_NAME.findall(kb or ""):
            if m.isupper() and len(m) <= 4:   # VAR / GK / team codes — not player names
                continue
            if len(m) >= 3 and m not in roster_txt:
                kbs[i] = "(待核)" + kb        # name not in whitelist → flag, don't trust
                break
    return problems


def generate_matchup(conn, home, away, ph, pa, key, model, force=False) -> dict | None:
    if not (ph and pa):
        return None
    abs_h, abs_a = _team_absences(conn, home), _team_absences(conn, away)
    sig = "|".join(sorted(abs_h)) + "##" + "|".join(sorted(abs_a))
    if not force:
        cur = conn.execute("SELECT payload, absences_sig FROM match_tactics WHERE home=? AND away=?",
                           (home, away)).fetchone()
        if cur and (cur["absences_sig"] or "") == sig:  # cache valid only while absences unchanged
            return json.loads(cur["payload"])
    form_h, form_a = _team_form(conn, home), _team_form(conn, away)
    roster_h, roster_a = _roster_whitelist(conn, home, ph), _roster_whitelist(conn, away, pa)

    def blk(code, prof, ab, fm, roster):
        d = _compact(prof)
        d["当前缺阵"] = ab or "无已知"
        d["近期状态"] = fm or "无数据"
        d["锚定球员(对位人名只能逐字从中选)"] = roster or "无 → 对位只写位置"
        return f"{code}: {json.dumps(d, ensure_ascii=False)}"

    wx = _fixture_weather(conn, home, away)
    head = (f"本场对阵: 主队 {home}={CODE_ZH.get(home, home)} vs 客队 {away}={CODE_ZH.get(away, away)}"
            "(队名以此为准,勿写成任何其他国家队)")
    user = (head + "\n主队 " + blk(home, ph, abs_h, form_h, roster_h)
            + "\n客队 " + blk(away, pa, abs_a, form_a, roster_a)
            + (f"\n场地/天气: {wx}" if wx else ""))
    payload = {"model": model, "temperature": 0.3, "max_tokens": 900,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": MATCHUP_SYS},
                            {"role": "user", "content": user}]}
    mu = None
    for attempt in (1, 2):
        try:
            r = httpx.post(f"{DEEPSEEK_BASE}/chat/completions",
                           headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                           json=payload, timeout=60)
            if r.status_code != 200:
                print(f"  matchup LLM HTTP {r.status_code}: {r.text[:150]}"); return None
            content = r.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            mu = json.loads(content)
        except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"  matchup LLM error: {type(e).__name__}: {e}"); return None
        problems = _validate_matchup(mu, home, away, roster_h + "、" + roster_a)
        if not problems:
            break
        print(f"  [{home}-{away}] 推演校验未过({'; '.join(problems)})"
              + ("→ 重试一次" if attempt == 1 else "→ 保留(已标记),人工留意"))
    conn.execute("INSERT OR REPLACE INTO match_tactics(home,away,payload,confidence,absences_sig,generated_at)"
                 " VALUES (?,?,?,?,?,datetime('now'))",
                 (home, away, json.dumps(mu, ensure_ascii=False), mu.get("confidence"), sig))
    conn.commit()
    return mu


def _fmt_profile(code: str, prof: dict) -> str:
    st = prof.get("style", {})
    dims = "  ".join(f"{k}:{(st.get(k) or {}).get('score','?')}" for k in
                     ("tempo", "width", "press", "directness", "defensive_line", "setpiece_threat"))
    lines = [f"【{code}】阵型 {prof.get('formation','?')} · 置信 {prof.get('confidence','?')}",
             f"  打法: {prof.get('summary_zh','')}",
             f"  风格(1-5): {dims}",
             f"  进球倾向: {prof.get('goals_lean',{}).get('dir','?')} — {prof.get('goals_lean',{}).get('note','')}",
             f"  角球/牌: {prof.get('corners_cards_note','')}",
             "  关键球员:"]
    for kp in prof.get("key_players", [])[:4]:
        rep = "可替代" if kp.get("replaceable") else "难替代"
        lines.append(f"    · {kp.get('name','?')}({kp.get('role','')},{rep}): {kp.get('why_key','')}")
    return "\n".join(lines)


def brief(home: str, away: str, db_path=DEFAULT_DB_PATH) -> None:
    key, model = load_api_config()
    if not key:
        print("⚠️  DEEPSEEK_API_KEY 未设置。"); return
    conn = get_conn(db_path)
    ensure_table(conn)
    ensure_matchup_table(conn)
    print(f"生成球探档案 + 战术推演 {home} / {away} (DeepSeek, 命中缓存则跳过) ...")
    ph = generate_profile(conn, home, key, model)
    pa = generate_profile(conn, away, key, model)
    mu = generate_matchup(conn, home, away, ph, pa, key, model)

    # quant total-goals from DC. Kept BLIND from the matchup LLM (above) so the
    # tactical lean stays INDEPENDENT — anchoring it to the number would hide the
    # disagreements, and disagreement is the whole point.
    tg_line, q_lean = "", None
    try:
        p = fit(elo_prior_strength=0.5)
        if home in p.attack and away in p.attack:
            lh, la = p.predict_lambda(home, away, neutral=(home not in HOSTS))
            M = score_matrix(lh, la, rho=p.rho, max_goals=12)
            q_o25 = prob_over_under(M, 2.5)[0]
            o35 = prob_over_under(M, 3.5)[0]
            q_exp = float(lh + la)
            # 0.55/0.45 — same cut as market_total_lean, so model vs market leans
            # are comparable (the old asymmetric 0.57/0.47 had no backtest basis).
            q_lean = "over" if q_o25 >= 0.55 else ("under" if q_o25 <= 0.45 else "neutral")
            tg_line = (f"λ {lh:.2f}+{la:.2f}=期望{q_exp:.2f}球 · P(>2.5)={q_o25*100:.0f}% · "
                       f"P(>3.5)={o35*100:.0f}%")
        else:
            tg_line = "该对阵缺球队强度,跳过总进球量化"
    except Exception as e:
        tg_line = f"量化跳过: {type(e).__name__}"
    # conn stays OPEN — the display + 分歧雷达 below still use it (absences + market line); closed at end.

    print("\n" + "=" * 78)
    print(f"球探简报  {home} vs {away}")
    print("=" * 78)
    for code, prof in ((home, ph), (away, pa)):
        if prof:
            print(_fmt_profile(code, prof)); print()
        else:
            print(f"【{code}】档案生成失败\n")
    # — tactical matchup推演 —
    if mu:
        arch = mu.get("archetype", {})
        bet = mu.get("betting", {})
        print("— 战术博弈推演 (置信 %s) —" % mu.get("confidence", "?"))
        print(f"  打法: {home} = {arch.get('home','?')}  ×  {away} = {arch.get('away','?')}")
        ab_h, ab_a = _team_absences(conn, home), _team_absences(conn, away)
        if ab_h or ab_a:
            print(f"  当前缺阵: {home} {('、'.join(ab_h) or '无')} | {away} {('、'.join(ab_a) or '无')}")
        print(f"  相撞: {mu.get('clash_zh','')}")
        print(f"  剧本: {mu.get('script_zh','')}")
        for kb in (mu.get("key_battles_zh") or [])[:3]:
            print(f"  对位: {kb}")
        gs = mu.get("game_states", {})
        print(f"  领先后: {home}先进→{gs.get('home_first_zh','')} | {away}先进→{gs.get('away_first_zh','')}")
        print(f"  变量: {bet.get('swing_factor_zh','')}")
        def _b(k, label):
            d = bet.get(k, {})
            return f"{label} {d.get('lean','?')}({d.get('why_zh','')})"
        print("  prop 倾向: " + " | ".join([_b("total_goals", "总进球"), _b("corners", "角球"), _b("cards", "牌")]))
    else:
        print("— 战术博弈推演 — (生成失败)")

    # — 分歧雷达: 战术 vs 市场(最锐利) vs 模型. 价值只在「战术≠市场」处. —
    bet = (mu or {}).get("betting", {})
    t_tg = bet.get("total_goals", {}).get("lean")
    t_why = bet.get("total_goals", {}).get("why_zh", "")
    mk = market_total_lean(conn, home, away)   # 付费 Odds API 大小球盘(最锐利基准)
    conn.close()
    print("\n— 分歧雷达 (战术 vs 市场[最锐利] vs 模型;价值只在战术≠市场处) —")
    if mk:
        print(f"  市场 OU2.5: 去水 P(over)={mk['p_over']*100:.0f}% ({mk['n_books']}书) → 市场倾向 {mk['lean']}")
    print(f"  模型量化: {tg_line} → {q_lean or '—'}")
    base = (mk["lean"] if mk else None) or q_lean
    bname = "市场" if mk else "模型"
    if t_tg and base and base != "neutral":
        if t_tg == base:
            print(f"  战术: {t_tg} → ✓ 与{bname}一致 → 没有独立 edge(已被定价),跳过。")
        else:
            big = "小" if t_tg == "under" else "大"
            print(f"  战术: {t_tg}（{t_why}）→ ⚑ 与{bname}分歧!")
            print(f"    → 战术看{big}球而{bname}相反。{bname}(尤其去水盘口)通常更准、早把摆大巴/对攻算进去了,")
            print(f"      所以这不是自动 edge —— 只有你掌握{bname}还没消化的东西(刚爆的伤病/特定克制),逆它才有意义。")
    else:
        print(f"  战术: {t_tg or '—'} → 基准中性/缺失,无法对照,纯参考。")
    print(f"  角球/牌: 纯战术,无市场对照 = 最高叙事风险 → 仅作软盘角度,最小注。")
    print("\n⚠ 纪律: 现在基准是真金白银的市场盘(最锐利)。战术≠市场=看点,但市场通常更准——逆它要有过硬")
    print("  战术/伤病理由,软盘没漏价=没edge。低 conf 仅参考;小注、记 CLV。")


def _quant_view(p, home, away, topn: int = 3):
    """DC quant view for a fixture: (total-goals lean, top scorelines text).
    Both come from the same score matrix — the scorelines make the model's view
    concrete ('1-0 12% · 2-1 10%' reads better than a bare under), and honest:
    even the most likely score is a ~1-in-8 shot, which the percentages show."""
    if p is None or home not in p.attack or away not in p.attack:
        return None, None
    lh, la = p.predict_lambda(home, away, neutral=(home not in HOSTS))
    M = score_matrix(lh, la, rho=p.rho, max_goals=12)
    o25 = prob_over_under(M, 2.5)[0]
    lean = "over" if o25 >= 0.55 else ("under" if o25 <= 0.45 else "neutral")
    cells = [(float(M[i, j]), i, j) for i in range(8) for j in range(8)]
    cells.sort(reverse=True)
    scores = " · ".join(f"{i}-{j} {pr*100:.0f}%" for pr, i, j in cells[:topn])
    return lean, scores


def daily_briefs(days_ahead: int = 3, db_path=DEFAULT_DB_PATH, force: bool = False) -> None:
    """Pre-generate + cache tactical briefs for upcoming WC fixtures → daily_tactics.
    Profiles/matchups are cached, so steady-state this only LLM-calls genuinely-new fixtures."""
    key, model = load_api_config()
    if not key:
        print("⚠️  DEEPSEEK_API_KEY 未设置。"); return
    conn = get_conn(db_path)
    ensure_table(conn); ensure_matchup_table(conn); ensure_daily_table(conn)
    today = dt.date.today()
    lo, hi = today.isoformat(), (today + dt.timedelta(days=days_ahead)).isoformat()
    fx = conn.execute("""SELECT home_code h, away_code a, match_date d FROM matches
                         WHERE finished=0 AND match_date BETWEEN ? AND ? ORDER BY match_date""",
                      (lo, hi)).fetchall()
    if not fx:
        print(f"窗口内无比赛 ({lo} ~ {hi})"); conn.close(); return
    try:
        p = fit(elo_prior_strength=0.5)
    except Exception as e:
        p = None; print(f"DC fit 跳过: {type(e).__name__}")
    n = 0
    for r in fx:
        h, a = r["h"], r["a"]
        ph = generate_profile(conn, h, key, model)
        pa = generate_profile(conn, a, key, model)
        mu = generate_matchup(conn, h, a, ph, pa, key, model, force=force)
        if not mu:
            continue
        bet = mu.get("betting", {})
        q, top_scores = _quant_view(p, h, a)
        mk = market_total_lean(conn, h, a)          # ← paid Odds API OU line (sharpest)
        m_tg = mk["lean"] if mk else None
        m_pover = mk["p_over"] if mk else None
        t_tg = bet.get("total_goals", {}).get("lean")
        # market is the sharper baseline; a 50/50 market has no opinion → fall back to model
        # (the old `m_tg or q` let a truthy "neutral" suppress the model view: 22/47 fixtures
        # ended up with no verdict at all)
        base = m_tg if (m_tg and m_tg != "neutral") else q
        agree = (1 if (base and t_tg and base != "neutral" and base == t_tg)
                 else (0 if (base and t_tg and base != "neutral" and base != t_tg) else None))
        arch = mu.get("archetype", {})
        conn.execute("""INSERT OR REPLACE INTO daily_tactics
            (home,away,match_date,arch_h,arch_a,q_lean,t_tg,agree,corners,cards,conf,m_tg,m_pover,top_scores,generated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
            (h, a, r["d"], arch.get("home"), arch.get("away"), q, t_tg, agree,
             bet.get("corners", {}).get("lean"), bet.get("cards", {}).get("lean"), mu.get("confidence"),
             m_tg, m_pover, top_scores))
        conn.commit()
        n += 1
        mkstr = f"市场{m_tg}(P{m_pover*100:.0f}%)" if mk else "市场—"
        print(f"  {r['d']} {h}-{a}: {arch.get('home')}×{arch.get('away')} 总进球 {mkstr}/模型{q}/战术{t_tg}"
              f" {'✓' if agree==1 else ('⚑' if agree==0 else '—')}")
    conn.close()
    print(f"done: {n} fixtures → daily_tactics ({lo} ~ {hi})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--brief", nargs=2, metavar=("HOME", "AWAY"))
    ap.add_argument("--team", metavar="CODE")
    ap.add_argument("--all", action="store_true", help="generate profiles for all 48 WC teams (cached)")
    ap.add_argument("--daily", action="store_true", help="pre-gen tactical briefs for upcoming WC fixtures → daily_tactics")
    ap.add_argument("--days", type=int, default=3, help="--daily window (days ahead)")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached")
    args = ap.parse_args()
    if args.daily:
        daily_briefs(days_ahead=args.days, force=args.force)
        return
    if args.all:
        from worldcup.ingest.llm_news_scorer import WC_TEAMS
        key, model = load_api_config()
        if not key:
            print("⚠️  DEEPSEEK_API_KEY 未设置。"); return
        conn = get_conn(DEFAULT_DB_PATH); ensure_table(conn)
        done = 0
        for i, code in enumerate(WC_TEAMS, 1):
            prof = generate_profile(conn, code, key, model, force=args.force)
            if prof:
                done += 1
                print(f"  [{i}/{len(WC_TEAMS)}] {code} ✓ conf={prof.get('confidence','?')} "
                      f"{prof.get('formation','?')}")
            else:
                print(f"  [{i}/{len(WC_TEAMS)}] {code} ✗ failed")
        conn.close()
        print(f"\ndone: {done}/{len(WC_TEAMS)} profiles cached in team_profiles.")
        return
    if args.brief:
        brief(args.brief[0].upper(), args.brief[1].upper())
    elif args.team:
        key, model = load_api_config()
        if not key:
            print("⚠️  DEEPSEEK_API_KEY 未设置。"); return
        conn = get_conn(DEFAULT_DB_PATH); ensure_table(conn)
        prof = generate_profile(conn, args.team.upper(), key, model, force=args.force)
        conn.close()
        print(_fmt_profile(args.team.upper(), prof) if prof else "生成失败")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
