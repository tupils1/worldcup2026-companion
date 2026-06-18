"""Compose the daily radar digest in Chinese (for the Telegram push).

Sources:
  - cross-book arb / best-line, Polymarket edges, injury-lag → parsed from the
    bets log produced by daily_refresh.sh (argv[1]); fixed English tokens are
    mapped to Chinese.
  - news (sev≥4 injuries / squad-changes) + morale watch-list → queried from the
    DB, using the LLM Chinese gloss (llm_title_zh) when available.
Source-data tokens (team codes, odds, player names, English headlines without a
gloss) are left as-is.

Usage:
  python -m worldcup.notify.daily_digest <bets_log_path>
  python -m worldcup.notify.daily_digest <bets_log_path> | python -m worldcup.notify.telegram
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.strategy.radar import morale_watchlist
from worldcup.strategy.group_incentives import md3_board

# Fixed English → Chinese map (regex, longest/most-specific first; word-boundary
# safe so team codes / player names / odds are never touched).
_MAP: list[tuple[str, str]] = [
    (r"thin market unmoved but single-source — corroborate before acting",
     "薄盘未动且仅单一来源——先交叉验证再行动"),
    (r"market still ≈ pre-injury level — consider LAY",
     "市场仍≈受伤前水平——可考虑反向挂(LAY)"),
    (r"single-source — corroborate before acting", "单一来源——先交叉验证"),
    (r"liquid market unmoved — judges the player replaceable; trust market",
     "流动盘未动——判定球员可替代;信市场"),
    (r"market ≈ model injury view — no edge", "市场≈模型受伤判断——无价差"),
    (r"market priced more drop than our estimate — it may know more",
     "市场定价跌幅超出我们估计——它可能知道更多"),
    (r"market below model's injured level — market more bearish than our estimate",
     "低于模型受伤后水平——市场比我们更看空"),
    (r"injury too mild \(clamped\) to move this market — no tradeable signal",
     "伤病过轻(已压制),不足以撼动此盘——无可交易信号"),
    (r"market rates (\S+) ABOVE model's pre-injury level — model underrates \S+, NOT lag",
     r"市场对 \1 评级高于模型受伤前——模型低估,非滞后"),
    (r"\bover-priced:", "过度定价:"),
    (r"\bmkt-bearish:", "市场更看空:"),
    (r"\bmodel-bias:", "模型偏差:"),
    (r"\bnegligible:", "可忽略:"),
    (r"advance \(to-qualify\)", "出线(晋级)"),
    (r"\bchampion\b", "夺冠"),
    (r"No Polymarket edge.*", "无 Polymarket 价差"),
    (r"\bBUY NO\b", "卖出NO"),
    (r"\bBUY YES\b", "买入YES"),
    (r"\bPROFIT\b", "利润"),
    (r"\bSummary:", "汇总:"),
    (r"best-line \+EV bets", "个最优线+EV 注"),
    (r"avg cross-book spread", "平均跨书价差"),
    (r"\barbs\b", "个套利"),
    (r"·thin", "·薄盘"),
    (r"\bLAG\?:", "滞后?:"),
    (r"\bwatch:", "观察:"),
    (r"\bpriced:", "已定价:"),
    (r"\bliquid-unmoved:", "流动盘未动:"),
    (r"✓agree", "✓一致"),
    (r"✗disagree", "✗不一致"),
    (r"1X2", "胜平负"),
    (r"inv-sum", "逆和"),
]

# FIFA 3-letter code → Chinese — shared with the scout prompts (worldcup/teams_zh.py).
from worldcup.teams_zh import CODE_ZH

# Market/tactical lean values → Chinese (raw over/under/high/low read like debug logs).
LEAN_ZH: dict[str, str] = {"over": "大球", "under": "小球", "neutral": "五五开",
                           "high": "偏多", "low": "偏少"}


def lean_zh(v: str | None) -> str:
    return LEAN_ZH.get(v or "", v or "—")


_CN_TZ = dt.timezone(dt.timedelta(hours=8))
_WD = "一二三四五六日"


def kickoff_cn(kickoff_utc: str | None, match_date: str) -> str:
    """Kickoff in Beijing time — the watcher's first question is '今晚几点'.
    '06-12 周五 03:00' from a full ISO kickoff, or just the date if none."""
    if kickoff_utc:
        try:
            d = dt.datetime.fromisoformat(kickoff_utc).astimezone(_CN_TZ)
            return f"{d:%m-%d} 周{_WD[d.weekday()]} {d:%H:%M}"
        except (ValueError, TypeError):
            pass
    return (match_date or "")[5:]


def team_zh(code: str | None) -> str:
    return CODE_ZH.get(code or "", code or "?")


def _codes_zh(text: str) -> str:
    """Replace standalone 3-letter FIFA codes with Chinese (leaves 1X2, EV, etc.)."""
    return re.sub(r"\b[A-Z]{3}\b", lambda m: CODE_ZH.get(m.group(0), m.group(0)), text)


def tr(line: str) -> str:
    """Full line translation: fixed tokens + country codes."""
    return _codes_zh(zh(line))


def zh(text: str) -> str:
    for pat, repl in _MAP:
        text = re.sub(pat, repl, text)
    return text


def _between(log: str, start: str, end: str) -> list[str]:
    out, on = [], False
    for ln in log.splitlines():
        if start in ln:
            on = True
            continue
        if on and end in ln:
            break
        if on:
            out.append(ln)
    return out


def arb_block(log: str) -> list[str]:
    lines = _between(log, "CROSS-BOOK ARBITRAGE", "OUTRIGHT VALUE BETS")
    keep = [l for l in lines if ("🎯" in l or l.strip().startswith("Summary"))]
    return [tr(l) for l in keep[:16]]


def poly_block(log: str) -> list[str]:
    keep = [l for l in log.splitlines()
            if re.search(r"BUY YES|BUY NO|No Polymarket edge", l) and "Reading:" not in l]
    return [tr(l) for l in keep[:8]]


def lag_block(log: str) -> list[str]:
    # Only the actionable verdicts (rare real lag + corroborate-first watch); the
    # priced / liquid-unmoved / model-bias rows are no-edge noise, skip them.
    keep = [l for l in log.splitlines() if re.search(r"(LAG\?|watch):", l)]
    return [tr(l) for l in keep[:6]]


def news_rows(conn, limit: int = 6) -> list[str]:
    # Pull a wide set, then DEDUPE by (team, player) so 6 near-identical Gilmour
    # headlines collapse to one. Rows WITH a Chinese gloss sort first so the kept
    # representative is the translated one.
    rows = conn.execute("""
        SELECT COALESCE(llm_team, team_code) team,
               COALESCE(llm_severity, severity) sev,
               llm_title_zh, title, llm_player player, llm_impact_type itype,
               ROUND((julianday('now') - julianday(published)) * 24) age_h
        FROM news_alerts
        WHERE COALESCE(llm_severity, severity) >= 4
          AND COALESCE(llm_impact_type, '') NOT IN ('recovery', 'other')
        ORDER BY COALESCE(llm_severity, severity) DESC, (llm_title_zh IS NULL), id DESC
    """).fetchall()
    itag = {"squad_change": " ·硬缺阵", "suspension": " ·停赛"}
    seen: set = set()
    out: list[str] = []
    for r in rows:
        # Dedup by (team, player SURNAME/first-token) so "Neymar" and "Neymar
        # Junior" collapse; fall back to a title prefix when no player.
        pkey = r["player"].split()[0].lower() if r["player"] else None
        key = (r["team"], pkey) if pkey else ("_t", (r["title"] or "")[:30].lower())
        if key in seen:
            continue
        seen.add(key)
        title = r["llm_title_zh"] or (r["title"] or "")[:48]
        out.append(f"严重度{r['sev']} · {team_zh(r['team'])} · {int(r['age_h'] or 0)}h前"
                   f"{itag.get(r['itype'], '')} · {title}")
        if len(out) >= limit:
            break
    return out


def watch_rows(conn) -> list[str]:
    out = []
    for w in morale_watchlist(conn):
        tag = (f"置信{w['conf']:.2f}" if w["conf"] is not None else "已评分") if w["scored"] else "未核验"
        title = w["title_zh"] or (w["title"] or "")[:48]
        out.append(f"• {team_zh(w['team'])} [{tag}] {title}")
    return out


def tactics_block(conn, limit: int = 4) -> list[str]:
    """Read-only: upcoming WC fixtures' tactical briefs (pre-gen'd by scout --daily).
    Joins daily_tactics (date/leans/verdict) with match_tactics (the full推演 prose:
    clash / script / swing factor / per-prop reasons) for a detailed per-fixture block."""
    try:
        # Only fixtures that haven't kicked off yet. Prefer the precise kickoff_utc
        # (so a match already played today drops off); fall back to the date for any
        # row still missing a kickoff time.
        rows = conn.execute("""
            SELECT d.home, d.away, d.match_date, d.arch_h, d.arch_a, d.q_lean, d.t_tg,
                   d.agree, d.corners, d.cards, d.conf, d.m_tg, d.m_pover, d.top_scores,
                   m.kickoff_utc, t.payload
            FROM daily_tactics d LEFT JOIN match_tactics t
              ON t.home = d.home AND t.away = d.away
            LEFT JOIN matches m
              ON m.home_code = d.home AND m.away_code = d.away AND m.match_date = d.match_date
            WHERE COALESCE(m.finished, 0) = 0
              AND COALESCE(m.kickoff_utc, d.match_date || 'T23:59:00+00:00') >= strftime('%Y-%m-%dT%H:%M:%S+00:00','now')
            ORDER BY COALESCE(m.kickoff_utc, d.match_date) LIMIT ?""", (limit,)).fetchall()
    except Exception:
        return []  # tables not created yet (scout --daily never run)
    out: list[str] = []
    for r in rows:
        mu = {}
        if r["payload"]:
            try:
                mu = json.loads(r["payload"])
            except json.JSONDecodeError:
                mu = {}
        bet = mu.get("betting", {})
        gs = mu.get("game_states", {})
        md = kickoff_cn(r["kickoff_utc"], r["match_date"])
        cv = r["conf"]
        conf = ("" if cv is None else
                "[把握较高]" if cv >= 0.75 else ("[把握一般]" if cv >= 0.55 else "[纯参考]"))
        # 总进球三方对照 — 按基准来源如实措辞(市场有态度 > 市场五五开 > 无盘回落模型),
        # 不再把市场五五开整场弃判,也不再在无盘回落模型时谎称"市场"。
        m, q, t = r["m_tg"], r["q_lean"], r["t_tg"]
        if m and m != "neutral" and r["m_pover"] is not None:
            base = f"市场{lean_zh(m)}(P大球{r['m_pover']*100:.0f}%)"
            if t and t == m:
                verdict = f"✓战术和市场都看{lean_zh(t)}——没有下注角度,但剧本一致,放心看"
            elif t and t != "neutral":
                verdict = f"⚑战术看{lean_zh(t)}、市场看{lean_zh(m)}——逆市场要有它没消化的料,先当看点"
            else:
                verdict = "战术没表态,跟市场走"
        elif m == "neutral" and r["m_pover"] is not None:
            base = f"市场五五开(P大球{r['m_pover']*100:.0f}%)"
            if t and q and q != "neutral" and t == q:
                verdict = f"盘口没态度;模型和战术都看{lean_zh(t)}(弱信号)"
            elif t and q and q != "neutral" and t != "neutral":
                verdict = f"盘口没态度;模型{lean_zh(q)}/战术{lean_zh(t)}各执一词,纯看球"
            else:
                verdict = "盘口没态度,纯看球"
        else:
            base = f"无市场盘·模型{lean_zh(q)}"
            if t and q and q != "neutral" and t != "neutral":
                verdict = ("✓与模型方向一致(弱基准,当参考)" if t == q
                           else "⚑与模型相左(弱基准,当故事听)")
            else:
                verdict = "弱基准,当参考"
        out.append(f"▌{md} {team_zh(r['home'])}({r['arch_h'] or '?'}) × {team_zh(r['away'])}({r['arch_a'] or '?'}) {conf}")
        if mu.get("clash_zh"):
            out.append(f"  相撞: {mu['clash_zh']}")
        if mu.get("script_zh"):
            out.append(f"  剧本: {mu['script_zh']}")
        for kb in (mu.get("key_battles_zh") or [])[:2]:
            out.append(f"  对位: {kb}")
        if gs.get("home_first_zh") or gs.get("away_first_zh"):
            hf = (gs.get("home_first_zh") or "?").rstrip("。;；")
            out.append(f"  领先后: {team_zh(r['home'])}先进→{hf}；"
                       f"{team_zh(r['away'])}先进→{gs.get('away_first_zh','?')}")
        if bet.get("swing_factor_zh"):
            out.append(f"  变量: {bet['swing_factor_zh']}")
        out.append(f"  总进球: {base} · 战术{lean_zh(t)} → {verdict}")
        if r["top_scores"]:
            # 百分比一起给:最可能的比分也只有 ~1/8 的命中率,这是诚实也是彩票心态管理
            out.append(f"  比分嗅觉(DC模型): {r['top_scores']}")
        cw = bet.get("corners", {}).get("why_zh", "")
        kw = bet.get("cards", {}).get("why_zh", "")
        out.append(f"  角球{lean_zh(r['corners'])}（{cw}）· 牌{lean_zh(r['cards'])}（{kw}）")
        out.append("")  # blank line between fixtures
    return out


# ── digest section toggles ───────────────────────────────────────────────────
# User (2026-06-01) chose ENJOYMENT-FIRST ("B"): the WC is once every 4 years, not an
# EV grind ("if I wanted to squeeze EV I'd go to the stock market"). So cross-book
# arbitrage is dropped from the push and the digest leads with the football read.
# Flip these to re-enable.
INCLUDE_ARBITRAGE = False
INCLUDE_POLYMARKET = False  # dropped 2026-06-04 (user: enjoyment-first, no EV-grind either)


def md3_lines(conn, limit: int = 8) -> list[str]:
    """末轮出线利益(死亡之组)— 只显示已可判定的(组内前两轮打完);赛前为空。"""
    try:
        board = md3_board(conn)
    except Exception:
        return []
    live = [b for b in board if b.get("state") != "待前两轮"]
    out = []
    for b in live[:limit]:
        tag = {"under": "↓小球", "over": "↑大球", "neutral": "中性"}.get(b["lean"], "")
        out.append(f"[{b['group']}] {b['date'][5:]} {team_zh(b['home'])}-{team_zh(b['away'])} 【{b['state']}】{tag}")
        out.append(f"   {b['note']}")
    return out


def _freshness_lines(conn) -> list[str]:
    """数据时点水印:盘口快照/推演生成各是什么时候,>24h 标⚠。
    管线静默断档过 9 天而简报毫无表示——水印让断档当天就被肉眼看见。"""
    def _parse(ts):
        if not ts:
            return None
        try:
            d = dt.datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None
        # sqlite datetime('now') stores naive UTC
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)

    now = dt.datetime.now(dt.timezone.utc)
    cn = dt.timezone(dt.timedelta(hours=8))
    bits, stale = [], False
    for label, sql in (("盘口快照", "SELECT MAX(captured_at) FROM odds"),
                       ("推演生成", "SELECT MAX(generated_at) FROM daily_tactics")):
        try:
            d = _parse(conn.execute(sql).fetchone()[0])
        except Exception:
            d = None
        if d is None:
            bits.append(f"{label}=无")
            continue
        age_h = (now - d).total_seconds() / 3600
        tag = ""
        if age_h > 24:
            tag = f" ⚠{age_h/24:.0f}天前"
            stale = True
        bits.append(f"{label} {d.astimezone(cn):%m-%d %H:%M}{tag}")
    out = ["数据: " + " · ".join(bits) + " (北京时间)"]
    if stale:
        out.append("⚠ 有数据超过24h没刷新——管线可能没跑,相关数字请打折看")
    return out


def review_block(conn, limit: int = 6) -> list[str]:
    """昨日对答案 — graded 推演 vs reality + a cumulative scoreboard.
    Reads tactics_review (written by eval.tactics_review). Empty when no WC fixture
    has finished yet, in which case compose() omits the whole section."""
    try:
        from worldcup.eval.tactics_review import scoreboard
        rows = conn.execute("""SELECT * FROM tactics_review
            WHERE match_date >= date('now', '-2 day')
            ORDER BY match_date DESC, reviewed_at DESC LIMIT ?""", (limit,)).fetchall()
    except Exception:
        return []  # table not created yet (review never run)
    if not rows:
        return []

    def mk(hit):
        return {1: "✓", 0: "✗"}.get(hit, "—")

    out: list[str] = []
    for r in rows:
        head = f"▌{team_zh(r['home'])} {r['home_score']}-{r['away_score']} {team_zh(r['away'])}"
        tg = (f"总进球{r['total_goals']}: 战术{lean_zh(r['t_tg'])}{mk(r['hit_t'])} "
              f"模型{lean_zh(r['q_lean'])}{mk(r['hit_q'])} 市场{lean_zh(r['m_tg'])}{mk(r['hit_m'])}")
        out.append(f"{head} │ {tg}")
        extras = []
        if r["total_corners"] is not None:
            extras.append(f"角球{r['total_corners']} {lean_zh(r['corners_lean'])}{mk(r['hit_corners'])}")
        if r["total_cards"] is not None:
            extras.append(f"牌{r['total_cards']} {lean_zh(r['cards_lean'])}{mk(r['hit_cards'])}")
        if extras:
            out.append("  " + " · ".join(extras))
    sb = scoreboard(conn)
    tg_board = " · ".join(f"{k} {h}/{g}" for k, (h, g) in sb.items()
                          if g and k in ("战术", "模型", "市场"))
    if tg_board:
        out.append(f"累计总进球命中: {tg_board}")
    return out


def _esc(s: str) -> str:
    """Escape for Telegram HTML (only & < > are special in text nodes)."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _spoiler_scores(line: str) -> str:
    """Wrap the score list of a 比分嗅觉 line in a spoiler (escaped), keep label plain."""
    label, sep, scores = line.partition(": ")
    if not sep:
        return _esc(line)
    return f"{_esc(label)}: <tg-spoiler>{_esc(scores)}</tg-spoiler>"


def _html_card_section(title: str, lines: list[str]) -> list[str]:
    """Render the tactics section as Telegram cards: bold header per fixture, the rest
    folded into an <blockquote expandable>. Each card (header + its blockquote) is one
    block separated by a blank line, so _chunks_html can split BETWEEN cards but never
    inside one (header and its blockquote stay together). Cards split on the '▌' marker."""
    out = [f"<b>{_esc(title)}</b>", ""]   # title is its own block
    header: str | None = None
    detail: list[str] = []

    def flush():
        nonlocal header
        if header is None:
            return
        card = header
        if detail:
            card += "\n<blockquote expandable>" + "\n".join(detail) + "</blockquote>"
        out.append(card)
        out.append("")                   # blank line → card boundary for chunking
        header, detail[:] = None, []

    for ln in lines:
        if ln.startswith("▌"):
            flush()
            header = f"<b>{_esc(ln)}</b>"
        elif ln.strip() == "":
            continue
        elif header is not None:
            detail.append(_spoiler_scores(ln) if "比分嗅觉" in ln else _esc(ln))
        else:                            # e.g. the "窗口内无 WC 比赛" empty placeholder
            out.append(_esc(ln)); out.append("")
    flush()
    return out


def qualification_block(conn, n_sims: int = 20000) -> list[str]:
    """出线形势 + 出线动机 — conditional MC projection per group (once group games
    start). Each group is one card: standings + P(advance) + incentive label.
    Returns [] before any group game is played or if the model can't fit."""
    try:
        from worldcup.strategy.qualification import board, zh
    except Exception:
        return []
    try:
        played = conn.execute("""SELECT COUNT(*) FROM matches WHERE finished=1
            AND match_date>='2026-06-11' AND stage LIKE 'Group Stage%'""").fetchone()[0]
        if not played:
            return []
        b = board(conn, n_sims=n_sims)
    except Exception:
        return []
    out: list[str] = []
    for g, gd in b["groups"].items():
        if all(r["played"] == 0 for r in gd["rows"]):
            continue  # this group hasn't kicked off — skip
        tag = "已收官" if gd["all_played"] else f"剩{len(gd['remaining'])}场"
        out.append(f"▌{g}组 ({tag})")
        for r in gd["rows"]:
            out.append(f"  {zh(r['team'])} {r['pts']}分 净{r['gd']:+d} · 出线{r['p_adv']*100:.0f}%"
                       f"·夺头名{r['p_win']*100:.0f}% → {r['incentive']}({r['incentive_note']})")
        out.append("")
    return out


def compose(bets_log: str = "", html: bool = False) -> str:
    log = ""
    if bets_log and Path(bets_log).exists():
        log = Path(bets_log).read_text(errors="ignore")
    conn = get_conn(DEFAULT_DB_PATH)
    try:
        news, watch, tactics = news_rows(conn), watch_rows(conn), tactics_block(conn)
        md3 = md3_lines(conn)
        fresh = _freshness_lines(conn)
        review = review_block(conn)
        qualif = qualification_block(conn)
        try:
            from worldcup.strategy.form_scan import digest_lines as _fs
            formscan = _fs(conn)
        except Exception:
            formscan = []
    finally:
        conn.close()

    today = dt.date.today()
    wd = "一二三四五六日"[today.weekday()]  # Mon=0 … Sun=6
    title_line = f"🌍 世界杯2026 每日雷达  {today:%Y-%m-%d} 周{wd}"

    # Collect sections as (title, lines, empty, is_cards); render plain or HTML.
    sections: list[tuple[str, list[str], str, bool]] = []
    if review:  # 只在有已完赛的 WC 比赛后出现
        sections.append(("── 昨日对答案 (推演 vs 实际;对了不吹,错了认账)──", review, "", False))
    sections.append(("── 战术推演 (打法/剧本/变量;⚑分歧才有下注角度)──", tactics, "窗口内无 WC 比赛", True))
    if qualif:  # 小组赛开打后:条件化出线概率 + 出线动机
        sections.append(("── 出线形势 + 出线动机 (基于已踢结果模拟)──", qualif, "", True))
    if formscan:  # 进球 vs xG 回归观察名单(软盘假设,非绿灯)
        sections.append(("── 状态扫描 (进球vsxG;⚑虚高可fade / ⚑被低估可背,仍要过门禁)──", formscan, "", False))
    sections.append(("── 末轮出线利益 (默契平/生死对攻/走过场 → 总进球)──", md3, "待小组前两轮打完后激活", False))
    sections.append(("── 新闻 / 伤病 / 硬缺阵 ──", news, "近期无 严重度≥4 伤病/缺阵", False))
    sections.append(("── 内讧/士气(看球谈资,不下注)──", watch, "近 14 天无", False))
    sections.append(("── 伤病滞后(薄盘未再价,FYI)──", lag_block(log), "无", False))
    if INCLUDE_POLYMARKET:
        sections.append(("── Polymarket 价差(NO=偏贵该卖 / YES=偏便宜该买;真值=Betfair)──", poly_block(log), "无价差", False))
    if INCLUDE_ARBITRAGE:
        sections.append(("── 跨书套利 & 最优赔率 ──", arb_block(log), "今日无套利 / 最优线", False))
    footer = "娱乐为主:小注 · 单注优先(串关 EV 最差)· 每注记 CLV · 亏完即止不补仓。"

    if not html:
        parts = [title_line, *fresh, ""]
        for ttl, lines, empty, _cards in sections:
            parts.append(ttl)
            parts.extend(lines if lines else [f"  {empty}"])
            parts.append("")
        parts.append(footer)
        return "\n".join(parts)

    # HTML render (Telegram parse_mode=HTML): bold headers, tactical cards folded.
    parts = [f"<b>{_esc(title_line)}</b>", *[_esc(f) for f in fresh], ""]
    for ttl, lines, empty, cards in sections:
        if cards:
            parts.extend(_html_card_section(ttl, lines or [f"  {empty}"]))
        else:
            parts.append(f"<b>{_esc(ttl)}</b>")
            parts.extend(_esc(l) for l in (lines if lines else [f"  {empty}"]))
            parts.append("")
    parts.append(f"<i>{_esc(footer)}</i>")
    return "\n".join(parts)


def main() -> None:
    args = [a for a in sys.argv[1:]]
    html = "--html" in args
    bets = next((a for a in args if not a.startswith("--")), "")
    print(compose(bets, html=html))


if __name__ == "__main__":
    main()
