"""Opportunity radar — the daily driver for the 'opportunistic + discipline' path.

You can bet only Polymarket + 竞彩/任九, and the honest conclusion is: no steady alpha
pump — edge is OPPORTUNISTIC (catch rare mispricing/lag windows) and the real money-saver
is DISCIPLINE (don't make −EV bets). This one command:

  1. Prints the BET GATE (every bet must pass ALL points, else NO BET — the default).
  2. Runs the FREE, bettable-venue scanners and shows ONLY actionable signals:
       - Polymarket crypto vs Deribit options (fully live, free).
       - Polymarket football vs Betfair (only if Betfair odds in DB are still fresh —
         warns if stale, e.g. after cancelling The Odds API).
       - Top injury/news alerts → watch for SLOW-venue (Polymarket/竞彩) lag.
  3. Reminds the manual checks (竞彩 选择比例 / odds via Chrome MCP near deadlines).

Run:
    PYTHONPATH=src python -m worldcup.strategy.radar --bankroll 1000
"""

from __future__ import annotations

import argparse
import datetime as dt

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

GATE = [
    "1. EDGE SOURCE: a sharper benchmark (Pinnacle/Betfair/Deribit/options/futures) says it's"
    " mispriced, OR you genuinely know more than the marginal bettor. Just 'my view/model'? → NO BET.",
    "2. NOT vs SHARP: you are NOT betting against the sharp consensus on a LIQUID market"
    " (that's your bias, not edge — the Neymar/CAN/PSG-Arsenal lesson).",
    "3. CLEARS COST: net edge survives vig/spread/竞彩-take(13-35%)/gas. Polymarket: ≥1.5pp.",
    "4. RESOLUTION: you read the EXACT resolution wording (esp. Polymarket UMA) — no ambiguity"
    " that could settle against you while you're 'right'.",
    "5. FILLABLE: enough liquidity to get your size at that price.",
    "6. SIZED: ¼-Kelly, capped 3% of bankroll. No exceptions, no 'this one's a lock'.",
    "7. CORROBORATED: single-source news → wait/verify before sizing up.",
]


def betfair_freshness(conn) -> tuple[str | None, float | None]:
    row = conn.execute(
        "SELECT MAX(captured_at) FROM odds WHERE bookmaker LIKE 'betfair_ex_%' "
        "AND market_scope='outright' AND market='winner'"
    ).fetchone()
    if not row or not row[0]:
        return None, None
    try:
        ts = dt.datetime.fromisoformat(row[0].replace("Z", "+00:00"))
        age_h = (dt.datetime.now(dt.timezone.utc) - ts).total_seconds() / 3600.0
        return row[0], age_h
    except Exception:
        return row[0], None


def top_news(conn, limit=5) -> list[dict]:
    try:
        rows = conn.execute("""
            SELECT llm_team team, llm_player player, COALESCE(llm_severity,severity) sev,
                   llm_impact_type itype, title
            FROM news_alerts
            WHERE COALESCE(llm_severity,severity) >= 4
              AND COALESCE(llm_impact_type,'') NOT IN ('recovery','other')
            ORDER BY COALESCE(llm_severity,severity) DESC, id DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def morale_watchlist(conn, limit=6, days=14) -> list[dict]:
    """Off-field morale / feud items — DISPLAY-ONLY (att locked 1.0, never a bet input).
    Auto-drops rows the LLM has scored AND rejected as not-a-WC-team (the noise); keeps
    LLM-confirmed-team items + not-yet-scored ones (flagged unverified)."""
    try:
        rows = conn.execute("""
            SELECT COALESCE(llm_team, team_code) team, llm_confidence conf,
                   llm_scored_at scored, title, llm_title_zh title_zh,
                   ROUND((julianday('now') - julianday(published)) * 24, 1) age_h
            FROM news_alerts
            WHERE impact_category = 'morale_watch'
              AND julianday('now') - julianday(captured_at) <= ?
              AND NOT (llm_scored_at IS NOT NULL AND llm_team IS NULL)
            ORDER BY id DESC LIMIT ?
        """, (days, limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--min-edge", type=float, default=0.015)
    ap.add_argument("--stale-hours", type=float, default=24.0,
                    help="Betfair odds older than this → football scan untrusted")
    ap.add_argument("--watchlist-only", action="store_true",
                    help="print ONLY the morale watch-list (for the daily digest); no network scans")
    args = ap.parse_args()

    if args.watchlist_only:
        conn = get_conn(DEFAULT_DB_PATH)
        watch = morale_watchlist(conn)
        conn.close()
        if not watch:
            print("(none in last 14d)")
        for w in watch:
            tag = (f"置信{w['conf']:.2f}" if w["conf"] is not None else "已评分") if w["scored"] else "未核验"
            title = w["title_zh"] or (w["title"] or "")[:54]
            print(f"• {(w['team'] or '?'):5} [{tag}] {title}")
        return

    print("█" * 84)
    print("  OPPORTUNITY RADAR  —  opportunistic + discipline  —  bet ONLY when the GATE passes")
    print("█" * 84)
    print("\n┌─ BET GATE (every bet passes ALL 7, else NO BET — NO BET is the default) ─┐")
    for g in GATE:
        print("  □ " + g)
    print("└" + "─" * 74 + "┘")

    # ── 1. Polymarket crypto vs Deribit (live, free) ──
    print("\n① POLYMARKET CRYPTO  (vs Deribit options-implied — live, free)")
    try:
        from worldcup.strategy.polymarket_crypto import scan as crypto_scan
        cb = crypto_scan(min_edge=0.03, bankroll=args.bankroll)
        if not cb:
            print("   no edge ≥3pp (expected — short-dated digitals are bot-arbed vs Deribit).")
        for b in cb[:8]:
            print(f"   {b['q']:<46} {b['side']} @{b['price']:.2f}  EV {b['ev_pct']:+.1f}%  ${b['stake']:.0f}")
    except Exception as e:
        print(f"   (crypto scan skipped: {type(e).__name__})")

    # ── 2. Polymarket football vs Betfair (freshness-gated) ──
    conn = get_conn(DEFAULT_DB_PATH)
    print("\n② POLYMARKET FOOTBALL  (vs Betfair de-vig — needs fresh Betfair odds)")
    ts, age = betfair_freshness(conn)
    if ts is None:
        print("   no Betfair odds in DB — football truth source unavailable.")
    elif age is not None and age > args.stale_hours:
        print(f"   ⚠ Betfair odds are {age:.0f}h old (>{args.stale_hours:.0f}h) — STALE "
              f"(The Odds API stopped?). Football edge UNTRUSTED until you refresh truth")
        print("     (re-enable Odds API free tier, or scrape 500.com 百家平均/夺冠欧赔 via Chrome MCP).")
    else:
        try:
            from worldcup.strategy.polymarket_lag import scan_edges
            r = scan_edges(bankroll=args.bankroll, min_edge=args.min_edge)
            if not r["bets"]:
                print(f"   no edge ≥{args.min_edge*100:.1f}pp (Betfair odds {age:.0f}h fresh).")
            for b in r["bets"][:8]:
                conf = "✓" if b["agree"] else "⚠book-disagrees"
                print(f"   {b['team']:<5} {b['side']:<8} @{b['price']:.3f}  "
                      f"poly {b['poly_pct']:.1f}% vs sharp {b['betfair_pct']:.1f}%  "
                      f"EV {b['ev_pct']:+.1f}%  ${b['stake']:.0f}  {conf}")
        except Exception as e:
            print(f"   (football scan skipped: {type(e).__name__})")

    # ── 3. News → watch for slow-venue lag ──
    print("\n③ NEWS / INJURY / SQUAD-CHANGE  (watch: did Polymarket/竞彩 NOT reprice yet? = the lag window)")
    news = top_news(conn)
    if not news:
        print("   no fresh sev≥4 injuries / squad-changes. (Run daily_refresh news step to populate.)")
    for n in news:
        itype = f" [{n['itype']}]" if n.get("itype") and n["itype"] != "injury" else ""
        print(f"   sev{n['sev']} {n['team']} {n['player'] or '?'}{itype} — {(n['title'] or '')[:58]}")

    # ── 3·watch. Off-field morale / feud — DISPLAY ONLY, NOT a bet input ──
    watch = morale_watchlist(conn)
    print("\n   ┄ WATCH-LIST: off-field / morale / feud  (NOT a bet signal — λ untouched, att=1.0) ┄")
    if not watch:
        print("     none in last 14d. (feeds: WC disruption / camp tension)")
    for w in watch:
        team = w["team"] or "?"
        tag = (f"conf {w['conf']:.2f}" if w["conf"] is not None else "scored") if w["scored"] else "unverified"
        print(f"     • {team:5} [{tag:10}] {(w['title'] or '')[:56]}")
    if watch:
        print("     → narrative only; usually noise or already priced. Act ONLY if it hardens")
        print("       into a confirmed absence (sent home/dropped/banned) → then it enters ③.")
    conn.close()

    # ── 4. Manual checks ──
    print("\n④ MANUAL (Chrome MCP, near deadlines):")
    print("   • 竞彩/任九: scrape 选择比例 + 比分/总进球 odds → renjiu_ev / jingcai_crossmarket")
    print("     (only fire when crowd is extreme enough to clear the 13-35% take).")
    print("   • Polymarket internal: near-certain markets (≤0.98) = lockup yield; obvious retail")
    print("     overreaction spikes = fade. Both pass the GATE first.")
    print("\nDefault action is NO BET. Act only on a signal that clears the GATE above.")


if __name__ == "__main__":
    main()
