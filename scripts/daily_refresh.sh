#!/usr/bin/env bash
# Daily refresh: pull all data sources + re-detect value bets.
#
# Pre-tournament (now until 2026-06-10): run once a day (morning).
# During tournament (2026-06-11+): run morning + 1h before each match day.
# After matches finish: re-run to refit DC on latest results.
#
# Usage:
#     bash scripts/daily_refresh.sh
#     bash scripts/daily_refresh.sh --quick     # skip historical & xG ingest (faster)
#     bash scripts/daily_refresh.sh --outright-only   # only refresh outright odds + bets
#
# Logs go to data/logs/<YYYY-MM>/<timestamp>-{full,bets}.log

set -euo pipefail

# ─── Setup ───────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use the venv directly without `source activate` (more portable in cron).
PY="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT/src"

MODE="${1:-full}"

TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR="data/logs/$(date +%Y-%m)"
mkdir -p "$LOG_DIR"
FULL_LOG="$LOG_DIR/$TS-full.log"
BETS_LOG="$LOG_DIR/$TS-bets.log"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$FULL_LOG"; }

log "===== Daily refresh ($MODE) ====="

# ─── 1. Ingest sources ───────────────────────────────────────────────────────
if [[ "$MODE" != "--outright-only" ]]; then
    log "[1/5] Elo ratings (eloratings.net)"
    $PY -m worldcup.ingest.elo >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

    if [[ "$MODE" != "--quick" ]]; then
        log "[2/5] Historical matches (martj42)"
        $PY -m worldcup.ingest.historical_matches >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"
    else
        log "[2/5] Historical matches (martj42) — SKIPPED (--quick)"
    fi

    log "[3/5] API-Football fixtures + WC2026"
    $PY -m worldcup.ingest.api_football >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

    # During the tournament (>=06-12) ALWAYS pull post-match stats even in --quick:
    # xG/corners/cards are the data source for the nightly 对答案 review loop.
    if [[ "$MODE" != "--quick" || "$(date +%F)" > "2026-06-11" ]]; then
        log "[4/5] post-match stats (xG proxy + corners/cards) for finished matches"
        $PY -m worldcup.ingest.api_football_stats >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"
    else
        log "[4/5] post-match stats — SKIPPED (--quick, pre-tournament)"
    fi
fi

log "[5/7] The Odds API (match + outright)"
$PY -m worldcup.ingest.odds_api --mode both >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

# Freshness assertion: a stale odds snapshot silently poisons every "market
# baseline" downstream (it already went unnoticed for 9 days once). Warn loudly.
ODDS_AGE_H=$(sqlite3 data/worldcup.db "SELECT CAST((julianday('now')-julianday(MAX(captured_at)))*24 AS INT) FROM odds" 2>/dev/null || echo "")
if [[ "$ODDS_AGE_H" =~ ^[0-9]+$ ]] && (( ODDS_AGE_H > 24 )); then
    log "⚠⚠ odds snapshot is ${ODDS_AGE_H}h old — market baselines below are STALE (odds ingest failing?)"
fi

log "[6/7] Polymarket (prediction market sharp baseline)"
$PY -m worldcup.ingest.polymarket >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

log "[6b] Pinnacle guest API (sharp sub-markets: group winner / to-qualify)"
$PY -m worldcup.ingest.pinnacle >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

log "[7/8] Polymarket edge scanner (sharp truth=Betfair → divergence → Kelly stake; bettable venue)"
$PY -m worldcup.strategy.polymarket_lag --bankroll 1000 --min-edge 0.015 >> "$BETS_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see $BETS_LOG"
echo "" >> "$BETS_LOG"

log "[8/9] News monitor — RSS-based injury/squad alerts"
echo "" >> "$BETS_LOG"
echo "──────────────── NEWS / INJURY ALERTS ────────────────" >> "$BETS_LOG"
$PY -m worldcup.ingest.news_monitor >> "$BETS_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see $BETS_LOG"

log "[9/10] Weather forecast for upcoming WC fixtures (Open-Meteo, 16-day window)"
echo "" >> "$BETS_LOG"
echo "──────────────── WEATHER IMPACT ────────────────" >> "$BETS_LOG"
$PY -m worldcup.ingest.weather >> "$BETS_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see $BETS_LOG"

log "[10/11] LLM re-score news alerts (DeepSeek, skip if no API key)"
echo "" >> "$BETS_LOG"
echo "──────────────── LLM NEWS RE-SCORING ────────────────" >> "$BETS_LOG"
$PY -m worldcup.ingest.llm_news_scorer --limit 30 >> "$BETS_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see $BETS_LOG"

log "[11/13] Injury-lag scan (MC vs market implied-multiplier reconciliation)"
echo "" >> "$BETS_LOG"
echo "──────────────── INJURY-LAG (MC vs market implied multiplier) ────────────────" >> "$BETS_LOG"
# Depends on llm_news_scorer (reads clamped llm multipliers). Material-injury gate
# keeps this to baseline + a few conditional MC runs (~1.5 min).
$PY -m worldcup.strategy.injury_lag --n-sims 10000 --min-sev 4 >> "$BETS_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see $BETS_LOG"

log "[12/13] Market-anchored forecast: champion + group-winner + to-qualify (DC vs de-vigged market)"
echo "" >> "$BETS_LOG"
echo "──────────────── MARKET-ANCHORED (champion + sub-markets; use for sizing, not raw model) ────────────────" >> "$BETS_LOG"
# Lightweight: de-vigged market baseline + baseline MC + group-only sim (~45s, no tilt).
$PY -m worldcup.strategy.market_anchored --n-sims 15000 --submarkets >> "$BETS_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see $BETS_LOG"

log "[13/13] Cross-book line shopping + arbitrage (50 books, model-free alpha)"
echo "" >> "$BETS_LOG"
echo "──────────────── CROSS-BOOK ARBITRAGE & BEST-LINE ────────────────" >> "$BETS_LOG"
$PY -m worldcup.strategy.line_shopping --min-ev 0.02 --top 20 >> "$BETS_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see $BETS_LOG"

# ─── 2. Detect value bets (APPEND, don't overwrite the alpha sections above) ──
log "Running value-bet detection ..."
echo "" >> "$BETS_LOG"
echo "──────────────── OUTRIGHT VALUE BETS (model vs market) ────────────────" >> "$BETS_LOG"
$PY -m worldcup.strategy.value_bets \
    --mode outright \
    --bankroll 1000 \
    --top 20 \
    --prior 0.5 \
    --dc-weight 0.5 \
    --mc-sims 100000 \
    >> "$BETS_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see $BETS_LOG"

log "Done. Value-bet report: $BETS_LOG"
log "Full log:               $FULL_LOG"

# ── Compact daily digest → stdout + file + Telegram push ─────────────────────
log "[13a] API-Football structured injuries (paid /injuries → injuries table; feeds 推演 absences)"
$PY -m worldcup.ingest.api_football_squad >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

log "[13b] 昨日对答案: grade finished 推演 vs reality → tactics_review"
$PY -m worldcup.eval.tactics_review >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

log "[13b'] 回归信号对答案: grade xG over/under-performance flags → form_review"
$PY -m worldcup.eval.form_review >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

log "[13c] Scout tactical briefs for upcoming WC fixtures (DeepSeek; market OU + injuries-grounded → daily_tactics)"
$PY -m worldcup.strategy.scout --daily --days 3 >> "$FULL_LOG" 2>&1 || log "⚠ step FAILED (continuing) — see lines above in $FULL_LOG"

# Plain digest → console + log file (human-readable archive).
DIGEST="$LOG_DIR/$TS-digest.txt"
$PY -m worldcup.notify.daily_digest "$BETS_LOG" | tee "$DIGEST"

# HTML digest (expandable cards) for the Telegram push; fall back to plain on any
# failure so a markup bug can never cost us the digest (HTML rendering can't be
# verified headlessly — the plain push is the safety net).
DIGEST_HTML="$LOG_DIR/$TS-digest.html"
$PY -m worldcup.notify.daily_digest "$BETS_LOG" --html > "$DIGEST_HTML"
if $PY -m worldcup.notify.telegram --file "$DIGEST_HTML" --html >> "$FULL_LOG" 2>&1; then
    log "Telegram digest pushed (HTML cards; or skipped if unconfigured)."
elif $PY -m worldcup.notify.telegram --file "$DIGEST" --plain >> "$FULL_LOG" 2>&1; then
    log "⚠ HTML push failed — fell back to plain text (check markup)."
else
    log "⚠ Telegram push FAILED (both HTML and plain) — see $FULL_LOG."
fi
