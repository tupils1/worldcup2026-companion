"""LLM-based news scoring via DeepSeek API.

Replaces keyword-based classifier in news_monitor.py for higher accuracy:
    - Understands context (e.g. "England hopes boosted as Ghana's Kudus out"
      → identifies Ghana, NOT England, as the affected team)
    - Extracts player name reliably
    - Calibrates λ multipliers based on player importance + injury severity
    - Filters out recovery / non-news ("X returns from injury") cleanly

API: OpenAI-compatible. Set DEEPSEEK_API_KEY in configs/secrets.env.
Cost: ~$0.0005-0.001 per news item (cheap enough to score 100+/day).

Falls back to keyword scoring if API key absent or LLM call fails — so the
pipeline always produces SOMETHING, just lower accuracy without LLM.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

DEEPSEEK_BASE = "https://api.deepseek.com/v1"
SECRETS_PATH = Path(__file__).resolve().parents[3] / "configs" / "secrets.env"

WC_TEAMS = (
    "ARG ALG AUS AUT BEL BIH BRA CAN CIV COD COL CPV CRO CUW CZE ECU EGY "
    "ENG ESP FRA GER GHA HAI IRN IRQ JOR JPN KOR KSA MAR MEX NED NOR NZL "
    "PAN PAR POR QAT RSA SCO SEN SUI SWE TUN TUR URU USA UZB"
).split()

SYSTEM_PROMPT = f"""You are an analyst specializing in 2026 FIFA World Cup team news. Given a news headline + summary, extract structured impact on the affected national team.

WC 2026 has 48 teams (3-letter FIFA codes):
{', '.join(WC_TEAMS)}

Output STRICT JSON only (no markdown, no extra text):
{{
  "team_code": "<one of the 48 codes, or null if no WC team affected>",
  "player_name": "<full name, or null if no specific player mentioned>",
  "severity": <integer 1-5>,
  "impact_type": "<injury | suspension | squad_change | form_concern | recovery | other>",
  "suggested_attack_multiplier": <float 0.50 to 1.00>,
  "suggested_defense_multiplier": <float 0.50 to 1.00>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one-sentence justification>",
  "title_zh": "<≤20-字 中文事件摘要, e.g. 苏格兰吉尔莫膝伤无缘世界杯 / 西班牙亚马尔与巴萨不和>",
  "player_zh": "<球员中文名(用通行译名), e.g. 三笘薰 / 阿方索·戴维斯 / 内马尔; null 若无具体球员>"
}}

Severity rubric:
- 5: KEY starter definitely OUT for the tournament (e.g. ACL tear, season-ending)
- 4: KEY starter likely out for tournament OR doubtful with major injury
- 3: Rotation player out; OR minor injury concern for key player ("doubt", "fitness test")
- 2: Squad announcement, minor news, single match doubt
- 1: Marginal/speculative news, recovery, training updates

Multiplier rubric (λ scaling in Dixon-Coles):
- 0.65-0.75: Catastrophic loss (top scorer / playmaker out for tournament)
- 0.80-0.90: Major loss (regular starter for tournament)
- 0.90-0.97: Moderate (rotation player out, key player one match)
- 0.98-1.00: Marginal / no impact

CRITICAL RULES:
1. The team_code is the NATIONALITY of the AFFECTED player, not whichever country is mentioned in the headline for context. Example: "Ghana's Kudus to miss World Cup — boost for England" → team_code=GHA (not ENG).
2. For "recovery" / "returns from injury" / "passed fitness test" → severity = 1, multipliers = 1.00, impact_type = "recovery".
3. If headline mentions a CLUB (not national team) injury without WC context → team_code = null.
4. Be conservative on severity 4-5: must be clear the player is a tournament starter AND clearly affected.
5. OFF-FIELD events (non-injury):
   - "squad_change" = player CONFIRMED ABSENT for non-injury reasons (sent home, dropped/axed, banned, refuses to play, quits). Treat impact like a real injury: a confirmed-out starter → 0.82-0.90; rotation player → 0.92-0.97.
   - "form_concern" / "other" = morale/feud/rift/dressing-room rumour where the player is STILL EXPECTED TO PLAY. These are noisy, often media speculation, and usually ALREADY priced by the market. Default to multiplier 0.98-1.00 and confidence ≤ 0.4 unless there is concrete on-pitch evidence. Do NOT haircut a team's λ for a reported "rift" alone."""


# ── λ-multiplier clamp: guardrail against brand-name overweighting ───────────
# The LLM systematically overstates single-player injury impact for deep squads.
# Verified live (2026-05-28): it gave Neymar att×0.70, which made conditional MC
# print BRA champion −6.8pp — but the sharp market (Polymarket + 7 books) moved
# Brazil only ~−0.3pp on the same news (≈ att×0.97). A single player essentially
# never removes >25-30% of a national team's goal expectation; for an elite deep
# squad the real single-player ceiling is ~5%. So we FLOOR the LLM's downward
# multiplier by team tier. This bounds magnitude only — never flips direction or
# invents impact.
#
# Tune these if betting philosophy shifts: LOWER floors = trust news over market.
ELITE_DEEP_SQUADS = {"BRA", "FRA", "ENG", "ESP", "ARG", "GER", "POR", "NED", "BEL"}
ELITE_MULT_FLOOR = 0.95   # deep elite squad: one player ≤5% team-strength swing
GLOBAL_MULT_FLOOR = 0.72  # any team: no single player removes >28% of its rate


def clamp_multiplier(team_code: str | None, mult: float | None) -> float | None:
    """Floor an LLM-suggested λ multiplier so a single-player injury can't imply an
    implausibly large team-strength swing. Only clamps DOWNWARD (impact) multipliers
    (< 1.0); returns >=1.0 and None inputs unchanged. Deep elite squads (huge attack
    depth) get a tighter floor than star-dependent smaller sides — e.g. Neymar/BRA is
    raised 0.70→0.95, but Kudus/GHA stays ≈0.72 because Ghana really is built around him."""
    if mult is None:
        return None
    try:
        m = float(mult)
    except (TypeError, ValueError):
        return mult
    if m >= 1.0:
        return m  # only temper downward (impact) multipliers, never upward
    floor = ELITE_MULT_FLOOR if (team_code in ELITE_DEEP_SQUADS) else GLOBAL_MULT_FLOOR
    return max(m, floor)


def load_api_config() -> tuple[str | None, str]:
    """Returns (api_key, model_name)."""
    key = None
    model = "deepseek-chat"
    if SECRETS_PATH.exists():
        for line in SECRETS_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip() or None
            elif line.startswith("DEEPSEEK_MODEL="):
                v = line.split("=", 1)[1].split("#")[0].strip()
                if v: model = v
    # Env override
    if os.environ.get("DEEPSEEK_API_KEY"):
        key = os.environ["DEEPSEEK_API_KEY"]
    if os.environ.get("DEEPSEEK_MODEL"):
        model = os.environ["DEEPSEEK_MODEL"]
    return key, model


def score_item_llm(title: str, summary: str, api_key: str, model: str,
                   timeout: int = 30) -> dict | None:
    """LLM-score one news item. Returns parsed JSON or None on failure."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Title: {title}\n\nSummary: {summary[:600]}"},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }
    try:
        r = httpx.post(
            f"{DEEPSEEK_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload, timeout=timeout,
        )
        if r.status_code != 200:
            return {"_error": f"HTTP {r.status_code}: {r.text[:200]}"}
        content = r.json()["choices"][0]["message"]["content"]
        # Strip markdown if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:150]}"}


def ensure_llm_columns(conn: sqlite3.Connection):
    """Add LLM result columns to news_alerts table if missing."""
    cols = [
        ("llm_team", "TEXT"),
        ("llm_player", "TEXT"),
        ("llm_severity", "INTEGER"),
        ("llm_impact_type", "TEXT"),
        ("llm_attack_mult", "REAL"),
        ("llm_defense_mult", "REAL"),
        ("llm_confidence", "REAL"),
        ("llm_reasoning", "TEXT"),
        ("llm_title_zh", "TEXT"),
        ("llm_player_zh", "TEXT"),
        ("llm_scored_at", "TEXT"),
    ]
    for cname, ctype in cols:
        try:
            conn.execute(f"ALTER TABLE news_alerts ADD COLUMN {cname} {ctype}")
        except sqlite3.OperationalError:
            pass


def rescore_alerts(db_path: Path | str = DEFAULT_DB_PATH,
                   limit: int = 30,
                   only_missing: bool = True,
                   where_sql: str | None = None,
                   verbose: bool = True) -> dict:
    """Re-score recent news_alerts using LLM. Updates news_alerts table.

    where_sql, if given, overrides the row filter (e.g. backfill rows missing the
    Chinese gloss)."""
    key, model = load_api_config()
    if not key:
        return {"error": "DEEPSEEK_API_KEY not set in configs/secrets.env"}

    conn = get_conn(db_path)
    ensure_llm_columns(conn)

    where = where_sql if where_sql is not None else ("WHERE llm_scored_at IS NULL" if only_missing else "")
    alerts = list(conn.execute(f"""
        SELECT id, title, url, source, severity, composite
        FROM news_alerts
        {where}
        ORDER BY composite DESC LIMIT ?
    """, (limit,)))

    if verbose:
        print(f"Scoring {len(alerts)} alerts with DeepSeek model={model} ...")

    results = []
    errors = 0
    t0 = time.time()
    for i, row in enumerate(alerts, 1):
        # Use empty summary fallback — title alone is usually enough
        res = score_item_llm(row["title"], "", key, model)
        if res is None or "_error" in res:
            errors += 1
            if verbose:
                print(f"  [{i}/{len(alerts)}] ERROR: {(res or {}).get('_error', 'unknown')[:80]}")
            continue
        # Temper the LLM multipliers (brand-name overweighting guardrail).
        team = res.get("team_code")
        raw_att = res.get("suggested_attack_multiplier")
        raw_def = res.get("suggested_defense_multiplier")
        att = clamp_multiplier(team, raw_att)
        deff = clamp_multiplier(team, raw_def)
        reasoning = res.get("reasoning") or ""
        clamp_notes = []
        if raw_att is not None and att is not None and abs(att - raw_att) > 1e-6:
            clamp_notes.append(f"att {raw_att:.2f}→{att:.2f}")
        if raw_def is not None and deff is not None and abs(deff - raw_def) > 1e-6:
            clamp_notes.append(f"def {raw_def:.2f}→{deff:.2f}")
        if clamp_notes:
            tier = "elite-deep-squad" if team in ELITE_DEEP_SQUADS else "single-player ceiling"
            reasoning = f"{reasoning} [clamped: {', '.join(clamp_notes)} — {tier} guardrail]"
        conn.execute("""
            UPDATE news_alerts SET
                llm_team = ?, llm_player = ?, llm_severity = ?,
                llm_impact_type = ?, llm_attack_mult = ?, llm_defense_mult = ?,
                llm_confidence = ?, llm_reasoning = ?, llm_title_zh = ?,
                llm_player_zh = ?, llm_scored_at = datetime('now')
            WHERE id = ?
        """, (
            team, res.get("player_name"), res.get("severity"),
            res.get("impact_type"), att, deff, res.get("confidence"),
            reasoning, res.get("title_zh"), res.get("player_zh"), row["id"],
        ))
        results.append({**dict(row), **res,
                        "suggested_attack_multiplier": att,
                        "suggested_defense_multiplier": deff})
        if verbose and i % 10 == 0:
            print(f"  [{i}/{len(alerts)}] {time.time()-t0:.1f}s elapsed")
        time.sleep(0.1)  # avoid rate-limit

    conn.close()
    return {
        "scored": len(results),
        "errors": errors,
        "elapsed_sec": time.time() - t0,
        "model": model,
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--all", action="store_true",
                    help="Re-score all alerts (otherwise only those without LLM scores)")
    ap.add_argument("--missing-zh", action="store_true",
                    help="Backfill rows missing the Chinese gloss (only digest-visible: sev≥4 or morale_watch)")
    args = ap.parse_args()

    key, model = load_api_config()
    if not key:
        print("⚠️  DEEPSEEK_API_KEY not set. Add it to configs/secrets.env first.")
        print("    Example: DEEPSEEK_API_KEY=sk-...")
        return

    print(f"=== DeepSeek LLM News Re-Score ===")
    print(f"Model: {model}\n")
    where_sql = ("WHERE (llm_title_zh IS NULL OR llm_player_zh IS NULL) AND "
                 "(COALESCE(llm_severity,severity) >= 3 OR impact_category = 'morale_watch')"
                 ) if args.missing_zh else None
    result = rescore_alerts(limit=args.limit, only_missing=not args.all, where_sql=where_sql)
    print(f"\nScored: {result.get('scored', 0)}  errors: {result.get('errors', 0)}  "
          f"elapsed: {result.get('elapsed_sec', 0):.1f}s")

    # Show comparison: keyword vs LLM
    conn = sqlite3.connect(str(DEFAULT_DB_PATH))
    conn.row_factory = sqlite3.Row
    print(f"\n=== Compare KEYWORD vs LLM scoring ===\n")
    print(f"{'title':<60}  {'kw-team':>8} {'kw-sev':>7} | "
          f"{'llm-team':>8} {'llm-sev':>7} {'mult':>5} {'verdict'}")
    print("-" * 130)
    rows = list(conn.execute("""
        SELECT title, team_code AS kw_team, severity AS kw_sev,
               llm_team, llm_severity, llm_attack_mult, llm_player, llm_reasoning
        FROM news_alerts
        WHERE llm_scored_at IS NOT NULL
        ORDER BY llm_severity DESC NULLS LAST, composite DESC LIMIT 15
    """))
    for r in rows:
        title = (r["title"] or "")[:60]
        same = r["kw_team"] == r["llm_team"]
        verdict = "✓ agree" if same else f"⚠️ disagree (LLM correct?)"
        if r["llm_team"] is None:
            verdict = "❌ LLM says NO WC impact (kw false positive)"
        mult = r["llm_attack_mult"] or 0
        print(f"{title:<60}  {r['kw_team'] or '-':>8} {r['kw_sev'] or 0:>7} | "
              f"{r['llm_team'] or '-':>8} {r['llm_severity'] or 0:>7} {mult:>5.2f}  {verdict}")
    conn.close()


if __name__ == "__main__":
    main()
