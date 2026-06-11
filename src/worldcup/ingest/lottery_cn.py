"""China Sports Lottery (足彩) ingester — 胜负彩 / 任选九 pari-mutuel results.

WHY THIS MATTERS (a genuinely different pricing logic):
    足彩 pools are PARI-MUTUEL (奖池按中奖注数均分) — priced by the CROWD's betting
    distribution, NOT by a bookmaker. Payout per winning ticket = pool / #winners.
    This is the ONE market our results-based model can plausibly beat: the public,
    not sharp books (which it can't). Edge = back correctly-assessed outcomes the
    crowd UNDER-bets → fewer co-winners → bigger share. The wall: ~31-35% take.

WHAT THIS PULLS:
    Historical draw results (14 W/D/L), first/second-prize winner counts + per-ticket
    payouts, and sales — for 胜负彩 (14/14) and 任选九 (9 of 14). These let us measure
    (a) the effective return rate (the take wall, incl. jackpot rollover) and (b) the
    "upset premium" (upset-heavy rounds → fewer winners → far higher payout).

SOURCE / LIMITS:
    East Money (caipiao.eastmoney.com) — SERVER-RENDERED HTML, ~5 latest issues per
    fetch (the deeper 45 issues sit behind a JS dropdown with no loopable URL).
    So this ACCUMULATES over time (INSERT OR IGNORE by issue), like our other
    ingesters. Per-match crowd 选择比例 is NOT here (it's on JS-rendered 500/竞彩网) —
    pool_share.py uses an odds-based proxy until that's wired in (Chrome MCP / manual).

Run:
    PYTHONPATH=src python -m worldcup.ingest.lottery_cn
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

SFC_URL = "https://caipiao.eastmoney.com/Result/Category/sfc"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

DDL = """
CREATE TABLE IF NOT EXISTS lottery_draws (
    issue       TEXT NOT NULL,           -- 期号 e.g. '26081'
    game        TEXT NOT NULL,           -- 'sfc' (胜负彩 14/14) | 'r9' (任选九 9of14)
    draw_date   TEXT,
    results     TEXT,                    -- 14 outcomes space-joined: '3'=home '1'=draw '0'=away
    first_n     INTEGER,                 -- 一等奖 中奖注数
    first_pay   REAL,                    -- 一等奖 单注奖金 (元)
    second_n    INTEGER,                 -- 二等奖 中奖注数 (sfc only; 猜中13场)
    second_pay  REAL,
    sales       REAL,                    -- 该玩法投注金额 (元)
    captured_at TEXT,
    PRIMARY KEY (issue, game)
);
"""


def _num(s: str | None) -> float | None:
    if not s:
        return None
    s = s.replace(",", "").replace("元", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def fetch_sfc_page(client: httpx.Client) -> str:
    r = client.get(SFC_URL, headers=UA, timeout=20, follow_redirects=True)
    r.raise_for_status()
    return r.text


def parse_draws(html: str) -> list[dict]:
    """Parse East Money's 胜负彩 category page → one dict per (issue, game)."""
    soup = BeautifulSoup(html, "lxml")
    out: list[dict] = []
    for panel in soup.select("div.tabs-panel"):
        issue = (panel.get("id") or "").strip()
        if not re.fullmatch(r"\d{5}", issue):
            continue
        info = panel.select_one(".lottery-result-info")
        info_txt = info.get_text(" ", strip=True) if info else ""
        mdate = re.search(r"(\d{4}-\d{2}-\d{2})", info_txt)
        draw_date = mdate.group(1) if mdate else None
        msfc = re.search(r"胜负彩投注金额[:：]\s*([\d.]+)万", info_txt)
        mr9 = re.search(r"任选?9场投注金额[:：]\s*([\d.]+)万", info_txt)
        sfc_sales = float(msfc.group(1)) * 1e4 if msfc else None
        r9_sales = float(mr9.group(1)) * 1e4 if mr9 else None

        # ── results (the table whose single row is 14 one-digit cells) ──
        results = None
        prize_rows: list[list[str]] = []
        for tb in panel.select("table"):
            trs = [[td.get_text(strip=True) for td in tr.select("td")]
                   for tr in tb.select("tr") if tr.select("td")]
            if not trs:
                continue
            if results is None and len(trs) == 1 and len(trs[0]) >= 14 \
               and all(c in ("0", "1", "3") for c in trs[0][:14]):
                results = " ".join(trs[0][:14])
            else:
                prize_rows.extend(trs)

        # ── prize tiers: classify rows by 中奖条件 (猜中14场 / 13场 / 9场) ──
        sfc_1n = sfc_1p = sfc_2n = sfc_2p = r9_1n = r9_1p = None
        for row in prize_rows:
            cond = " ".join(row)
            count = _num(row[1]) if len(row) > 1 else None
            pay = _num(row[2]) if len(row) > 2 else None
            if pay is None:        # a real prize tier always has a 单注奖金 column;
                continue           # rows without it (summary/header) are misreads
            if "14场" in cond or "一等奖" in row[0]:
                sfc_1n, sfc_1p = count, pay
            elif "13场" in cond or "二等奖" in row[0]:
                sfc_2n, sfc_2p = count, pay
            elif "9场" in cond:
                r9_1n, r9_1p = count, pay

        out.append({"issue": issue, "game": "sfc", "draw_date": draw_date,
                    "results": results, "first_n": sfc_1n, "first_pay": sfc_1p,
                    "second_n": sfc_2n, "second_pay": sfc_2p, "sales": sfc_sales})
        if r9_1n is not None or r9_sales is not None:
            out.append({"issue": issue, "game": "r9", "draw_date": draw_date,
                        "results": results, "first_n": r9_1n, "first_pay": r9_1p,
                        "second_n": None, "second_pay": None, "sales": r9_sales})
    return out


def ingest_lottery_cn(db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, Any]:
    conn = get_conn(db_path)
    try:
        conn.executescript(DDL)
        captured = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        with httpx.Client() as client:
            draws = parse_draws(fetch_sfc_page(client))
        before = conn.execute("SELECT COUNT(*) FROM lottery_draws").fetchone()[0]
        for d in draws:
            conn.execute(
                """INSERT INTO lottery_draws
                   (issue, game, draw_date, results, first_n, first_pay,
                    second_n, second_pay, sales, captured_at)
                   VALUES (:issue,:game,:draw_date,:results,:first_n,:first_pay,
                           :second_n,:second_pay,:sales,:cap)
                   ON CONFLICT(issue, game) DO UPDATE SET
                     draw_date=excluded.draw_date, results=excluded.results,
                     first_n=excluded.first_n, first_pay=excluded.first_pay,
                     second_n=excluded.second_n, second_pay=excluded.second_pay,
                     sales=COALESCE(excluded.sales, sales)""",
                {**d, "cap": captured},
            )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM lottery_draws").fetchone()[0]
        total = conn.execute("SELECT COUNT(DISTINCT issue) FROM lottery_draws").fetchone()[0]
        return {"parsed": len(draws), "new_rows": after - before,
                "total_rows": after, "distinct_issues": total}
    finally:
        conn.close()


def main() -> None:
    import json
    print("=== China Sports Lottery (足彩) ingest ===")
    print(json.dumps(ingest_lottery_cn(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
