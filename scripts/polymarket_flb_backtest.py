"""Polymarket favorite-longshot / retail-mispricing backtest (structural premise check).

Edge premise behind the live polymarket_lag fade: Polymarket RETAIL systematically
overprices favourites (esp. glamour names). The exact 2026-WC claim can't be backtested
(future event), but the STRUCTURAL premise can: across many RESOLVED binary markets, is
the market-implied probability mis-calibrated — do high-priced favourites realize BELOW
their price?

Method (read-only public API; no betting):
  - Gamma /markets closed=true, ordered by volume → the liquid universe (real price discovery).
    Keep binary YES/NO, cleanly resolved (outcomePrices ∈ {["1","0"],["0","1"]}), volume ≥ floor.
  - winner = index of "1" in outcomePrices; YES token = clobTokenIds[0].
  - pre-resolution price from CLOB prices-history (daily): point closest to endDate − {7d,30d}.
  - bin by implied price; per bin: n, mean implied, realized YES rate, diff, bootstrap 95% CI.
    Favourites overpriced ⇒ realized < implied (diff<0) in the high-price bins.

ROBUST: every priced market is appended to a JSONL cache immediately (kill-safe). Re-running
RESUMES (skips cached ids). A wall-clock --budget stops fetching gracefully; analysis always
runs on whatever is cached. `--analyze` re-analyses the cache without fetching.

Run:  PYTHONPATH=src python scripts/polymarket_flb_backtest.py --max 6000 --min-vol 50000 --budget 480
      PYTHONPATH=src python scripts/polymarket_flb_backtest.py --analyze        # anytime
"""
from __future__ import annotations
import argparse
import ast
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
LOOKBACKS = [7, 30]
CACHE = Path("data/scraped/polymarket_flb_cache.jsonl")


def unix(iso: str) -> int | None:
    try:
        return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def classify(q: str) -> str:
    s = q.lower()
    if any(k in s for k in ("election", "president", "senate", "governor", "nominee",
                            "democrat", "republican", "parliament", "prime minister")):
        return "politics"
    if any(k in s for k in ("bitcoin", "ethereum", " btc", " eth", "crypto", "solana", "dogecoin")):
        return "crypto"
    if any(k in s for k in (" vs ", " vs. ", " beat ", "premier league", "nba", "nfl", "mlb",
                            "nhl", "ufc", "champions league", "la liga", "serie a", "bundesliga",
                            "super bowl", "world cup", "ligue 1", "playoff", "to win the",
                            " cup", "f1 ", "grand prix", "tennis", "open final")):
        return "sports"
    return "other"


def fetch_market_list(max_n: int, min_vol: float, client: httpx.Client) -> list[dict]:
    """Gamma closed markets ordered by volume desc — metadata only (fast)."""
    out, offset = [], 0
    while len(out) < max_n:
        try:
            r = client.get(f"{GAMMA}/markets", params={
                "closed": "true", "limit": 500, "offset": offset,
                "order": "volumeNum", "ascending": "false"}, timeout=30)
        except httpx.HTTPError:
            break
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        last_vol = None
        for m in batch:
            try:
                op = ast.literal_eval(m.get("outcomePrices") or "[]")
                oc = ast.literal_eval(m.get("outcomes") or "[]")
            except Exception:
                continue
            last_vol = m.get("volumeNum") or 0
            if (len(op) == 2 and len(oc) == 2 and set(op) == {"1", "0"}
                    and last_vol >= min_vol and m.get("umaResolutionStatus") == "resolved"
                    and m.get("clobTokenIds") and m.get("endDate")):
                out.append(m)
        offset += 500
        if last_vol is not None and last_vol < min_vol:  # volume-desc → rest are below floor
            break
    return out[:max_n]


def price_series(token: str, client: httpx.Client):
    try:
        r = client.get(f"{CLOB}/prices-history",
                       params={"market": token, "interval": "max", "fidelity": 1440}, timeout=15)
        if r.status_code != 200:
            return []
        return [(p["t"], float(p["p"])) for p in r.json().get("history", [])]
    except httpx.HTTPError:
        return []


def price_at(series, target_ts, tol_days=4):
    best, bestd = None, tol_days * 86400
    for t, p in series:
        d = abs(t - target_ts)
        if d <= bestd:
            best, bestd = p, d
    return best


def load_cache() -> list[dict]:
    if not CACHE.exists():
        return []
    out = []
    for line in CACHE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def fetch_and_cache(max_n, min_vol, budget_s, client):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in load_cache()}
    print(f"fetching market list (vol≥${min_vol:,.0f}, up to {max_n}) ... [{len(done)} already cached]")
    markets = fetch_market_list(max_n, min_vol, client)
    print(f"  {len(markets)} binary resolved markets; pricing (budget {budget_s}s) ...")
    t0, n_new = time.time(), 0
    with open(CACHE, "a") as fh:
        for i, m in enumerate(markets):
            mid = m.get("id")
            if mid in done:
                continue
            if time.time() - t0 > budget_s:
                print(f"  ⏹ budget hit at {i}/{len(markets)} — stopping (cache safe, resumable)")
                break
            try:
                op = ast.literal_eval(m["outcomePrices"])
                won_yes = 1 if op[0] == "1" else 0
                yes_tok = ast.literal_eval(m["clobTokenIds"])[0]
                end = unix(m["endDate"])
            except Exception:
                continue
            if end is None:
                continue
            series = price_series(yes_tok, client)
            if not series:
                continue
            rec = {"id": mid, "cat": classify(m.get("question", "")), "won": won_yes,
                   "vol": m.get("volumeNum") or 0, "q": (m.get("question") or "")[:90]}
            for lb in LOOKBACKS:
                rec[f"p{lb}"] = price_at(series, end - lb * 86400)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done.add(mid)
            n_new += 1
            if n_new % 100 == 0:
                print(f"  ...priced {n_new} new ({i}/{len(markets)}, {time.time()-t0:.0f}s)")
            time.sleep(0.04)
    print(f"  +{n_new} new records; cache now {len(load_cache())} total")


BINS = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50),
        (0.50, 0.65), (0.65, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 0.985)]


def calib(records, nboot=2000):
    rng = np.random.RandomState(7)
    print(f"  {'price bin':<13}{'n':>5}{'implied':>9}{'realized':>10}{'diff':>8}{'95% CI diff':>20}")
    print("  " + "-" * 66)
    for lo, hi in BINS:
        rows = [(p, w) for p, w in records if lo <= p < hi]
        if len(rows) < 8:
            if rows:
                print(f"  {f'{lo:.2f}-{hi:.2f}':<13}{len(rows):>5}   (n<8, skip)")
            continue
        p = np.array([x[0] for x in rows]); w = np.array([x[1] for x in rows], float)
        implied, realized = p.mean(), w.mean()
        idx = np.arange(len(rows)); diffs = []
        for _ in range(nboot):
            b = rng.choice(idx, len(rows), replace=True)
            diffs.append(w[b].mean() - p[b].mean())
        clo, chi = np.percentile(diffs, [2.5, 97.5])
        sig = "  ←overpriced" if chi < 0 else ("  ←underpriced" if clo > 0 else "")
        print(f"  {f'{lo:.2f}-{hi:.2f}':<13}{len(rows):>5}{implied*100:>8.1f}%{realized*100:>9.1f}%"
              f"{(realized-implied)*100:>+7.1f}{f'[{clo*100:+.1f}, {chi*100:+.1f}]':>20}{sig}")


def analyze():
    rows = load_cache()
    print(f"analysing cache: {len(rows)} markets\n")
    if not rows:
        print("(empty cache — run a fetch first)")
        return
    by_cat = {}
    for r in rows:
        by_cat[r["cat"]] = by_cat.get(r["cat"], 0) + 1
    print("by category:", ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items(), key=lambda x: -x[1])))
    for lb in LOOKBACKS:
        recs = [(r[f"p{lb}"], r["won"]) for r in rows
                if r.get(f"p{lb}") is not None and 0.02 <= r[f"p{lb}"] <= 0.985]
        sports = [(r[f"p{lb}"], r["won"]) for r in rows
                  if r["cat"] == "sports" and r.get(f"p{lb}") is not None and 0.02 <= r[f"p{lb}"] <= 0.985]
        print(f"\n{'='*72}\nALL categories — price {lb}d before close vs realized   (n={len(recs)})\n{'='*72}")
        calib(recs)
        if len(sports) >= 30:
            print(f"\n  -- SPORTS subset (n={len(sports)}) --")
            calib(sports)
    print("\nReading: diff = realized − implied. Favourites OVERPRICED ⇒ diff<0 in HIGH bins")
    print("(0.80-0.985) with CI excluding 0 = the structural premise behind the fade.")
    print("Longshots (low bins) diff<0 = classic favorite-longshot bias (longshots overbet).")
    print("CAVEAT: volume≥floor → selection skews to big/liquid markets (where the fade would live).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=6000)
    ap.add_argument("--min-vol", type=float, default=50_000)
    ap.add_argument("--budget", type=float, default=480, help="wall-clock seconds for fetching")
    ap.add_argument("--analyze", action="store_true", help="analyse the cache only (no fetch)")
    args = ap.parse_args()
    if not args.analyze:
        client = httpx.Client(timeout=30)
        try:
            fetch_and_cache(args.max, args.min_vol, args.budget, client)
        finally:
            client.close()
    analyze()


if __name__ == "__main__":
    main()
