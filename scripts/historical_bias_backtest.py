"""Rigorous multi-league market-bias backtest — TRUE per-match closing lines.

v3: top-5 leagues, 2025-26, true closing lines (snapshot ~10min pre-kickoff). Samples the
DENSEST kickoff slots (most simultaneous matches) to maximise matches/credit. Focus: the
DRAW under-pricing lean (EPL showed +3pp) — pooled across leagues + significance (SE, z).
Also reports favorite-longshot calibration + closing-line Brier on the pooled sample.

Quota: ~--snaps-per-league snapshots/league × 10 credits.
Run:  PYTHONPATH=src python scripts/historical_bias_backtest.py --snaps-per-league 45
"""
from __future__ import annotations
import argparse, datetime as dt, sys, time
from collections import defaultdict
import httpx
import numpy as np
sys.path.insert(0, "src")
from worldcup.strategy.value_bets import devig_shin

def secret(k):
    for ln in open("configs/secrets.env"):
        if ln.startswith(k + "="): return ln.split("=", 1)[1].split("#")[0].strip()
OA, AF = secret("ODDS_API_KEY"), secret("API_FOOTBALL_KEY")
def norm(s): return "".join(c for c in s.lower() if c.isalnum())

# (Odds API sport key, API-Football league id, name)
LEAGUES = [("soccer_epl", 39, "EPL"), ("soccer_spain_la_liga", 140, "LaLiga"),
           ("soccer_germany_bundesliga", 78, "Bundes"), ("soccer_italy_serie_a", 135, "SerieA"),
           ("soccer_france_ligue_one", 61, "Ligue1")]

def fetch_fixtures(af_league):
    r = httpx.get("https://v3.football.api-sports.io/fixtures",
                  params={"league": af_league, "season": 2025}, headers={"x-apisports-key": AF}, timeout=30).json()
    out = []
    for f in r.get("response", []):
        g = f.get("goals", {})
        if g.get("home") is None: continue
        ko = dt.datetime.fromisoformat(f["fixture"]["date"].replace("Z", "+00:00"))
        out.append({"h": f["teams"]["home"]["name"], "a": f["teams"]["away"]["name"],
                    "hs": g["home"], "as": g["away"], "ko": ko})
    return out

def pull_closing(sport_key, ko_dt):
    snap = (ko_dt - dt.timedelta(minutes=10)).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = httpx.get(f"https://api.the-odds-api.com/v4/historical/sports/{sport_key}/odds",
                      params={"apiKey": OA, "regions": "eu,uk", "markets": "h2h",
                              "date": snap, "oddsFormat": "decimal"}, timeout=30).json()
    except Exception:
        return {}
    out = {}
    for e in r.get("data", []):
        ct = dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        if abs((ct - ko_dt).total_seconds()) > 1800: continue
        per = defaultdict(list)
        for bk in e.get("bookmakers", []):
            o = {}
            for m in bk.get("markets", []):
                if m["key"] == "h2h":
                    for oc in m["outcomes"]:
                        if oc["name"] == e["home_team"]: o["h"] = oc["price"]
                        elif oc["name"] == e["away_team"]: o["a"] = oc["price"]
                        elif oc["name"] == "Draw": o["d"] = oc["price"]
            if all(k in o for k in ("h", "d", "a")):
                for k, p in zip(("h", "d", "a"), devig_shin([o["h"], o["d"], o["a"]])): per[k].append(p)
        if per.get("h"):
            out[(norm(e["home_team"]), norm(e["away_team"]))] = \
                (float(np.mean(per["h"])), float(np.mean(per["d"])), float(np.mean(per["a"])))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps-per-league", type=int, default=45)
    args = ap.parse_args()
    pooled = []          # (ph,pd,pa,outcome)
    per_league = {}
    for sport, afid, name in LEAGUES:
        print(f"[{name}] fixtures ...", end=" ", flush=True)
        fx = fetch_fixtures(afid)
        fxmap = {(norm(f["h"]), norm(f["a"])): f for f in fx}
        # EVEN-STRIDE across the season (UNBIASED — densest-slot sampling over-weights
        # congested/dead-rubber rounds which are draw-prone + favorite-rotation-prone).
        by_slot = defaultdict(int)
        for f in fx: by_slot[f["ko"]] += 1
        allslots = sorted(by_slot)  # chronological
        if len(allslots) > args.snaps_per_league:
            stride = len(allslots) / args.snaps_per_league
            slots = [allslots[int(i * stride)] for i in range(args.snaps_per_league)]
        else:
            slots = allslots
        rows = []
        for ko in slots:
            for key, (ph, pd, pa) in pull_closing(sport, ko).items():
                f = fxmap.get(key)
                if not f: continue
                o = 0 if f["hs"] > f["as"] else (1 if f["hs"] == f["as"] else 2)
                rows.append((ph, pd, pa, o))
            time.sleep(0.15)
        per_league[name] = rows; pooled += rows
        di = np.mean([pd for _, pd, _, _ in rows]) if rows else 0
        dr = np.mean([1 if o == 1 else 0 for *_, o in rows]) if rows else 0
        print(f"{len(fx)} fx → {len(rows)} matches | draw implied {di*100:.1f}% realized {dr*100:.1f}%")
    n = len(pooled)
    print(f"\n=== POOLED {n} matches (5 leagues, true closing) ===")
    # DRAW test with significance
    di = np.mean([pd for _, pd, _, _ in pooled])
    draws = [1 if o == 1 else 0 for *_, o in pooled]
    dr = np.mean(draws); se = (dr * (1 - dr) / n) ** 0.5
    z = (dr - di) / se if se else 0
    print(f"① DRAW: implied {di*100:.2f}%  realized {dr*100:.2f}%  diff {(dr-di)*100:+.2f}pp")
    print(f"   SE {se*100:.2f}pp  z = {z:.2f}  → {'SIGNIFICANT (|z|>2)' if abs(z)>2 else 'borderline' if abs(z)>1.3 else 'not significant'}")
    print("   per-league diff:", "  ".join(
        f"{nm} {(np.mean([1 if o==1 else 0 for *_,o in r])-np.mean([pd for _,pd,_,_ in r]))*100:+.1f}"
        for nm, r in per_league.items() if r))
    # favorite-longshot
    print("\n② FAVORITE-LONGSHOT (pooled):")
    bins = [(0.0,0.25),(0.25,0.40),(0.40,0.55),(0.55,0.70),(0.70,1.01)]
    agg = {b: [] for b in bins}
    for ph,pd,pa,o in pooled:
        for prob,win in ((ph,o==0),(pd,o==1),(pa,o==2)):
            for b in bins:
                if b[0] <= prob < b[1]: agg[b].append((prob,win)); break
    for b in bins:
        v = agg[b]
        if v: print(f"   {f'{b[0]:.2f}-{b[1]:.2f}':<12} n={len(v):>4}  implied {np.mean([x[0] for x in v])*100:5.1f}%  realized {np.mean([x[1] for x in v])*100:5.1f}%")
    briers = [sum((p-t)**2 for p,t in zip((ph,pd,pa),[1 if o==i else 0 for i in range(3)])) for ph,pd,pa,o in pooled]
    print(f"\n③ CLOSING-LINE Brier {np.mean(briers):.4f}")

if __name__ == "__main__":
    main()
