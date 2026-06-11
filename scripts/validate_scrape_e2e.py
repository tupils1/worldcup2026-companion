"""End-to-end validation: real Chrome-MCP-scraped 任九 odds → our model → value.

Data scraped 2026-05-29 from 500.com 任选九 round 26083 (trade.500.com/rj/) via the
user's browser (Chrome MCP) — the live 欧赔 that WebFetch could only see as '- - -'.
选择比例 (本站用户投注比例) was empty (round just opened; it populates near the 05-31
deadline), so we validate the odds path here and use de-vigged odds as the crowd proxy.
"""
from __future__ import annotations
import json
from pathlib import Path

from worldcup.models.dixon_coles import fit
from worldcup.models.markets import score_matrix, prob_1x2
from worldcup.strategy.value_bets import devig_shin

# Scraped round 26083 — national-team friendlies (matches 1-10; 11-14 were Swedish/
# Finnish club games not in our international model). (home, away, [home,draw,away] 欧赔).
SCRAPED = [
    ("SUI", "JOR", [1.25, 5.51, 10.18]),
    ("GER", "FIN", [1.14, 8.22, 14.11]),
    ("USA", "SEN", [2.32, 3.23, 2.93]),
    ("BRA", "PAN", [1.11, 8.42, 17.95]),
    ("BUL", "MNE", [2.62, 3.12, 2.61]),
    ("NOR", "SWE", [1.86, 3.55, 3.82]),
    ("TUR", "MKD", [1.27, 5.22, 9.40]),
    ("AUT", "TUN", [1.54, 3.85, 5.72]),
    ("COL", "CRC", [1.26, 5.26, 10.60]),
    ("CAN", "UZB", [1.44, 3.99, 6.96]),
]

def main():
    out_dir = Path("data/scraped"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "renjiu_26083.json").write_text(
        json.dumps({"issue": "26083", "source": "500.com/rj", "scraped": "2026-05-29",
                    "selection_pct": None, "matches": SCRAPED}, ensure_ascii=False, indent=2))
    print("saved data/scraped/renjiu_26083.json\n")

    print("Fitting Dixon-Coles ...")
    p = fit(elo_prior_strength=0.5)
    print(f"  {p.n_matches} matches\n")
    print("REAL scraped 欧赔 (de-vigged) vs OUR model — friendlies, home venue assumed:")
    print(f"{'match':<10}{'mkt H/D/A (de-vig)':<26}{'model H/D/A':<26}{'|Δ|max':>7}")
    print("-" * 72)
    covered = 0
    for h, a, odds in SCRAPED:
        if h not in p.attack or a not in p.attack:
            print(f"{h}-{a:<6} (one team not in model fit — skipped)")
            continue
        covered += 1
        mh, md, ma = devig_shin(odds)
        lh, la = p.predict_lambda(h, a, neutral=False)
        Mh, Md, Ma = prob_1x2(score_matrix(lh, la, rho=p.rho))
        dmax = max(abs(mh-Mh), abs(md-Md), abs(ma-Ma)) * 100
        flag = "  ← big gap" if dmax >= 10 else ""
        print(f"{h}-{a:<6}  {mh*100:4.0f}/{md*100:4.0f}/{ma*100:4.0f}%"
              f"           {Mh*100:4.0f}/{Md*100:4.0f}/{Ma*100:4.0f}%        {dmax:5.1f}{flag}")
    print(f"\n  {covered}/10 matches model-covered. Δ = model−market (per-leg crowd proxy = de-vig odds).")
    print("  Validates: Chrome-MCP scrape → de-vig → model comparison runs on REAL data.")
    print("  Caveat: friendlies + raw model carry ENG-under/MAR-over bias — anchor to market for sizing.")

if __name__ == "__main__":
    main()
