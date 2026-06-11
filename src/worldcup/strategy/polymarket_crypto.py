"""Polymarket crypto-price edge scanner — sharp benchmark = Deribit options-implied prob.

Generalises the football Polymarket scanner to crypto: the "truth" for
"Will BTC be ABOVE $K on date T?" is the RISK-NEUTRAL probability P(S_T > K) implied
by Deribit's option smile (free public API, institutional-grade). Bet the divergence
on Polymarket.

Method (standard option-implied digital): under the forward measure, P(S_T > K) = N(d2)
with d2 = [ln(F/K) − ½σ²T] / (σ√T), where F = Deribit forward for that expiry and σ =
the smile IV interpolated at strike K. (F already embeds carry → no separate rate term.)

ONLY European "above $K on/at date T" markets map cleanly. Barrier/"hit $K by T" (touch,
= one-touch option) and ultra-short (<1 day) markets are DETECTED and SKIPPED — they need
different math, not the European digital.

Run:
    PYTHONPATH=src python -m worldcup.strategy.polymarket_crypto --validate   # benchmark sanity
    PYTHONPATH=src python -m worldcup.strategy.polymarket_crypto               # scan vs Polymarket
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re

import httpx
from scipy.stats import norm

from worldcup.strategy.value_bets import kelly_fraction

CUR_ALIAS = {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH"}
_MONTHS = ("january february march april may june july august september "
           "october november december").split()

DERIBIT = "https://www.deribit.com/api/v2/public"
GAMMA = "https://gamma-api.polymarket.com"
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, **params):
    """GET with retry (this host sees transient SSL EOF)."""
    last = None
    for attempt in range(4):
        try:
            return httpx.get(url, params=params, headers=UA, timeout=25).json()
        except Exception as e:  # noqa
            last = e
            import time
            time.sleep(1.0)
    raise last


# ─────────────────────────── Deribit benchmark ───────────────────────────────
def _parse_instrument(name: str):
    """'BTC-25DEC26-76000-C' → (expiry_date, strike, 'C'|'P')."""
    parts = name.split("-")
    if len(parts) != 4:
        return None
    _cur, exp, strike, cp = parts
    try:
        d = dt.datetime.strptime(exp, "%d%b%y").date()
        return d, float(strike), cp
    except ValueError:
        return None


def fetch_chain(currency: str = "BTC") -> tuple[dict, float]:
    """Return ({expiry_date: {strike: iv_pct}}, spot/index). Uses mark_iv from the
    book summary, averaging call/put IV at the same (expiry, strike)."""
    data = _get(f"{DERIBIT}/get_book_summary_by_currency", currency=currency, kind="option")
    rows = data.get("result", [])
    smile: dict[dt.date, dict[float, list[float]]] = {}
    fwd: dict[dt.date, float] = {}
    for r in rows:
        p = _parse_instrument(r.get("instrument_name", ""))
        iv = r.get("mark_iv")
        if not p or iv is None or iv <= 0:
            continue
        exp, strike, _cp = p
        smile.setdefault(exp, {}).setdefault(strike, []).append(float(iv))
        if r.get("underlying_price"):
            fwd[exp] = float(r["underlying_price"])
    iv_by_exp = {e: {k: sum(v) / len(v) for k, v in strikes.items()}
                 for e, strikes in smile.items()}
    idx = _get(f"{DERIBIT}/get_index_price", index_name=f"{currency.lower()}_usd")
    spot = float(idx.get("result", {}).get("index_price", 0.0))
    return {"iv": iv_by_exp, "fwd": fwd, "spot": spot}, spot


def _interp_iv(strikes_iv: dict[float, float], K: float) -> float | None:
    """Linear-in-strike IV interpolation (clamped to nearest at the wings)."""
    ks = sorted(strikes_iv)
    if not ks:
        return None
    if K <= ks[0]:
        return strikes_iv[ks[0]]
    if K >= ks[-1]:
        return strikes_iv[ks[-1]]
    for i in range(len(ks) - 1):
        if ks[i] <= K <= ks[i + 1]:
            lo, hi = ks[i], ks[i + 1]
            w = (K - lo) / (hi - lo)
            return strikes_iv[lo] * (1 - w) + strikes_iv[hi] * w
    return strikes_iv[ks[-1]]


def implied_prob_above(chain: dict, K: float, expiry: dt.date,
                       now: dt.date | None = None) -> dict | None:
    """Risk-neutral P(S_T > K) from the Deribit smile. Picks the listed expiry nearest
    to `expiry`. Returns dict with prob + the inputs used (so callers can sanity-check)."""
    now = now or dt.date.today()
    expiries = sorted(chain["iv"])
    if not expiries:
        return None
    exp = min(expiries, key=lambda e: abs((e - expiry).days))
    T = max((exp - now).days, 0) / 365.0
    if T <= 0:
        return None
    F = chain["fwd"].get(exp) or chain["spot"]
    sigma = _interp_iv(chain["iv"][exp], K)
    if not sigma or sigma <= 0:
        return None
    sigma /= 100.0  # mark_iv is in percent
    d2 = (math.log(F / K) - 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    return {"prob": float(norm.cdf(d2)), "expiry_used": exp.isoformat(),
            "T_years": T, "forward": F, "iv_at_K": sigma * 100,
            "expiry_gap_days": abs((exp - expiry).days)}


# ─────────────────────────── validation ──────────────────────────────────────
def validate(currency: str = "BTC") -> None:
    chain, spot = fetch_chain(currency)
    print(f"=== Deribit benchmark sanity — {currency} spot ${spot:,.0f} ===")
    expiries = sorted(chain["iv"])
    print(f"listed expiries: {len(expiries)} "
          f"({expiries[0].isoformat()} … {expiries[-1].isoformat()})")
    # pick an expiry ~30d out
    target = dt.date.today() + dt.timedelta(days=30)
    exp = min(expiries, key=lambda e: abs((e - target).days))
    print(f"\nP({currency} > K at {exp.isoformat()}) across a strike ladder "
          f"(must be monotone ↓, ATM≈0.5):")
    print(f"  {'strike':>10}{'moneyness':>11}{'IV%':>7}{'P(>K)':>9}")
    for mult in (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25, 1.50):
        K = round(spot * mult, -2)
        r = implied_prob_above(chain, K, exp)
        if r:
            print(f"  {K:>10,.0f}{mult:>10.0%}{r['iv_at_K']:>7.1f}{r['prob']*100:>8.1f}%")
    print("\n✓ if P(>K) decreases monotonically and ATM (1.00×) ≈ 45-55%, the engine is sound.")


# ─────────────────────────── Polymarket matching ─────────────────────────────
def discover_crypto_markets() -> list[dict]:
    """Find Polymarket crypto markets via the crypto/bitcoin/ethereum event tags."""
    out, seen = [], set()
    for slug in ("bitcoin", "ethereum", "crypto"):
        try:
            evs = _get(f"{GAMMA}/events", closed="false", tag_slug=slug, limit=40)
        except Exception:
            continue
        for e in evs:
            for m in e.get("markets", []) or []:
                mid = m.get("id")
                if mid and mid not in seen and m.get("active") and not m.get("closed"):
                    seen.add(mid)
                    m["_event_title"] = e.get("title")
                    out.append(m)
    return out


def classify_market(q: str) -> str:
    """European 'above $K on date' (mappable) vs barrier 'hit by' vs other."""
    ql = q.lower()
    if any(k in ql for k in (" hit ", "reach", "all time high", "ath", "before gta")):
        return "barrier"      # touch / first-passage — NOT a European digital
    if "up or down" in ql or "5m" in ql:
        return "ultrashort"
    if any(k in ql for k in ("above", "below", "≥", ">", "<", "be over", "be under")):
        return "european"
    return "other"


def parse_european(q: str, now: dt.date | None = None):
    """'Bitcoin above 95,000 on June 5, 7AM ET?' → ('BTC', 95000.0, date(2026,6,5)).
    Year is inferred (roll to next year if the month/day already passed)."""
    now = now or dt.date.today()
    m = re.search(r"(bitcoin|ethereum|btc|eth)\s+above\s+([\d,]+)\s+on\s+"
                  r"([a-z]+)\s+(\d{1,2})", q, re.IGNORECASE)
    if not m:
        return None
    cur = CUR_ALIAS.get(m.group(1).lower())
    strike = float(m.group(2).replace(",", ""))
    mon = m.group(3).lower()
    if mon not in _MONTHS:
        return None
    month, day = _MONTHS.index(mon) + 1, int(m.group(4))
    try:
        d = dt.date(now.year, month, day)
    except ValueError:
        return None
    if d < now - dt.timedelta(days=2):
        d = dt.date(now.year + 1, month, day)
    return cur, strike, d


def scan(min_edge: float = 0.03, cost_pp: float = 1.0, bankroll: float = 1000.0,
         kelly_scaling: float = 0.25, max_stake_frac: float = 0.03,
         max_expiry_gap_days: int = 3) -> list[dict]:
    """Edge = Deribit-implied P(S_T>K) − Polymarket YES price, on future European
    'above $K on date' markets. Skip if Deribit's nearest expiry is >gap days off."""
    mkts = [m for m in discover_crypto_markets()
            if classify_market(m.get("question", "")) == "european"]
    chains: dict[str, dict] = {}
    bets = []
    for m in mkts:
        parsed = parse_european(m.get("question", ""))
        px = m.get("outcomePrices")
        if not parsed or not px:
            continue
        cur, K, exp = parsed
        if exp < dt.date.today():
            continue
        try:
            yes_price = float(eval(px)[0]) if isinstance(px, str) else float(px[0])
        except Exception:
            continue
        # Skip settled / near-certain markets: price ≤2% or ≥98% = resolved-but-not-delisted
        # (these are where stale past-dated dailies sit) or no tradeable retail edge anyway.
        if not (0.02 < yes_price < 0.98):
            continue
        if cur not in chains:
            try:
                chains[cur], _ = fetch_chain(cur)
            except Exception:
                chains[cur] = None
        if not chains.get(cur):
            continue
        bench = implied_prob_above(chains[cur], K, exp)
        if not bench or bench["expiry_gap_days"] > max_expiry_gap_days:
            continue
        truth = bench["prob"]
        raw = abs(truth - yes_price)
        net = raw - cost_pp / 100.0
        if net < min_edge:
            continue
        if truth > yes_price:
            side, price, p_side = "BUY YES", yes_price, truth
        else:
            side, price, p_side = "BUY NO", 1.0 - yes_price, 1.0 - truth
        if not (0.0 < price < 1.0):
            continue
        dec = 1.0 / price
        kf = kelly_fraction(p_side, dec, scaling=kelly_scaling)
        bets.append({
            "q": m.get("question", "")[:50], "cur": cur, "strike": K,
            "expiry": exp.isoformat(), "side": side, "price": price,
            "poly_yes": yes_price, "deribit": truth, "iv": bench["iv_at_K"],
            "gap_days": bench["expiry_gap_days"], "net_edge_pp": net * 100,
            "ev_pct": (p_side * dec - 1) * 100, "stake": min(kf * bankroll, max_stake_frac * bankroll),
        })
    bets.sort(key=lambda x: -x["ev_pct"])
    return bets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true", help="Benchmark sanity check only")
    ap.add_argument("--triage", action="store_true", help="Show market mappability buckets")
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    args = ap.parse_args()

    if args.validate:
        validate(args.currency)
        return
    if args.triage:
        mkts = discover_crypto_markets()
        buckets: dict[str, list] = {"european": [], "barrier": [], "ultrashort": [], "other": []}
        for m in mkts:
            buckets[classify_market(m.get("question", ""))].append(m)
        for kind, ms in buckets.items():
            tag = "← options-mappable" if kind == "european" else "← skip"
            print(f"\n{kind.upper()} ({len(ms)}) {tag}")
            for m in ms[:6]:
                print(f"   {(m.get('question') or '')[:66]:<66} px={m.get('outcomePrices')}")
        return

    print("=" * 96)
    print("POLYMARKET CRYPTO EDGE — truth = Deribit options-implied P(S>K); bet on Polymarket")
    print("=" * 96)
    bets = scan(min_edge=args.min_edge, bankroll=args.bankroll)
    if not bets:
        print(f"\n  No edge ≥ {args.min_edge*100:.0f}pp on future European markets.")
        print("  Expected: short-dated crypto digitals are HEAVILY arbed vs Deribit/Binance by")
        print("  bots — Polymarket ≈ options here. Edge only on a fast move retail hasn't caught.")
        return
    print(f"\n{'market':<52}{'side':<8}{'@':>6}{'poly':>6}{'deribit':>8}{'gap':>4}{'netEdge':>8}{'EV':>7}{'stake$':>7}")
    print("-" * 96)
    for b in bets[:20]:
        print(f"{b['q']:<52}{b['side']:<8}{b['price']:>6.2f}{b['poly_yes']*100:>5.0f}%"
              f"{b['deribit']*100:>7.0f}%{b['gap_days']:>4}{b['net_edge_pp']:>+7.1f}%"
              f"{b['ev_pct']:>+6.1f}%{b['stake']:>7.0f}")
    print("\nBUY YES = Polymarket cheaper than Deribit-implied; BUY NO = pricier. gap = days between")
    print("Polymarket date and nearest Deribit expiry (>3 skipped). ¼-Kelly, capped 3% bankroll.")


if __name__ == "__main__":
    main()
