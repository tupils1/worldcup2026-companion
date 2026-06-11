"""Weather impact for WC 2026 fixtures.

Goes against the books' assumption of "neutral weather" by pulling forecast
for each match's venue at kickoff time. Adjusts expected goals:

    - Heavy rain (>3 mm/h):     λ × 0.88 (slick ball, sloppy passing)
    - Very heavy rain (>8 mm/h): λ × 0.78 (mistake-prone)
    - High wind (>40 km/h):     λ × 0.92 (long balls unreliable)
    - Extreme heat (>30°C):     λ × 0.95 (player fatigue, slower)
    - Very hot (>35°C):         λ × 0.90
    - Cold (<5°C):              λ × 0.97 (slight defensive bias)

Open-Meteo is free, no auth, no rate limit. Used for hourly forecast at each
WC venue around kickoff time.

Most WC books treat weather as a constant, so this gives marginal but real
edge especially in Mexico (heat) + Toronto/Vancouver (rain risk).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import httpx

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# WC 2026 venues with (lat, lon, name)
# 16 host cities (3 hosts: USA + CAN + MEX)
VENUES: dict[str, tuple[float, float, str]] = {
    "Mexico City":   (19.3030, -99.1505,  "Estadio Azteca"),
    "Guadalajara":   (20.6818, -103.4630, "Estadio Akron"),
    "Zapopan":       (20.6818, -103.4630, "Estadio Akron"),
    "Monterrey":     (25.6691, -100.2440, "Estadio BBVA"),
    "Guadalupe":     (25.6691, -100.2440, "Estadio BBVA"),  # Monterrey metro (Estadio BBVA is in Guadalupe, NL)
    "Toronto":       (43.6332, -79.4180,  "BMO Field"),
    "Vancouver":     (49.2767, -123.1118, "BC Place"),
    "Atlanta":       (33.7553, -84.4006,  "Mercedes-Benz Stadium"),
    "Boston":        (42.0909, -71.2643,  "Gillette Stadium"),
    "Foxborough":    (42.0909, -71.2643,  "Gillette Stadium"),
    "Dallas":        (32.7473, -97.0945,  "AT&T Stadium"),
    "Arlington":     (32.7473, -97.0945,  "AT&T Stadium"),
    "Houston":       (29.6847, -95.4107,  "NRG Stadium"),
    "Kansas City":   (39.0489, -94.4839,  "Arrowhead Stadium"),
    "Los Angeles":   (33.9535, -118.3392, "SoFi Stadium"),
    "Inglewood":     (33.9535, -118.3392, "SoFi Stadium"),
    "Miami":         (25.9580, -80.2389,  "Hard Rock Stadium"),
    "Miami Gardens": (25.9580, -80.2389,  "Hard Rock Stadium"),
    "New York":      (40.8135, -74.0743,  "MetLife Stadium"),
    "East Rutherford": (40.8135, -74.0743,"MetLife Stadium"),
    "Philadelphia":  (39.9008, -75.1675,  "Lincoln Financial Field"),
    "San Francisco": (37.4030, -121.9700, "Levi's Stadium"),
    "Santa Clara":   (37.4030, -121.9700, "Levi's Stadium"),
    "Seattle":       (47.5952, -122.3316, "Lumen Field"),
    "Cincinnati":    (39.0954, -84.5160,  "Paycor Stadium"),  # might not be host
    "Detroit":       (42.3400, -83.0456,  "Ford Field"),       # if used
}

# Climate-controlled venues: retractable/fixed roof that will close + air-condition in summer
# heat → indoor, weather (heat/rain/wind) NEUTRALISED. The rest are open-air.
# (Fixes the bug where e.g. Houston/Dallas got a heat haircut despite being roofed + AC.)
VENUE_ROOF: dict[str, bool] = {
    "Atlanta": True, "Dallas": True, "Arlington": True, "Houston": True,
    "Los Angeles": True, "Inglewood": True, "Vancouver": True,
    # open-air (heat/humidity real): Mexico City, Guadalajara/Zapopan, Monterrey/Guadalupe,
    # Toronto, Boston/Foxborough, Kansas City, Miami(canopy over seats, pitch open),
    # New York/East Rutherford, Philadelphia, San Francisco/Santa Clara, Seattle.
}


@dataclass
class WeatherSnapshot:
    venue_city: str
    kickoff_utc: str
    temp_c: float
    precip_mm_h: float
    wind_kmh: float
    humidity_pct: float
    cloud_cover_pct: float
    impact_multiplier: float  # < 1 means λ should be reduced
    notes: str


def compute_impact(temp_c: float, precip_mm_h: float, wind_kmh: float,
                   humidity_pct: float = 0.0, roofed: bool = False) -> tuple[float, str]:
    """Return (multiplier_on_lambda, note).

    - Climate-controlled (roofed) venues close + air-condition in summer → indoor →
      heat/rain/wind NEUTRALISED (factor 1.0). Fixes the roofed-venue heat-haircut bug.
    - Heat is HUMIDITY-AWARE: a humid 32°C taxes players far more than a dry one, so we
      apply the haircut on a 'feels-like' temp, not the dry bulb."""
    if roofed:
        return 1.0, f"顶棚恒温场馆(室内)→ 天气中和 (外 {temp_c:.0f}°C)"

    factor = 1.0
    notes = []

    # Rain
    if precip_mm_h > 8:
        factor *= 0.78
        notes.append(f"VERY HEAVY rain {precip_mm_h:.1f}mm/h → λ×0.78")
    elif precip_mm_h > 3:
        factor *= 0.88
        notes.append(f"heavy rain {precip_mm_h:.1f}mm/h → λ×0.88")
    elif precip_mm_h > 1:
        factor *= 0.95
        notes.append(f"light rain {precip_mm_h:.1f}mm/h → λ×0.95")

    # Wind
    if wind_kmh > 50:
        factor *= 0.88
        notes.append(f"VERY HIGH wind {wind_kmh:.0f}km/h → λ×0.88")
    elif wind_kmh > 40:
        factor *= 0.92
        notes.append(f"high wind {wind_kmh:.0f}km/h → λ×0.92")

    # Heat (player fatigue) — humidity-aware "feels-like" (humid air bites harder)
    humid_bonus = max(0.0, (humidity_pct - 55) / 8.0) if (temp_c > 27 and humidity_pct) else 0.0
    feels = temp_c + humid_bonus
    hb = f"/湿{humidity_pct:.0f}%(体感{feels:.0f})" if humid_bonus > 0.5 else ""
    if feels > 35:
        factor *= 0.90
        notes.append(f"酷热 {temp_c:.0f}°C{hb} → λ×0.90")
    elif feels > 30:
        factor *= 0.95
        notes.append(f"炎热 {temp_c:.0f}°C{hb} → λ×0.95")

    # Cold (mild defensive bias)
    if temp_c < 5:
        factor *= 0.97
        notes.append(f"cold {temp_c:.0f}°C → λ×0.97")

    if not notes:
        notes.append(f"normal {temp_c:.0f}°C, {precip_mm_h:.1f}mm/h rain, {wind_kmh:.0f}km/h wind")

    return factor, "; ".join(notes)


def fetch_weather(city: str, kickoff_iso: str) -> WeatherSnapshot | None:
    """Pull Open-Meteo hourly forecast for a venue + kickoff hour."""
    if city not in VENUES:
        # Try fuzzy match
        for v_city in VENUES:
            if city.lower() in v_city.lower() or v_city.lower() in city.lower():
                city = v_city
                break
        else:
            return None
    lat, lon, _ = VENUES[city]

    # Parse kickoff (handle date-only or full ISO)
    try:
        if "T" in kickoff_iso:
            ko = dt.datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
        else:
            # Date only: assume kickoff at 18:00 UTC
            ko = dt.datetime.fromisoformat(kickoff_iso + "T18:00:00+00:00")
    except Exception:
        return None

    date = ko.date().isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m,cloud_cover",
        "start_date": date,
        "end_date": date,
        "timezone": "UTC",
    }
    r = None
    for attempt in range(3):  # transient SSL/connection errors are common
        try:
            r = httpx.get(OPEN_METEO_FORECAST, params=params, timeout=15)
            break
        except Exception:
            if attempt == 2:
                return None
            import time as _t
            _t.sleep(1.0)
    if r is None or r.status_code != 200:
        return None
    data = r.json().get("hourly", {})
    if not data:
        return None

    # Find the kickoff hour
    times = data.get("time", [])
    kickoff_hr = ko.strftime("%Y-%m-%dT%H:00")
    idx = None
    for i, t in enumerate(times):
        if t.startswith(kickoff_hr[:13]):
            idx = i
            break
    if idx is None:
        idx = min(range(len(times)), key=lambda i: abs(
            (dt.datetime.fromisoformat(times[i]) - ko.replace(tzinfo=None)).total_seconds()
        )) if times else None
    if idx is None:
        return None

    temp = data["temperature_2m"][idx]
    precip = data["precipitation"][idx]
    wind = data["wind_speed_10m"][idx]
    humidity = data.get("relative_humidity_2m", [None])[idx] if data.get("relative_humidity_2m") else 0
    cloud = data.get("cloud_cover", [None])[idx] if data.get("cloud_cover") else 0

    factor, notes = compute_impact(temp, precip, wind, humidity_pct=float(humidity or 0),
                                   roofed=VENUE_ROOF.get(city, False))
    return WeatherSnapshot(
        venue_city=city,
        kickoff_utc=ko.isoformat(),
        temp_c=float(temp),
        precip_mm_h=float(precip),
        wind_kmh=float(wind),
        humidity_pct=float(humidity or 0),
        cloud_cover_pct=float(cloud or 0),
        impact_multiplier=factor,
        notes=notes,
    )


def ingest_wc_weather(db_path: Path | str = DEFAULT_DB_PATH, days_ahead: int = 16) -> dict:
    """Pull weather for all WC 2026 fixtures starting within `days_ahead` days."""
    conn = get_conn(db_path)
    try:
        # Ensure storage column
        try:
            conn.execute("ALTER TABLE matches ADD COLUMN weather_factor REAL")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE matches ADD COLUMN weather_notes TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()

        cutoff = (dt.date.today() + dt.timedelta(days=days_ahead)).isoformat()
        fixtures = list(conn.execute(
            "SELECT id, home_code, away_code, match_date, venue FROM matches "
            "WHERE finished=0 AND match_date <= ? AND match_date >= ? AND venue IS NOT NULL",
            (cutoff, dt.date.today().isoformat()),
        ))
        if not fixtures:
            return {"fixtures_in_window": 0, "weather_fetched": 0}

        fetched = 0
        adjusted = []
        for fx in fixtures:
            snap = fetch_weather(fx["venue"], fx["match_date"])
            if snap is None:
                continue
            conn.execute(
                "UPDATE matches SET weather_factor=?, weather_notes=? WHERE id=?",
                (snap.impact_multiplier, snap.notes, fx["id"]),
            )
            adjusted.append({
                "match": f"{fx['home_code']}-{fx['away_code']}",
                "date": fx["match_date"],
                "venue": fx["venue"],
                "temp": snap.temp_c,
                "precip": snap.precip_mm_h,
                "wind": snap.wind_kmh,
                "factor": snap.impact_multiplier,
                "notes": snap.notes,
            })
            fetched += 1
        conn.commit()
        return {"fixtures_in_window": len(fixtures), "weather_fetched": fetched,
                "adjusted": adjusted}
    finally:
        conn.close()


def main():
    import json
    print("=== WC Weather Impact ===")
    result = ingest_wc_weather(days_ahead=21)
    print(f"\nFixtures in window: {result['fixtures_in_window']}")
    print(f"Weather fetched:    {result['weather_fetched']}")
    if not result.get('adjusted'):
        print("\nNo fixtures in window with venue info. (WC starts 2026-06-11.)")
        return

    print(f"\n{'Match':<13} {'Date':<10} {'Venue':<22} {'Temp':>6} {'Rain':>6} {'Wind':>6} {'Factor':>7}  Notes")
    print("-" * 130)
    notable = [a for a in result['adjusted'] if a['factor'] < 0.98]
    for a in notable[:20]:
        venue_disp = a['venue'][:20]
        print(f"{a['match']:<13} {a['date']:<10} {venue_disp:<22} "
              f"{a['temp']:5.1f}°C {a['precip']:5.1f} {a['wind']:5.1f}  "
              f"{a['factor']:7.3f}  {a['notes']}")
    print(f"\n{len(notable)} fixtures with material weather impact (factor < 0.98)")
    print("These have λ adjusted DOWNWARD in your model → over-bets become under-bets")


if __name__ == "__main__":
    main()
