"""API-Football player data ingester for the golden boot model.

Strategy:
    Don't pull every player individually (1200+ requests).
    Instead pull /players/topscorers across competitions where WC-eligible
    players actually score. This gives ~200-400 unique scorers with goals +
    minutes, which is enough to model the golden boot race.

Competitions we sweep:
    - WC 2022 (league 1, season 2022) — historical baseline
    - WC 2026 qualifications: CAF 29, CONCACAF 31, UEFA 32, OFC 33, CONMEBOL 34, IC 37
    - UEFA Nations League 2024-25 (league 5)
    - Euro 2024 (league 4)
    - Copa América 2024 (league 9)
    - AFCON 2023 (league 6)
    - AFC Asian Cup 2023 (league 7)
    - FIFA Club World Cup 2025 (league 15) — for elite club form
    - Top European leagues' top scorers (EPL 39, LaLiga 140, Bundesliga 78,
      Serie A 135, Ligue 1 61) — for club-form per-90 baselines

Each call returns up to 20 scorers per competition. Total ≈ 17 requests for
the full sweep. Quota burn negligible (Ultra plan = 75k/day).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldcup.db.schema import DEFAULT_DB_PATH, get_conn
from worldcup.ingest.api_football import APIFootballClient

# (league_id, season, label, country-filter-or-None)
# label is for reporting; country-filter restricts to relevant WC teams.
COMPETITIONS_TO_SWEEP: list[tuple[int, int, str]] = [
    (1, 2022, "WC 2022"),
    (29, 2026, "WC qualif CAF 2026"),
    (31, 2026, "WC qualif CONCACAF 2026"),
    (32, 2024, "WC qualif UEFA 2024"),
    (33, 2026, "WC qualif OFC 2026"),
    (34, 2026, "WC qualif CONMEBOL 2026"),
    (37, 2026, "WC qualif Intercontinental 2026"),
    (5, 2024, "UEFA Nations League 2024-25"),
    (4, 2024, "UEFA Euro 2024"),
    (9, 2024, "Copa América 2024"),
    (6, 2023, "AFCON 2023"),
    (7, 2023, "AFC Asian Cup 2023"),
    (15, 2025, "FIFA Club World Cup 2025"),
    # Top European club leagues — current season for per-90 form
    (39, 2024, "EPL 2024-25"),
    (140, 2024, "LaLiga 2024-25"),
    (78, 2024, "Bundesliga 2024-25"),
    (135, 2024, "Serie A 2024-25"),
    (61, 2024, "Ligue 1 2024-25"),
]


@dataclass
class PlayerStatLine:
    af_player_id: int
    name: str
    nationality: str | None
    af_team_id: int
    team_name: str
    competition_label: str
    league_id: int
    season: int
    games_played: int
    minutes: int
    goals: int
    assists: int
    shots: int | None
    shots_on_target: int | None
    rating: float | None

    @property
    def goals_per_90(self) -> float:
        return self.goals * 90.0 / self.minutes if self.minutes > 0 else 0.0

    @property
    def minutes_per_game(self) -> float:
        return self.minutes / self.games_played if self.games_played > 0 else 0.0


def _safe_int(v: Any) -> int:
    if v is None:
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch_topscorers(
    league: int,
    season: int,
    label: str,
    client: APIFootballClient,
) -> list[PlayerStatLine]:
    """Pull top scorers for one competition. Returns list of PlayerStatLine."""
    data = client.get("/players/topscorers", league=league, season=season)
    out: list[PlayerStatLine] = []
    for entry in data.get("response", []):
        player = entry["player"]
        stats_blocks = entry.get("statistics", [])
        # Combine across blocks (rare: same player in multiple teams)
        total_games = total_minutes = total_goals = total_assists = 0
        total_shots = total_shots_on = 0
        team_id = 0
        team_name = ""
        rating_sum = 0.0
        rating_n = 0
        for s in stats_blocks:
            g = s.get("games", {})
            goals = s.get("goals", {})
            shots = s.get("shots", {})
            total_games += _safe_int(g.get("appearences"))
            total_minutes += _safe_int(g.get("minutes"))
            total_goals += _safe_int(goals.get("total"))
            total_assists += _safe_int(goals.get("assists"))
            total_shots += _safe_int(shots.get("total"))
            total_shots_on += _safe_int(shots.get("on"))
            if not team_id and s.get("team"):
                team_id = s["team"]["id"]
                team_name = s["team"]["name"]
            r = _safe_float(g.get("rating"))
            if r is not None:
                rating_sum += r
                rating_n += 1
        rating = rating_sum / rating_n if rating_n else None
        out.append(
            PlayerStatLine(
                af_player_id=player["id"],
                name=player["name"],
                nationality=player.get("nationality"),
                af_team_id=team_id,
                team_name=team_name,
                competition_label=label,
                league_id=league,
                season=season,
                games_played=total_games,
                minutes=total_minutes,
                goals=total_goals,
                assists=total_assists,
                shots=total_shots or None,
                shots_on_target=total_shots_on or None,
                rating=rating,
            )
        )
    return out


def sweep_all_competitions(
    client: APIFootballClient | None = None,
    competitions: list[tuple[int, int, str]] | None = None,
) -> list[PlayerStatLine]:
    """Pull top scorers from every configured competition."""
    own_client = client is None
    client = client or APIFootballClient()
    comps = competitions or COMPETITIONS_TO_SWEEP
    all_stats: list[PlayerStatLine] = []
    try:
        for league, season, label in comps:
            try:
                rows = fetch_topscorers(league, season, label, client)
                all_stats.extend(rows)
                print(f"  ✓ {label:<35} {len(rows):>3} scorers")
            except Exception as exc:
                print(f"  ✗ {label:<35} {type(exc).__name__}: {str(exc)[:100]}")
    finally:
        if own_client:
            client.close()
    return all_stats


def persist_to_player_table(
    stats: list[PlayerStatLine],
    af_team_to_fifa: dict[int, str],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Write a row per (player, competition, season) into player_match_stats-ish
    aggregated layer. Keeps the data accessible to the golden-boot model.

    Uses a dedicated aggregated table `player_season_stats` (created if missing).
    """
    conn = get_conn(db_path)
    try:
        # Create lightweight aggregated table on the fly
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS player_season_stats (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id       INTEGER NOT NULL,
                team_code       TEXT,
                competition     TEXT NOT NULL,
                league_id       INTEGER NOT NULL,
                season          INTEGER NOT NULL,
                games_played    INTEGER,
                minutes         INTEGER,
                goals           INTEGER,
                assists         INTEGER,
                shots           INTEGER,
                shots_on_target INTEGER,
                rating          REAL,
                source          TEXT,
                captured_at     TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (player_id) REFERENCES players(id),
                UNIQUE (player_id, league_id, season)
            );
            CREATE INDEX IF NOT EXISTS idx_pss_player ON player_season_stats(player_id);
            CREATE INDEX IF NOT EXISTS idx_pss_team ON player_season_stats(team_code, season);
            """
        )

        # Upsert players first (using api_football_id as natural key)
        existing = {
            r["api_football_id"]: r["id"]
            for r in conn.execute(
                "SELECT id, api_football_id FROM players WHERE api_football_id IS NOT NULL"
            )
        }

        players_inserted = 0
        for s in stats:
            if s.af_player_id in existing:
                continue
            fifa_code = af_team_to_fifa.get(s.af_team_id)
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO players
                    (api_football_id, name, nationality_code, current_club)
                VALUES (?, ?, ?, ?)
                """,
                (
                    s.af_player_id,
                    s.name,
                    fifa_code,
                    s.team_name,
                ),
            )
            if cur.lastrowid:
                existing[s.af_player_id] = cur.lastrowid
                players_inserted += 1

        # Now write stats rows
        season_rows = 0
        for s in stats:
            pid = existing.get(s.af_player_id)
            if not pid:
                continue
            fifa_code = af_team_to_fifa.get(s.af_team_id)
            conn.execute(
                """
                INSERT INTO player_season_stats
                    (player_id, team_code, competition, league_id, season,
                     games_played, minutes, goals, assists,
                     shots, shots_on_target, rating, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'api_football')
                ON CONFLICT (player_id, league_id, season) DO UPDATE SET
                    team_code       = COALESCE(excluded.team_code,       team_code),
                    competition     = excluded.competition,
                    games_played    = excluded.games_played,
                    minutes         = excluded.minutes,
                    goals           = excluded.goals,
                    assists         = excluded.assists,
                    shots           = excluded.shots,
                    shots_on_target = excluded.shots_on_target,
                    rating          = excluded.rating
                """,
                (
                    pid, fifa_code, s.competition_label,
                    s.league_id, s.season,
                    s.games_played, s.minutes, s.goals, s.assists,
                    s.shots, s.shots_on_target, s.rating,
                ),
            )
            season_rows += 1
        return {"players_inserted": players_inserted, "season_rows_written": season_rows}
    finally:
        conn.close()


def main() -> None:
    import json

    with APIFootballClient() as client:
        st = client.status()
        print(
            f"Plan: {st['subscription']['plan']}, "
            f"used {st['requests']['current']}/{st['requests']['limit_day']}\n"
        )

        print("=== Sweeping top scorers across competitions ===")
        stats = sweep_all_competitions(client=client)
        print(f"\n  Total stat lines collected: {len(stats)}")
        print(f"  Unique players: {len({s.af_player_id for s in stats})}")

        # Load AF team → FIFA mapping
        conn = get_conn()
        af_to_fifa = {
            int(r["source_code"]): r["fifa_code"]
            for r in conn.execute(
                "SELECT source_code, fifa_code FROM team_code_map WHERE source='api_football'"
            )
        }
        conn.close()
        print(f"  Known AF team mappings: {len(af_to_fifa)}")

        print("\n=== Persisting to DB ===")
        result = persist_to_player_table(stats, af_to_fifa)
        print(json.dumps(result, indent=2))

        st = client.status()
        print(f"\nrequests used after: {st['requests']['current']}/{st['requests']['limit_day']}")


if __name__ == "__main__":
    main()
