"""Understat data scraper.

Understat provides detailed xG, xA, and shot-level data for the
top 5 European leagues.

Source: https://understat.com

Data access:
- League data: AJAX endpoint at /getLeagueData/{league}/{season}
- Player data: AJAX endpoint at /getPlayerData/{player_id}
- Match data: AJAX endpoint at /getMatchData/{match_id}
- Legacy fallback: Inline JSON in HTML <script> tags (older pages)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

UNDERSTAT_BASE = "https://understat.com"

# Understat uses calendar years for season identification
# e.g., 2023 means the 2023-24 season
SEASON_MAP = {
    "2016-17": "2016",
    "2017-18": "2017",
    "2018-19": "2018",
    "2019-20": "2019",
    "2020-21": "2020",
    "2021-22": "2021",
    "2022-23": "2022",
    "2023-24": "2023",
    "2024-25": "2024",
    "2025-26": "2025",
    "2026-27": "2026",
}


def _decode_understat_json(encoded: str) -> Any:
    """Decode Understat's hex-escaped JSON strings.

    Understat encodes JSON data with \\xHH hex escapes in their
    inline JavaScript. We decode these to get valid JSON.
    """
    # Replace hex escapes like \x27 with actual characters
    decoded = encoded.encode("utf-8").decode("unicode_escape")
    return json.loads(decoded)


def _extract_json_var(html: str, var_name: str) -> Any:
    """Extract a JavaScript variable's JSON value from HTML source.

    Understat embeds data like:
        var datesData = JSON.parse('...');
    """
    pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
    match = re.search(pattern, html)
    if not match:
        raise ValueError(f"Could not find variable '{var_name}' in page source")
    return _decode_understat_json(match.group(1))


class UnderstatScraper:
    """Scraper for Understat xG data.

    Uses Understat's AJAX JSON API for league/player/match data.
    Falls back to HTML parsing for older pages if needed.

    Usage:
        scraper = UnderstatScraper(cache_dir="data/raw/understat")
        players = await scraper.get_league_players("2024-25")
        player_matches = await scraper.get_player_matches(player_id=1250)
        shots = await scraper.get_player_shots(player_id=1250)
    """

    def __init__(self, cache_dir: str | Path = "data/raw/understat") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, *parts: str) -> Path:
        """Build a cache file path."""
        path = self.cache_dir / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def _fetch_json(self, url: str, cache_path: Path | None = None) -> Any:
        """Fetch JSON from an AJAX endpoint, with optional caching."""
        if cache_path and cache_path.exists():
            logger.debug("Cache hit: %s", cache_path)
            return json.loads(cache_path.read_text(encoding="utf-8"))

        logger.info("Fetching: %s", url)
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "X-Requested-With": "XMLHttpRequest",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        data = response.json()
        if cache_path:
            cache_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
        return data

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def _fetch_page(self, url: str, cache_path: Path | None = None) -> str:
        """Fetch an HTML page, with optional caching (legacy fallback)."""
        if cache_path and cache_path.exists():
            logger.debug("Cache hit: %s", cache_path)
            return cache_path.read_text(encoding="utf-8")

        logger.info("Fetching: %s", url)
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        html = response.text
        if cache_path:
            cache_path.write_text(html, encoding="utf-8")
        return html

    # ─── League-level data (AJAX API) ────────────────────────────────────

    async def get_league_data(self, season: str) -> dict[str, Any]:
        """Get full league data for a season via AJAX endpoint.

        Returns dict with keys: teams, players, dates
        """
        year = SEASON_MAP.get(season, season)
        url = f"{UNDERSTAT_BASE}/getLeagueData/EPL/{year}"
        cache = self._cache_path(year, "league_data.json")
        return await self._fetch_json(url, cache)

    async def get_league_players(self, season: str) -> list[dict[str, Any]]:
        """Get all EPL player season summaries for a given season.

        Returns list of dicts with keys:
            id, player_name, team_title, games, minutes, goals, assists,
            xG, xA, npxG, xGChain, xGBuildup, shots, key_passes, ...
        """
        data = await self.get_league_data(season)
        players = data.get("players", {})
        # Players may be a dict keyed by ID or a list
        if isinstance(players, dict):
            return list(players.values())
        return players

    async def get_league_teams(self, season: str) -> dict[str, Any]:
        """Get team-level stats for a season.

        Returns dict mapping team IDs to their stats (history, title, etc).
        """
        data = await self.get_league_data(season)
        return data.get("teams", {})

    # ─── Player-level data ───────────────────────────────────────────────

    async def get_player_data(self, player_id: int) -> dict[str, Any]:
        """Get full player data via AJAX endpoint.

        Returns dict with keys: matches, shots, groups, etc.
        """
        url = f"{UNDERSTAT_BASE}/getPlayerData/{player_id}"
        cache = self._cache_path("players", f"{player_id}.json")
        return await self._fetch_json(url, cache)

    async def get_player_matches(self, player_id: int) -> list[dict[str, Any]]:
        """Get match-by-match data for a player.

        Returns list of dicts with per-match xG, xA, goals, assists, etc.
        """
        data = await self.get_player_data(player_id)
        matches = data.get("matches", data.get("matchesData", []))
        if isinstance(matches, dict):
            return list(matches.values())
        return matches

    async def get_player_shots(self, player_id: int) -> list[dict[str, Any]]:
        """Get shot-level data for a player.

        Returns list of dicts with x, y, xG, result, situation, etc.
        """
        data = await self.get_player_data(player_id)
        shots = data.get("shots", data.get("shotsData", []))
        if isinstance(shots, dict):
            return list(shots.values())
        return shots

    async def get_player_grouped_stats(self, player_id: int) -> dict[str, Any]:
        """Get grouped stats for a player (by season, situation, etc.)."""
        data = await self.get_player_data(player_id)
        return data.get("groups", data.get("groupsData", {}))

    # ─── Match-level data ────────────────────────────────────────────────

    async def get_match_shots(self, match_id: int) -> dict[str, list[dict[str, Any]]]:
        """Get shot data for a specific match.

        Returns dict with keys 'h' (home) and 'a' (away),
        each containing a list of shot dicts.
        """
        url = f"{UNDERSTAT_BASE}/getMatchData/{match_id}"
        cache = self._cache_path("matches", f"{match_id}.json")
        data = await self._fetch_json(url, cache)
        return data.get("shots", data.get("shotsData", {}))

    async def get_match_rosters(self, match_id: int) -> dict[str, Any]:
        """Get roster/lineup data for a match."""
        url = f"{UNDERSTAT_BASE}/getMatchData/{match_id}"
        cache = self._cache_path("matches", f"{match_id}.json")
        data = await self._fetch_json(url, cache)
        return data.get("rosters", data.get("rostersData", {}))

    # ─── Bulk operations ─────────────────────────────────────────────────

    async def get_all_player_ids(self, season: str) -> list[int]:
        """Get all Understat player IDs for EPL in a given season."""
        players = await self.get_league_players(season)
        return [int(p["id"]) for p in players]

    def clear_cache(self) -> None:
        """Remove all cached files."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared: %s", self.cache_dir)
