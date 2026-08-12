"""Historical FPL data loader.

Loads data from the vaastav/Fantasy-Premier-League GitHub repository,
which contains cleaned historical data going back to 2016-17.

Repo: https://github.com/vaastav/Fantasy-Premier-League
Structure:
    data/{season}/
        cleaned_players.csv
        fixtures.csv
        teams.csv
        gws/
            gw1.csv ... gw38.csv
        players/
            {player_name}_{id}/
                gw.csv
                history.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

# Seasons available in the repo
AVAILABLE_SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


class HistoricalDataLoader:
    """Load historical FPL data from GitHub or local cache.

    Usage:
        loader = HistoricalDataLoader(cache_dir="data/raw/historical")
        players = await loader.load_players("2023-24")
        gw_data = await loader.load_gameweek("2023-24", 1)
    """

    def __init__(self, cache_dir: str | Path = "data/raw/historical") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, season: str, filename: str) -> Path:
        """Get local cache path for a file."""
        path = self.cache_dir / season / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _fetch_csv(self, url: str, cache_path: Path) -> pd.DataFrame:
        """Fetch CSV from URL, cache locally, return as DataFrame."""
        if cache_path.exists():
            logger.debug("Cache hit: %s", cache_path)
            return pd.read_csv(cache_path)

        logger.info("Downloading: %s", url)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

        cache_path.write_bytes(response.content)
        return pd.read_csv(cache_path)

    # ─── Season-level data ───────────────────────────────────────────────

    async def load_players(self, season: str) -> pd.DataFrame:
        """Load cleaned player summary for a season.

        Columns include: element, first_name, second_name, web_name, team,
        position, total_points, minutes, goals_scored, assists, etc.
        """
        url = f"{GITHUB_RAW_BASE}/{season}/cleaned_players.csv"
        cache = self._cache_path(season, "cleaned_players.csv")
        return await self._fetch_csv(url, cache)

    async def load_fixtures(self, season: str) -> pd.DataFrame:
        """Load fixtures for a season."""
        url = f"{GITHUB_RAW_BASE}/{season}/fixtures.csv"
        cache = self._cache_path(season, "fixtures.csv")
        return await self._fetch_csv(url, cache)

    async def load_teams(self, season: str) -> pd.DataFrame:
        """Load team data for a season."""
        url = f"{GITHUB_RAW_BASE}/{season}/teams.csv"
        cache = self._cache_path(season, "teams.csv")
        return await self._fetch_csv(url, cache)

    # ─── Gameweek-level data ─────────────────────────────────────────────

    async def load_gameweek(self, season: str, gw: int) -> pd.DataFrame:
        """Load all player performances for a specific gameweek.

        This is the merged GW file with all players' stats for that round.
        """
        url = f"{GITHUB_RAW_BASE}/{season}/gws/gw{gw}.csv"
        cache = self._cache_path(season, f"gws/gw{gw}.csv")
        return await self._fetch_csv(url, cache)

    async def load_all_gameweeks(self, season: str, max_gw: int = 38) -> pd.DataFrame:
        """Load and concatenate all gameweek data for a season."""
        frames: list[pd.DataFrame] = []
        for gw in range(1, max_gw + 1):
            try:
                df = await self.load_gameweek(season, gw)
                df["gameweek"] = gw
                frames.append(df)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.info("GW%d not found for %s — season may not be complete", gw, season)
                    break
                raise
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ─── Multi-season ────────────────────────────────────────────────────

    async def load_multi_season_gameweeks(
        self,
        seasons: list[str] | None = None,
        max_gw: int = 38,
    ) -> pd.DataFrame:
        """Load gameweek data across multiple seasons.

        Adds a 'season' column to distinguish between seasons.
        """
        if seasons is None:
            seasons = AVAILABLE_SEASONS

        frames: list[pd.DataFrame] = []
        for season in seasons:
            logger.info("Loading season %s", season)
            df = await self.load_all_gameweeks(season, max_gw)
            if not df.empty:
                df["season"] = season
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
