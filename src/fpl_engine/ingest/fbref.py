"""FBref data scraper.

Uses the `soccerdata` library to extract advanced stats from FBref,
which provides StatsBomb-derived data including:
- Team shooting stats (shots, shots on target, xG, shot accuracy)
- Team defensive stats (via opponent shooting stats)
- Team passing and possession
- GK stats (save rate, goals against, shots on target faced)
- Player-level stats (per-90 rates for all major stats)
- Misc stats (cards, fouls, aerials)

Data is fetched via a headless browser (soccerdata handles Cloudflare bypass)
and cached locally. FBref has strict rate limiting (~20 requests/minute),
so fetching multiple stat types may take several minutes.

Requires: pip install soccerdata
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Map our season format to soccerdata's year format
# soccerdata uses the starting year: "2024-25" → 2024
SEASON_TO_YEAR = {
    "2021-22": 2021,
    "2022-23": 2022,
    "2023-24": 2023,
    "2024-25": 2024,
    "2025-26": 2025,
    "2026-27": 2026,
}

# Available stat types for team season stats
TEAM_STAT_TYPES = ["standard", "shooting", "keeper", "playing_time", "misc"]


class FBrefScraper:
    """FBref data scraper using soccerdata library.

    Handles Cloudflare bypass via headless browser, caching, and
    rate limiting. Extracts team and player stats for the EPL.

    Usage:
        scraper = FBrefScraper(cache_dir="data/raw/fbref")
        team_shooting = await scraper.get_team_stats("2025-26", "shooting")
        opp_shooting = await scraper.get_opponent_stats("2025-26", "shooting")
        players = await scraper.get_player_stats("2025-26", "standard")
    """

    def __init__(self, cache_dir: str | Path = "data/raw/fbref") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SOCCERDATA_DIR"] = str(self.cache_dir)

    def _get_fbref(self, season: str):
        """Create soccerdata FBref instance for a season."""
        import soccerdata as sd

        year = SEASON_TO_YEAR.get(season)
        if year is None:
            raise ValueError(f"Unknown season: {season}. Known: {list(SEASON_TO_YEAR.keys())}")

        return sd.FBref(leagues="ENG-Premier League", seasons=year)

    def _cache_path(self, season: str, stat_type: str, kind: str) -> Path:
        """Build a cache file path for parquet storage."""
        path = self.cache_dir / "processed" / season / f"{kind}_{stat_type}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-level column headers from FBref into single-level."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                f"{lvl1}_{lvl2}".strip("_") if lvl2 else lvl1
                for lvl1, lvl2 in df.columns
            ]
        return df

    # ─── Team Stats ──────────────────────────────────────────────────────

    def get_team_stats(self, season: str, stat_type: str = "standard") -> pd.DataFrame:
        """Get team season stats (what teams produce).

        Args:
            season: Season string (e.g., "2025-26").
            stat_type: One of: standard, shooting, keeper, playing_time, misc.

        Returns:
            DataFrame with team stats, flattened column names.
        """
        cache = self._cache_path(season, stat_type, "team")
        if cache.exists():
            logger.debug("Cache hit: %s", cache)
            return pd.read_parquet(cache)

        logger.info("Fetching FBref team %s stats for %s", stat_type, season)
        fbref = self._get_fbref(season)
        df = fbref.read_team_season_stats(stat_type=stat_type)

        # Reset index (league, season, team are in index)
        df = df.reset_index()
        df = self._flatten_columns(df)

        # Save to cache
        df.to_parquet(cache, index=False)
        logger.info("Saved %d teams to %s", len(df), cache)
        return df

    def get_opponent_stats(self, season: str, stat_type: str = "standard") -> pd.DataFrame:
        """Get opponent stats (what teams concede).

        This is the critical data for opposition modeling — it tells us
        how many shots, goals, etc. each team allows their opponents to have.

        Args:
            season: Season string.
            stat_type: One of: standard, shooting.

        Returns:
            DataFrame with opponent stats, flattened column names.
        """
        cache = self._cache_path(season, stat_type, "opponent")
        if cache.exists():
            logger.debug("Cache hit: %s", cache)
            return pd.read_parquet(cache)

        logger.info("Fetching FBref opponent %s stats for %s", stat_type, season)
        fbref = self._get_fbref(season)
        df = fbref.read_team_season_stats(stat_type=stat_type, opponent_stats=True)

        df = df.reset_index()
        df = self._flatten_columns(df)

        df.to_parquet(cache, index=False)
        logger.info("Saved %d teams (opponent stats) to %s", len(df), cache)
        return df

    # ─── Player Stats ────────────────────────────────────────────────────

    def get_player_stats(self, season: str, stat_type: str = "standard") -> pd.DataFrame:
        """Get player season stats.

        Args:
            season: Season string.
            stat_type: One of: standard, shooting, keeper, playing_time, misc.

        Returns:
            DataFrame with per-player stats, flattened column names.
        """
        cache = self._cache_path(season, stat_type, "player")
        if cache.exists():
            logger.debug("Cache hit: %s", cache)
            return pd.read_parquet(cache)

        logger.info("Fetching FBref player %s stats for %s", stat_type, season)
        fbref = self._get_fbref(season)
        df = fbref.read_player_season_stats(stat_type=stat_type)

        df = df.reset_index()
        df = self._flatten_columns(df)

        df.to_parquet(cache, index=False)
        logger.info("Saved %d players to %s", len(df), cache)
        return df

    # ─── GK Stats ────────────────────────────────────────────────────────

    def get_gk_stats(self, season: str) -> pd.DataFrame:
        """Get goalkeeper stats (saves, save%, goals against, etc.)."""
        return self.get_team_stats(season, stat_type="keeper")

    # ─── Bulk fetch ──────────────────────────────────────────────────────

    def fetch_all_team_stats(self, season: str) -> dict[str, pd.DataFrame]:
        """Fetch all available team stat types for a season.

        Returns dict mapping stat_type → DataFrame.
        Includes both team and opponent versions.
        """
        import time

        results = {}

        for stat_type in TEAM_STAT_TYPES:
            try:
                results[f"team_{stat_type}"] = self.get_team_stats(season, stat_type)
                time.sleep(4)
            except Exception as e:
                logger.warning("Failed team_%s: %s", stat_type, e)

        # Opponent stats (only standard and shooting available)
        for stat_type in ["standard", "shooting"]:
            try:
                results[f"opponent_{stat_type}"] = self.get_opponent_stats(season, stat_type)
                time.sleep(4)
            except Exception as e:
                logger.warning("Failed opponent_%s: %s", stat_type, e)

        logger.info("Fetched %d stat tables for %s", len(results), season)
        return results

    def fetch_all_player_stats(self, season: str) -> dict[str, pd.DataFrame]:
        """Fetch all player stat types for a season."""
        import time

        results = {}
        for stat_type in TEAM_STAT_TYPES:
            try:
                results[f"player_{stat_type}"] = self.get_player_stats(season, stat_type)
                time.sleep(4)
            except Exception as e:
                logger.warning("Failed player_%s: %s", stat_type, e)

        return results

    # ─── Utility ─────────────────────────────────────────────────────────

    def clear_cache(self, season: str | None = None) -> None:
        """Clear cached FBref data."""
        import shutil

        if season:
            target = self.cache_dir / "processed" / season
        else:
            target = self.cache_dir / "processed"

        if target.exists():
            shutil.rmtree(target)
            logger.info("Cleared FBref cache: %s", target)
