"""Parquet-based storage layer for FPL data.

Organizes data into a structured data lake:

    data/
    ├── processed/
    │   ├── players/
    │   │   ├── season=2023-24/
    │   │   │   └── players.parquet
    │   │   └── ...
    │   ├── gameweeks/
    │   │   ├── season=2023-24/
    │   │   │   └── gameweeks.parquet
    │   │   └── ...
    │   ├── fixtures/
    │   │   └── season=2023-24/
    │   │       └── fixtures.parquet
    │   ├── teams/
    │   │   └── season=2023-24/
    │   │       └── teams.parquet
    │   └── understat/
    │       ├── players/
    │       │   └── season=2023-24/
    │       │       └── players.parquet
    │       └── shots/
    │           └── player_id=1250/
    │               └── shots.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

DataDomain = Literal[
    "players",
    "gameweeks",
    "fixtures",
    "teams",
    "understat_players",
    "understat_shots",
    "understat_matches",
]


class ParquetStore:
    """Read/write Parquet files in a structured data lake layout.

    Usage:
        store = ParquetStore(base_dir="data/processed")
        store.save_players(df, season="2023-24")
        df = store.load_players(season="2023-24")
    """

    def __init__(self, base_dir: str | Path = "data/processed") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, domain: str, partition: str, filename: str) -> Path:
        """Build path for a partitioned parquet file."""
        path = self.base_dir / domain / partition / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ─── Write operations ────────────────────────────────────────────────

    def save_players(self, df: pd.DataFrame, season: str) -> Path:
        """Save player summary data for a season."""
        path = self._path("players", f"season={season}", "players.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %d players to %s", len(df), path)
        return path

    def save_gameweeks(self, df: pd.DataFrame, season: str) -> Path:
        """Save gameweek-level player performance data."""
        path = self._path("gameweeks", f"season={season}", "gameweeks.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %d GW rows to %s", len(df), path)
        return path

    def save_fixtures(self, df: pd.DataFrame, season: str) -> Path:
        """Save fixture data."""
        path = self._path("fixtures", f"season={season}", "fixtures.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %d fixtures to %s", len(df), path)
        return path

    def save_teams(self, df: pd.DataFrame, season: str) -> Path:
        """Save team data."""
        path = self._path("teams", f"season={season}", "teams.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %d teams to %s", len(df), path)
        return path

    def save_understat_players(self, df: pd.DataFrame, season: str) -> Path:
        """Save Understat player-level xG summary."""
        path = self._path("understat/players", f"season={season}", "players.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %d Understat player rows to %s", len(df), path)
        return path

    def save_understat_shots(self, df: pd.DataFrame, player_id: int) -> Path:
        """Save shot-level data for a player."""
        path = self._path("understat/shots", f"player_id={player_id}", "shots.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %d shots for player %d to %s", len(df), player_id, path)
        return path

    def save_understat_matches(self, df: pd.DataFrame, player_id: int) -> Path:
        """Save match-by-match data for a player."""
        path = self._path("understat/matches", f"player_id={player_id}", "matches.parquet")
        df.to_parquet(path, index=False, engine="pyarrow")
        logger.info("Saved %d matches for player %d to %s", len(df), player_id, path)
        return path

    # ─── Read operations ─────────────────────────────────────────────────

    def load_players(self, season: str) -> pd.DataFrame:
        """Load player summary for a season."""
        path = self._path("players", f"season={season}", "players.parquet")
        return pd.read_parquet(path, engine="pyarrow")

    def load_gameweeks(self, season: str) -> pd.DataFrame:
        """Load gameweek data for a season."""
        path = self._path("gameweeks", f"season={season}", "gameweeks.parquet")
        return pd.read_parquet(path, engine="pyarrow")

    def load_fixtures(self, season: str) -> pd.DataFrame:
        """Load fixtures for a season."""
        path = self._path("fixtures", f"season={season}", "fixtures.parquet")
        return pd.read_parquet(path, engine="pyarrow")

    def load_teams(self, season: str) -> pd.DataFrame:
        """Load teams for a season."""
        path = self._path("teams", f"season={season}", "teams.parquet")
        return pd.read_parquet(path, engine="pyarrow")

    def load_understat_players(self, season: str) -> pd.DataFrame:
        """Load Understat player summary for a season."""
        path = self._path("understat/players", f"season={season}", "players.parquet")
        return pd.read_parquet(path, engine="pyarrow")

    def load_understat_shots(self, player_id: int) -> pd.DataFrame:
        """Load shot data for a player."""
        path = self._path("understat/shots", f"player_id={player_id}", "shots.parquet")
        return pd.read_parquet(path, engine="pyarrow")

    def load_understat_matches(self, player_id: int) -> pd.DataFrame:
        """Load match-by-match data for a player."""
        path = self._path("understat/matches", f"player_id={player_id}", "matches.parquet")
        return pd.read_parquet(path, engine="pyarrow")

    # ─── Multi-season ────────────────────────────────────────────────────

    def load_all_gameweeks(self, seasons: list[str] | None = None) -> pd.DataFrame:
        """Load and concatenate gameweek data across seasons."""
        if seasons is None:
            # Discover available seasons
            gw_dir = self.base_dir / "gameweeks"
            if not gw_dir.exists():
                return pd.DataFrame()
            seasons = [
                d.name.replace("season=", "")
                for d in gw_dir.iterdir()
                if d.is_dir() and d.name.startswith("season=")
            ]

        frames = []
        for season in sorted(seasons):
            try:
                df = self.load_gameweeks(season)
                df["season"] = season
                frames.append(df)
            except FileNotFoundError:
                logger.warning("No gameweek data for %s", season)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ─── Incremental operations ──────────────────────────────────────────

    def append_gameweeks(self, new_data: pd.DataFrame, season: str) -> Path:
        """Append new gameweek rows to the existing season file.

        If the file doesn't exist, creates it. If it does, concatenates
        and deduplicates by (element, gameweek) keeping the latest version.

        Args:
            new_data: New GW rows to append.
            season: Season string.

        Returns:
            Path to the saved file.
        """
        path = self._path("gameweeks", f"season={season}", "gameweeks.parquet")

        if path.exists():
            existing = pd.read_parquet(path, engine="pyarrow")
            combined = pd.concat([existing, new_data], ignore_index=True)

            # Deduplicate: keep latest version of each (element, gameweek) pair
            dedup_cols = []
            if "element" in combined.columns and "gameweek" in combined.columns:
                dedup_cols = ["element", "gameweek"]
            elif "element" in combined.columns and "round" in combined.columns:
                dedup_cols = ["element", "round"]

            if dedup_cols:
                combined = combined.drop_duplicates(subset=dedup_cols, keep="last")

            combined.to_parquet(path, index=False, engine="pyarrow")
            logger.info(
                "Appended %d rows to %s (total: %d, season: %s)",
                len(new_data),
                path,
                len(combined),
                season,
            )
        else:
            new_data.to_parquet(path, index=False, engine="pyarrow")
            logger.info("Created %s with %d rows", path, len(new_data))

        return path

    def get_stored_gameweeks(self, season: str) -> list[int]:
        """Get sorted list of gameweek numbers already stored for a season.

        Returns empty list if no data exists for the season.
        """
        try:
            df = self.load_gameweeks(season)
        except FileNotFoundError:
            return []

        if "gameweek" in df.columns:
            return sorted(df["gameweek"].unique().tolist())
        elif "round" in df.columns:
            return sorted(df["round"].unique().tolist())
        return []

    def get_latest_gameweek(self, season: str) -> int | None:
        """Get the highest gameweek number stored for a season."""
        gws = self.get_stored_gameweeks(season)
        return gws[-1] if gws else None

    # ─── Utility ─────────────────────────────────────────────────────────

    def list_seasons(self, domain: str = "gameweeks") -> list[str]:
        """List available seasons for a data domain."""
        domain_dir = self.base_dir / domain
        if not domain_dir.exists():
            return []
        return sorted(
            d.name.replace("season=", "")
            for d in domain_dir.iterdir()
            if d.is_dir() and d.name.startswith("season=")
        )

    def exists(self, domain: str, partition: str, filename: str) -> bool:
        """Check if a specific parquet file exists."""
        return self._path(domain, partition, filename).exists()
