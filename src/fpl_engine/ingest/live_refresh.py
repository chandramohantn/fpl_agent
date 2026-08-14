"""Live season data refresher.

Handles incremental ingestion of the current season's data from the FPL API.
Unlike historical loading (which pulls a full season at once), this component:

1. Detects which gameweeks have completed since the last refresh
2. Fetches only the new GW data
3. Appends to the existing Parquet store
4. Updates player/team/fixture snapshots with latest state
5. Invalidates and refreshes Understat xG data

Designed to run after each gameweek completes (~weekly during the season).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from fpl_engine.ingest.fpl_api import FPLClient
from fpl_engine.ingest.understat import UnderstatScraper
from fpl_engine.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


@dataclass
class RefreshResult:
    """Summary of what a refresh operation changed."""

    season: str
    timestamp: datetime
    new_gameweeks: list[int] = field(default_factory=list)
    total_new_rows: int = 0
    players_updated: int = 0
    fixtures_updated: int = 0
    understat_refreshed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def has_new_data(self) -> bool:
        return len(self.new_gameweeks) > 0

    def summary(self) -> str:
        if not self.has_new_data:
            return (
                f"[{self.season}] No new data. "
                "Season may not have started or all GWs already ingested."
            )
        lines = [
            f"[{self.season}] Refresh complete at {self.timestamp:%Y-%m-%d %H:%M}",
            f"  New gameweeks: {self.new_gameweeks}",
            f"  New GW rows: {self.total_new_rows:,}",
            f"  Players snapshot: {self.players_updated}",
            f"  Fixtures updated: {self.fixtures_updated}",
            f"  Understat refreshed: {self.understat_refreshed}",
        ]
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)


class LiveSeasonRefresher:
    """Incrementally refresh current season data.

    Usage:
        refresher = LiveSeasonRefresher(
            store=ParquetStore("data/processed"),
            understat_cache_dir="data/raw/understat",
        )
        result = await refresher.refresh()
        print(result.summary())
    """

    def __init__(
        self,
        store: ParquetStore,
        understat_cache_dir: str = "data/raw/understat",
        season: str | None = None,
    ) -> None:
        self.store = store
        self.understat_cache_dir = understat_cache_dir
        self._season = season  # Auto-detect if None

    # ─── Main entry point ────────────────────────────────────────────────

    async def refresh(self, force_understat: bool = False) -> RefreshResult:
        """Run a full incremental refresh.

        Steps:
        1. Fetch bootstrap from FPL API to get current state
        2. Determine which GWs are new (finished but not yet stored)
        3. Fetch live GW data for each new GW
        4. Append to Parquet store
        5. Update player/team/fixture snapshots
        6. Refresh Understat if new GWs found

        Args:
            force_understat: Refresh Understat even if no new GWs found.
        """
        result = RefreshResult(
            season="unknown",
            timestamp=datetime.now(),
        )

        async with FPLClient() as client:
            # 1. Get current state
            bootstrap = await client.get_bootstrap()
            season = self._detect_season(bootstrap)
            result.season = season

            logger.info("Refreshing season %s", season)

            # 2. Determine which GWs are new
            all_events = bootstrap["events"]
            finished_gws = [e["id"] for e in all_events if e.get("finished")]
            stored_gws = self._get_stored_gameweeks(season)
            new_gws = sorted(set(finished_gws) - set(stored_gws))

            logger.info(
                "Finished GWs: %d, Stored GWs: %d, New: %d",
                len(finished_gws),
                len(stored_gws),
                len(new_gws),
            )

            # Update player/team/fixture snapshots (always, since prices/form change daily)
            players_df = pd.DataFrame(bootstrap["elements"])
            teams_df = pd.DataFrame(bootstrap["teams"])
            self.store.save_players(players_df, season)
            self.store.save_teams(teams_df, season)
            result.players_updated = len(players_df)

            fixtures_raw = await client.get_fixtures()
            fixtures_df = pd.DataFrame(fixtures_raw)
            self.store.save_fixtures(fixtures_df, season)
            result.fixtures_updated = len(fixtures_df)

            if not new_gws and not force_understat:
                logger.info("No new gameweeks to ingest. Player & fixture snapshots updated.")
                return result

        # 6. Refresh Understat
        if new_gws or force_understat:
            try:
                await self._refresh_understat(season)
                result.understat_refreshed = True
            except Exception as e:
                logger.error("Understat refresh failed: %s", e)
                result.errors.append(f"Understat: {e}")

        logger.info(result.summary())
        return result

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _detect_season(self, bootstrap: dict) -> str:
        """Detect the current season string from bootstrap data.

        The FPL API doesn't explicitly state the season, so we infer
        from the first gameweek's deadline.
        """
        if self._season:
            return self._season

        events = bootstrap["events"]
        if not events:
            return "unknown"

        # GW1 deadline tells us the season start year
        gw1_deadline = events[0].get("deadline_time", "")
        if gw1_deadline:
            year = int(gw1_deadline[:4])
            # EPL seasons start in Aug and end in May
            # If GW1 is in 2025, season is "2025-26"
            month = int(gw1_deadline[5:7])
            if month >= 7:  # Aug start
                return f"{year}-{str(year + 1)[-2:]}"
            else:  # Unlikely, but handle Jan+ start
                return f"{year - 1}-{str(year)[-2:]}"

        return "unknown"

    def _get_stored_gameweeks(self, season: str) -> list[int]:
        """Get list of gameweeks already stored for a season."""
        try:
            existing = self.store.load_gameweeks(season)
            if "gameweek" in existing.columns:
                return sorted(existing["gameweek"].unique().tolist())
            elif "round" in existing.columns:
                return sorted(existing["round"].unique().tolist())
            return []
        except FileNotFoundError:
            return []

    async def _fetch_new_gameweeks(
        self, client: FPLClient, gameweeks: list[int]
    ) -> list[pd.DataFrame]:
        """Fetch live data for specific gameweeks and transform to GW format.

        Uses the /event/{gw}/live/ endpoint which gives all player stats
        for a completed gameweek.
        """
        frames: list[pd.DataFrame] = []

        for gw in gameweeks:
            logger.info("Fetching live data for GW%d", gw)
            try:
                live_data = await client.get_live_gameweek(gw)
                elements = live_data.get("elements", [])

                # Transform live data to flat rows
                rows = []
                for element in elements:
                    stats = element.get("stats", {})
                    row = {
                        "element": element["id"],
                        "gameweek": gw,
                        **stats,
                    }
                    rows.append(row)

                if rows:
                    df = pd.DataFrame(rows)
                    df["gameweek"] = gw
                    frames.append(df)
                    logger.info("  GW%d: %d player records", gw, len(df))

            except Exception as e:
                logger.error("  GW%d fetch failed: %s", gw, e)

        return frames

    def _append_gameweeks(self, new_data: pd.DataFrame, season: str) -> None:
        """Append new gameweek data to the existing store.

        If the season file exists, concatenate. Otherwise create it.
        """
        try:
            existing = self.store.load_gameweeks(season)
            combined = pd.concat([existing, new_data], ignore_index=True)
            # Deduplicate in case of overlap (by element + gameweek)
            if "element" in combined.columns and "gameweek" in combined.columns:
                combined = combined.drop_duplicates(
                    subset=["element", "gameweek"], keep="last"
                )
            self.store.save_gameweeks(combined, season)
            logger.info(
                "Appended %d rows → total %d rows for %s",
                len(new_data),
                len(combined),
                season,
            )
        except FileNotFoundError:
            self.store.save_gameweeks(new_data, season)
            logger.info("Created new GW store with %d rows for %s", len(new_data), season)

    async def _refresh_understat(self, season: str) -> None:
        """Refresh Understat xG data for the current season.

        Invalidates the cached JSON and re-fetches from the AJAX endpoint.
        """
        from pathlib import Path

        from fpl_engine.ingest.understat import SEASON_MAP

        scraper = UnderstatScraper(cache_dir=self.understat_cache_dir)

        # Invalidate cache for this season's league data
        year = SEASON_MAP.get(season, season)
        cache_file = Path(self.understat_cache_dir) / year / "league_data.json"
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Invalidated Understat cache: %s", cache_file)

        # Re-fetch
        players = await scraper.get_league_players(season)
        players_df = pd.DataFrame(players)
        self.store.save_understat_players(players_df, season)
        logger.info("Understat refreshed: %d players for %s", len(players_df), season)
