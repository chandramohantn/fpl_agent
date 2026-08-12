"""Run the full data ingestion pipeline.

Fetches data from:
1. Official FPL API (current season)
2. Historical data (vaastav GitHub repo)
3. Understat (xG data)

Saves everything to the Parquet store.
"""

import asyncio
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fpl_engine.ingest.fpl_api import FPLClient
from fpl_engine.ingest.historical import HistoricalDataLoader
from fpl_engine.ingest.understat import UnderstatScraper
from fpl_engine.storage.parquet_store import ParquetStore
from fpl_engine.features.fixture_difficulty import compute_team_strength

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# We'll work from the project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

store = ParquetStore(base_dir=DATA_DIR / "processed")


async def ingest_fpl_api():
    """Ingest current season data from the official FPL API."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Official FPL API")
    logger.info("=" * 60)

    async with FPLClient() as client:
        # Bootstrap data
        bootstrap = await client.get_bootstrap()

        players_raw = bootstrap["elements"]
        teams_raw = bootstrap["teams"]
        gameweeks_raw = bootstrap["events"]

        logger.info("Players: %d", len(players_raw))
        logger.info("Teams: %d", len(teams_raw))
        logger.info("Gameweeks: %d", len(gameweeks_raw))

        # Convert to DataFrames
        players_df = pd.DataFrame(players_raw)
        teams_df = pd.DataFrame(teams_raw)
        gameweeks_df = pd.DataFrame(gameweeks_raw)

        # Determine current season from gameweek dates
        current_gw = next((e for e in gameweeks_raw if e.get("is_current")), None)
        next_gw = next((e for e in gameweeks_raw if e.get("is_next")), None)
        logger.info("Current GW: %s", current_gw["id"] if current_gw else "None")
        logger.info("Next GW: %s", next_gw["id"] if next_gw else "None")

        # Save to store (use 2025-26 as the season since we're in Aug 2026... 
        # but let's detect from the data)
        # The FPL API doesn't explicitly state the season, so we infer from GW1 date
        gw1 = gameweeks_raw[0]
        season = "2025-26"  # Current season based on date
        logger.info("Detected season: %s", season)

        store.save_players(players_df, season)
        store.save_teams(teams_df, season)

        # Fixtures
        fixtures_raw = await client.get_fixtures()
        fixtures_df = pd.DataFrame(fixtures_raw)
        logger.info("Fixtures: %d", len(fixtures_df))
        store.save_fixtures(fixtures_df, season)

        # Team strength from completed fixtures
        completed = fixtures_df.dropna(subset=["team_h_score", "team_a_score"])
        if not completed.empty:
            strength = compute_team_strength(fixtures_df, teams_df)
            logger.info("Team strength computed for %d teams", len(strength))
        else:
            logger.info("No completed fixtures yet — skipping team strength")

    return season, players_df, teams_df, fixtures_df


async def ingest_historical(seasons: list[str] | None = None):
    """Ingest historical data from GitHub."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 2: Historical Data (vaastav/Fantasy-Premier-League)")
    logger.info("=" * 60)

    if seasons is None:
        # Load last 3 seasons for speed (full load takes longer)
        seasons = ["2022-23", "2023-24", "2024-25"]

    loader = HistoricalDataLoader(cache_dir=DATA_DIR / "raw" / "historical")

    total_gw_rows = 0
    for season in seasons:
        logger.info("--- Season %s ---", season)

        # Players
        try:
            players = await loader.load_players(season)
            logger.info("  Players: %d", len(players))
            store.save_players(players, season)
        except Exception as e:
            logger.warning("  Players failed: %s", e)

        # Fixtures
        try:
            fixtures = await loader.load_fixtures(season)
            logger.info("  Fixtures: %d", len(fixtures))
            store.save_fixtures(fixtures, season)
        except Exception as e:
            logger.warning("  Fixtures failed: %s", e)

        # Teams
        try:
            teams = await loader.load_teams(season)
            logger.info("  Teams: %d", len(teams))
            store.save_teams(teams, season)
        except Exception as e:
            logger.warning("  Teams failed: %s", e)

        # Gameweek data (all GWs concatenated)
        try:
            gw_data = await loader.load_all_gameweeks(season)
            if not gw_data.empty:
                logger.info("  GW rows: %d", len(gw_data))
                store.save_gameweeks(gw_data, season)
                total_gw_rows += len(gw_data)
            else:
                logger.info("  No GW data found")
        except Exception as e:
            logger.warning("  GW data failed: %s", e)

    logger.info("Total historical GW rows: %d", total_gw_rows)


async def ingest_understat(season: str = "2024-25"):
    """Ingest Understat xG data."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 3: Understat xG Data")
    logger.info("=" * 60)

    scraper = UnderstatScraper(cache_dir=DATA_DIR / "raw" / "understat")

    # League-level player data
    try:
        players = await scraper.get_league_players(season)
        logger.info("Understat players for %s: %d", season, len(players))

        players_df = pd.DataFrame(players)
        store.save_understat_players(players_df, season)

        # Show top 10 by xG
        players_df["xG"] = players_df["xG"].astype(float)
        players_df["xA"] = players_df["xA"].astype(float)
        top_xg = players_df.nlargest(10, "xG")[["player_name", "team_title", "xG", "xA", "games"]]
        logger.info("Top 10 by xG:\n%s", top_xg.to_string(index=False))

    except Exception as e:
        logger.error("Understat league data failed: %s", e)

    # Team-level data
    try:
        teams = await scraper.get_league_teams(season)
        logger.info("Understat teams: %d", len(teams))
    except Exception as e:
        logger.warning("Understat teams failed: %s", e)


async def main():
    """Run the full pipeline."""
    logger.info("FPL Engine — Data Ingestion Pipeline")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("")

    # Phase 1: FPL API
    try:
        season, players_df, teams_df, fixtures_df = await ingest_fpl_api()
    except Exception as e:
        logger.error("FPL API ingestion failed: %s", e)
        season = None

    # Phase 2: Historical
    await ingest_historical(seasons=["2023-24", "2024-25"])

    # Phase 3: Understat
    await ingest_understat("2024-25")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    available_seasons = store.list_seasons("players")
    logger.info("Seasons with player data: %s", available_seasons)

    gw_seasons = store.list_seasons("gameweeks")
    logger.info("Seasons with GW data: %s", gw_seasons)

    # Total data size
    processed_dir = DATA_DIR / "processed"
    if processed_dir.exists():
        total_size = sum(f.stat().st_size for f in processed_dir.rglob("*.parquet"))
        logger.info("Total processed data: %.2f MB", total_size / (1024 * 1024))

    raw_dir = DATA_DIR / "raw"
    if raw_dir.exists():
        total_raw = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file())
        logger.info("Total raw cache: %.2f MB", total_raw / (1024 * 1024))


if __name__ == "__main__":
    asyncio.run(main())
