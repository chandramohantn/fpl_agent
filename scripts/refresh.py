"""Live season refresh script.

Run this after each gameweek completes to incrementally update the data store
with the latest results, stats, and xG data.

Usage:
    # Standard refresh (detect and fetch new GWs)
    python scripts/refresh.py

    # Force Understat refresh even if no new GWs
    python scripts/refresh.py --force-understat

    # Specify a season manually
    python scripts/refresh.py --season 2026-27

    # Dry run (show what would be fetched without saving)
    python scripts/refresh.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fpl_engine.ingest.live_refresh import LiveSeasonRefresher, RefreshResult
from fpl_engine.storage.parquet_store import ParquetStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


async def run_refresh(
    season: str | None = None,
    force_understat: bool = False,
    dry_run: bool = False,
) -> RefreshResult:
    """Run the live season refresh."""
    store = ParquetStore(base_dir=DATA_DIR / "processed")

    refresher = LiveSeasonRefresher(
        store=store,
        understat_cache_dir=str(DATA_DIR / "raw" / "understat"),
        season=season,
    )

    if dry_run:
        return await _dry_run(refresher, store, season)

    result = await refresher.refresh(force_understat=force_understat)
    return result


async def _dry_run(
    refresher: LiveSeasonRefresher,
    store: ParquetStore,
    season: str | None,
) -> RefreshResult:
    """Show what would be refreshed without writing anything."""
    from fpl_engine.ingest.fpl_api import FPLClient

    logger.info("DRY RUN — no data will be written")

    async with FPLClient() as client:
        bootstrap = await client.get_bootstrap()
        detected_season = refresher._detect_season(bootstrap)
        season = season or detected_season

        all_events = bootstrap["events"]
        finished_gws = [e["id"] for e in all_events if e.get("finished")]
        stored_gws = store.get_stored_gameweeks(season)
        new_gws = sorted(set(finished_gws) - set(stored_gws))

        current_gw = next((e for e in all_events if e.get("is_current")), None)
        next_gw = next((e for e in all_events if e.get("is_next")), None)

        logger.info("Season: %s", season)
        logger.info("Current GW: %s", current_gw["id"] if current_gw else "None")
        logger.info("Next GW: %s", next_gw["id"] if next_gw else "None")
        logger.info("Finished GWs from API: %s", finished_gws[-5:] if finished_gws else [])
        logger.info("Already stored GWs: %s", stored_gws[-5:] if stored_gws else [])
        logger.info("New GWs to fetch: %s", new_gws)
        logger.info("Players in API: %d", len(bootstrap["elements"]))

    from datetime import datetime

    return RefreshResult(
        season=season,
        timestamp=datetime.now(),
        new_gameweeks=new_gws,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Refresh current season data from FPL API + Understat"
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Season to refresh (e.g., '2026-27'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--force-understat",
        action="store_true",
        help="Force Understat refresh even if no new GWs found.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without writing any data.",
    )
    args = parser.parse_args()

    logger.info("FPL Engine — Live Season Refresh")
    logger.info("=" * 50)

    result = asyncio.run(
        run_refresh(
            season=args.season,
            force_understat=args.force_understat,
            dry_run=args.dry_run,
        )
    )

    print()
    print(result.summary())

    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
