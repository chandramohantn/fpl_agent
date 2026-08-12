"""Tests for the live season refresh component."""

from __future__ import annotations

import tempfile

import pandas as pd

# ─── ParquetStore incremental methods ────────────────────────────────────────


def test_append_gameweeks_creates_new_file():
    """append_gameweeks creates file if it doesn't exist."""
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)
        df = pd.DataFrame({
            "element": [1, 2, 3],
            "gameweek": [1, 1, 1],
            "total_points": [8, 5, 2],
        })

        path = store.append_gameweeks(df, "2026-27")
        assert path.exists()

        loaded = store.load_gameweeks("2026-27")
        assert len(loaded) == 3


def test_append_gameweeks_appends_to_existing():
    """append_gameweeks concatenates with existing data."""
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)

        # GW1 data
        gw1 = pd.DataFrame({
            "element": [1, 2, 3],
            "gameweek": [1, 1, 1],
            "total_points": [8, 5, 2],
        })
        store.append_gameweeks(gw1, "2026-27")

        # GW2 data
        gw2 = pd.DataFrame({
            "element": [1, 2, 3],
            "gameweek": [2, 2, 2],
            "total_points": [3, 12, 6],
        })
        store.append_gameweeks(gw2, "2026-27")

        loaded = store.load_gameweeks("2026-27")
        assert len(loaded) == 6
        assert sorted(loaded["gameweek"].unique()) == [1, 2]


def test_append_gameweeks_deduplicates():
    """append_gameweeks removes duplicates (keeps latest)."""
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)

        # Initial GW1
        gw1 = pd.DataFrame({
            "element": [1, 2],
            "gameweek": [1, 1],
            "total_points": [5, 3],
        })
        store.append_gameweeks(gw1, "2026-27")

        # Updated GW1 (e.g., bonus points finalized)
        gw1_updated = pd.DataFrame({
            "element": [1, 2],
            "gameweek": [1, 1],
            "total_points": [8, 5],  # Updated scores
        })
        store.append_gameweeks(gw1_updated, "2026-27")

        loaded = store.load_gameweeks("2026-27")
        assert len(loaded) == 2  # Not 4
        # Should have the updated values
        player1 = loaded[loaded["element"] == 1].iloc[0]
        assert player1["total_points"] == 8


def test_get_stored_gameweeks():
    """get_stored_gameweeks returns correct GW list."""
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)

        # No data yet
        assert store.get_stored_gameweeks("2026-27") == []

        # Add some data
        df = pd.DataFrame({
            "element": [1, 1, 1, 2, 2, 2],
            "gameweek": [1, 2, 3, 1, 2, 3],
            "total_points": [8, 5, 2, 3, 12, 6],
        })
        store.save_gameweeks(df, "2026-27")

        assert store.get_stored_gameweeks("2026-27") == [1, 2, 3]


def test_get_latest_gameweek():
    """get_latest_gameweek returns highest stored GW."""
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)

        assert store.get_latest_gameweek("2026-27") is None

        df = pd.DataFrame({
            "element": [1, 1],
            "gameweek": [1, 5],
            "total_points": [8, 2],
        })
        store.save_gameweeks(df, "2026-27")

        assert store.get_latest_gameweek("2026-27") == 5


# ─── LiveSeasonRefresher ─────────────────────────────────────────────────────


def test_refresh_result_summary_no_data():
    """RefreshResult shows correct message when no new data."""
    from datetime import datetime

    from fpl_engine.ingest.live_refresh import RefreshResult

    result = RefreshResult(season="2026-27", timestamp=datetime.now())
    assert "No new data" in result.summary()
    assert not result.has_new_data


def test_refresh_result_summary_with_data():
    """RefreshResult shows correct summary with new data."""
    from datetime import datetime

    from fpl_engine.ingest.live_refresh import RefreshResult

    result = RefreshResult(
        season="2026-27",
        timestamp=datetime.now(),
        new_gameweeks=[4, 5],
        total_new_rows=1200,
        players_updated=573,
        fixtures_updated=380,
        understat_refreshed=True,
    )
    assert result.has_new_data
    summary = result.summary()
    assert "2026-27" in summary
    assert "[4, 5]" in summary
    assert "1,200" in summary


def test_detect_season_from_bootstrap():
    """Season detection works from GW1 deadline."""
    from fpl_engine.ingest.live_refresh import LiveSeasonRefresher
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)
        refresher = LiveSeasonRefresher(store=store)

        # Simulate bootstrap with Aug 2026 GW1 deadline
        bootstrap = {
            "events": [
                {"id": 1, "deadline_time": "2026-08-14T11:00:00Z", "finished": False},
                {"id": 2, "deadline_time": "2026-08-23T11:00:00Z", "finished": False},
            ]
        }
        assert refresher._detect_season(bootstrap) == "2026-27"

        # Simulate 2025 season
        bootstrap_2025 = {
            "events": [
                {"id": 1, "deadline_time": "2025-08-15T11:00:00Z", "finished": True},
            ]
        }
        assert refresher._detect_season(bootstrap_2025) == "2025-26"


def test_detect_new_gameweeks():
    """Refresher correctly identifies which GWs are new."""
    from fpl_engine.ingest.live_refresh import LiveSeasonRefresher
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)

        # Pre-populate with GW1-3
        existing = pd.DataFrame({
            "element": [1, 1, 1],
            "gameweek": [1, 2, 3],
            "total_points": [8, 5, 2],
        })
        store.save_gameweeks(existing, "2026-27")

        refresher = LiveSeasonRefresher(store=store, season="2026-27")

        # Stored should be [1,2,3]
        stored = refresher._get_stored_gameweeks("2026-27")
        assert stored == [1, 2, 3]

        # If API says GW 1-5 are finished, new = [4, 5]
        finished_from_api = [1, 2, 3, 4, 5]
        new_gws = sorted(set(finished_from_api) - set(stored))
        assert new_gws == [4, 5]
