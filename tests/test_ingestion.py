"""End-to-end verification of the data ingestion layer.

Tests that all modules import correctly and core logic works
without requiring network access (uses mocked/synthetic data).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# ─── Import tests ────────────────────────────────────────────────────────────


def test_imports():
    """All modules should import without errors."""
    from fpl_engine import __version__
    from fpl_engine.features.fixture_difficulty import (  # noqa: F401
        compute_fixture_difficulty,
        compute_rolling_strength,
        compute_team_strength,
    )
    from fpl_engine.ingest.fpl_api import FPLClient  # noqa: F401
    from fpl_engine.ingest.historical import HistoricalDataLoader  # noqa: F401
    from fpl_engine.ingest.understat import UnderstatScraper  # noqa: F401
    from fpl_engine.models.schemas import (  # noqa: F401
        Fixture,
        Gameweek,
        GameweekHistory,
        Player,
        Position,
        Team,
        UnderstatPlayer,
        UnderstatShot,
    )
    from fpl_engine.storage.parquet_store import ParquetStore  # noqa: F401

    assert __version__ == "0.1.0"


# ─── Schema tests ────────────────────────────────────────────────────────────


def test_player_schema():
    """Player model validates correctly."""
    from fpl_engine.models.schemas import Player

    player = Player(
        id=1,
        web_name="Salah",
        first_name="Mohamed",
        second_name="Salah",
        team=10,
        element_type=3,
        now_cost=130,
        total_points=250,
        points_per_game=7.5,
        minutes=2800,
        goals_scored=20,
        assists=12,
        clean_sheets=10,
        saves=0,
        bonus=30,
        bps=600,
        form=8.5,
        selected_by_percent=45.2,
        transfers_in_event=50000,
        transfers_out_event=10000,
        status="a",
    )
    assert player.web_name == "Salah"
    assert player.now_cost == 130


def test_position_enum():
    """Position enum maps correctly."""
    from fpl_engine.models.schemas import Position

    assert Position.GKP == 1
    assert Position.FWD == 4


# ─── Storage tests ───────────────────────────────────────────────────────────


def test_parquet_store_roundtrip():
    """Data survives write → read cycle."""
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)

        # Create sample player data
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "web_name": ["Salah", "Haaland", "Palmer"],
            "total_points": [250, 280, 220],
            "team": [10, 13, 4],
        })

        # Write
        path = store.save_players(df, season="2023-24")
        assert path.exists()

        # Read
        loaded = store.load_players(season="2023-24")
        assert len(loaded) == 3
        assert loaded["web_name"].tolist() == ["Salah", "Haaland", "Palmer"]


def test_parquet_store_list_seasons():
    """Can discover available seasons."""
    from fpl_engine.storage.parquet_store import ParquetStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetStore(base_dir=tmpdir)

        df = pd.DataFrame({"id": [1], "name": ["test"]})
        store.save_gameweeks(df, season="2022-23")
        store.save_gameweeks(df, season="2023-24")

        seasons = store.list_seasons("gameweeks")
        assert "2022-23" in seasons
        assert "2023-24" in seasons


# ─── Fixture difficulty tests ────────────────────────────────────────────────


def test_team_strength_computation():
    """Team strength calculated from fixture results."""
    from fpl_engine.features.fixture_difficulty import compute_team_strength

    # Synthetic fixture data: 4 teams, some completed matches
    fixtures = pd.DataFrame({
        "team_h": [1, 2, 3, 4, 1, 2, 3, 4],
        "team_a": [2, 1, 4, 3, 3, 4, 1, 2],
        "team_h_score": [3, 1, 2, 0, 2, 1, 1, 0],
        "team_a_score": [1, 2, 0, 1, 0, 1, 2, 3],
    })

    teams = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Arsenal", "Chelsea", "Liverpool", "Spurs"],
    })

    strength = compute_team_strength(fixtures, teams)

    assert len(strength) == 4
    assert "attack_strength_home" in strength.columns
    assert "defence_strength_home" in strength.columns
    assert "overall_attack" in strength.columns
    assert "team_name" in strength.columns

    # Team 1 (Arsenal) scores well at home → higher home attack
    arsenal = strength[strength["team_id"] == 1].iloc[0]
    assert arsenal["attack_strength_home"] > 1.0  # Above league average


def test_fixture_difficulty_rating():
    """FDR computation produces 1-5 ratings."""
    from fpl_engine.features.fixture_difficulty import (
        compute_fixture_difficulty,
        compute_team_strength,
    )

    fixtures = pd.DataFrame({
        "team_h": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
        "team_a": [2, 3, 4, 1, 4, 1, 2, 3, 3, 4],
        "team_h_score": [3, 1, 2, 0, 2, 1, 1, 0, 4, 2],
        "team_a_score": [1, 2, 0, 1, 0, 1, 2, 3, 0, 1],
    })

    teams = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Arsenal", "Chelsea", "Liverpool", "Spurs"],
    })

    strength = compute_team_strength(fixtures, teams)

    # Upcoming fixtures
    upcoming = pd.DataFrame({
        "team_h": [1, 2, 3, 4],
        "team_a": [4, 3, 1, 2],
        "team_h_score": [None, None, None, None],
        "team_a_score": [None, None, None, None],
    })

    result = compute_fixture_difficulty(upcoming, strength)
    assert "fdr_home" in result.columns
    assert "fdr_away" in result.columns
    assert result["fdr_home"].between(1, 5).all()
    assert result["fdr_away"].between(1, 5).all()


# ─── FPL API client tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fpl_client_context_manager():
    """Client works as async context manager."""
    from fpl_engine.ingest.fpl_api import FPLClient

    async with FPLClient() as client:
        assert client._client is not None
    assert client._client is None


@pytest.mark.asyncio
async def test_fpl_client_requires_context():
    """Client raises error if used outside context manager."""
    from fpl_engine.ingest.fpl_api import FPLClient

    client = FPLClient()
    with pytest.raises(RuntimeError, match="async context manager"):
        _ = client.client


# ─── Historical loader tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_historical_loader_caching():
    """Historical loader caches files locally."""
    from fpl_engine.ingest.historical import HistoricalDataLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        loader = HistoricalDataLoader(cache_dir=tmpdir)

        # Pre-populate cache with CSV data
        cache_file = Path(tmpdir) / "2023-24" / "cleaned_players.csv"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("element,web_name,total_points\n1,Salah,250\n2,Haaland,280\n")

        # Should read from cache without network
        df = await loader.load_players("2023-24")
        assert len(df) == 2
        assert "Salah" in df["web_name"].values


# ─── Understat scraper tests ─────────────────────────────────────────────────


def test_understat_json_decoding():
    """Understat hex-encoded JSON decodes correctly."""
    from fpl_engine.ingest.understat import _decode_understat_json

    # Simulate Understat's encoding
    raw = '[{"player_name":"Salah","xG":"0.75"}]'
    encoded = raw.encode("unicode_escape").decode("utf-8")
    result = _decode_understat_json(encoded)
    assert result[0]["player_name"] == "Salah"


def test_understat_extract_json_var():
    """Can extract JSON variable from HTML source."""
    from fpl_engine.ingest.understat import _extract_json_var

    # Simulate page source
    data = [{"id": "1", "player_name": "Salah", "xG": "15.3"}]
    encoded = json.dumps(data).encode("unicode_escape").decode("utf-8")
    html = f"<script>var playersData = JSON.parse('{encoded}');</script>"

    result = _extract_json_var(html, "playersData")
    assert result[0]["player_name"] == "Salah"
    assert result[0]["xG"] == "15.3"
