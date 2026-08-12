"""Async client for the official Fantasy Premier League API.

Endpoints:
- bootstrap-static: All players, teams, gameweeks, game settings
- element-summary/{id}: Detailed player history and upcoming fixtures
- fixtures: All fixtures for the season
- event/{gw}/live: Live gameweek data (points as they happen)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"


class FPLClient:
    """Async HTTP client for the FPL API.

    Usage:
        async with FPLClient() as client:
            bootstrap = await client.get_bootstrap()
            players = bootstrap["elements"]
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> FPLClient:
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=self._timeout,
            headers={
                "User-Agent": "fpl-engine/0.1.0 (github.com/fpl-engine)",
            },
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("FPLClient must be used as async context manager")
        return self._client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _get(self, path: str) -> Any:
        """GET request with retry logic."""
        response = await self.client.get(path)
        response.raise_for_status()
        return response.json()

    # ─── Bootstrap (master data) ─────────────────────────────────────────

    async def get_bootstrap(self) -> dict[str, Any]:
        """Fetch bootstrap-static — the master dataset.

        Returns dict with keys:
            elements: list of all players
            teams: list of all teams
            events: list of all gameweeks
            element_types: position metadata
            game_settings: game configuration
        """
        logger.info("Fetching bootstrap-static")
        return await self._get("/bootstrap-static/")

    async def get_players(self) -> list[dict[str, Any]]:
        """All players (elements) from bootstrap."""
        data = await self.get_bootstrap()
        return data["elements"]

    async def get_teams(self) -> list[dict[str, Any]]:
        """All teams from bootstrap."""
        data = await self.get_bootstrap()
        return data["teams"]

    async def get_gameweeks(self) -> list[dict[str, Any]]:
        """All gameweeks (events) from bootstrap."""
        data = await self.get_bootstrap()
        return data["events"]

    # ─── Fixtures ────────────────────────────────────────────────────────

    async def get_fixtures(self, gameweek: int | None = None) -> list[dict[str, Any]]:
        """All fixtures, optionally filtered by gameweek."""
        logger.info("Fetching fixtures (gw=%s)", gameweek)
        params = {}
        if gameweek is not None:
            params["event"] = gameweek
        response = await self.client.get("/fixtures/", params=params)
        response.raise_for_status()
        return response.json()

    # ─── Player detail ───────────────────────────────────────────────────

    async def get_player_summary(self, player_id: int) -> dict[str, Any]:
        """Detailed player data — history + upcoming fixtures.

        Returns dict with keys:
            history: list of past GW performances
            fixtures: list of upcoming fixtures
            history_past: list of past season summaries
        """
        logger.debug("Fetching player summary for id=%d", player_id)
        return await self._get(f"/element-summary/{player_id}/")

    async def get_player_history(self, player_id: int) -> list[dict[str, Any]]:
        """Past gameweek performances for a player this season."""
        data = await self.get_player_summary(player_id)
        return data["history"]

    async def get_player_upcoming(self, player_id: int) -> list[dict[str, Any]]:
        """Upcoming fixtures for a player."""
        data = await self.get_player_summary(player_id)
        return data["fixtures"]

    # ─── Live Gameweek ───────────────────────────────────────────────────

    async def get_live_gameweek(self, gameweek: int) -> dict[str, Any]:
        """Live data for a specific gameweek (points, stats, bonus)."""
        logger.info("Fetching live data for GW%d", gameweek)
        return await self._get(f"/event/{gameweek}/live/")

    # ─── Convenience ─────────────────────────────────────────────────────

    async def get_current_gameweek(self) -> int | None:
        """Get the current gameweek number, or None if season hasn't started."""
        data = await self.get_bootstrap()
        for event in data["events"]:
            if event.get("is_current"):
                return event["id"]
        return None
