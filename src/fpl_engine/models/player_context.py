"""Player context for external information injection.

This module defines the PlayerContext structure — a per-player, per-gameweek
container for information that cannot be derived from historical GW data alone.

Sources for this information:
- FPL API: chance_of_playing_next_round, news, status
- External APIs: fixture lists (Champions League, cups), injury reports
- Manual input: user overrides for specific players

The system is designed so that:
1. If API data is available, it auto-populates the context
2. Any field can be manually overridden by the user
3. Missing fields default to "unknown" (model handles gracefully)

Usage:
    # From API
    contexts = PlayerContext.from_fpl_bootstrap(bootstrap_data)

    # Manual override
    contexts[salah_id] = PlayerContext(
        player_id=salah_id,
        chance_of_playing=75,
        days_since_last_match=3,
        returning_from_injury=True,
        injury_duration_weeks=4,
        important_match_in_days=3,
        important_match_type="Champions League Semi-Final",
    )

    # Build features from contexts
    df = inject_player_context(df, contexts)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AvailabilityStatus(str, Enum):
    """Player availability status."""

    AVAILABLE = "available"  # Fully fit, expected to play
    DOUBTFUL = "doubtful"  # 25-75% chance of playing
    INJURED = "injured"  # Currently injured, not expected
    SUSPENDED = "suspended"  # Suspended (cards/ban)
    UNAVAILABLE = "unavailable"  # Other reasons (personal, international duty)
    UNKNOWN = "unknown"  # No information available


@dataclass
class PlayerContext:
    """External context for a player in a specific gameweek.

    All fields are optional — missing values indicate "no information available"
    and the model will use historical patterns as the prior.

    Fields:
        player_id: FPL element ID for the player.
        gameweek: Target gameweek for the prediction.

        # Availability / Injury
        chance_of_playing: 0-100 probability of featuring (from FPL API or manual).
                          None = unknown (use historical availability rate).
        status: Categorical availability status.
        news: Free-text injury/status news (e.g., "Hamstring - Expected back GW12").
        returning_from_injury: Whether the player is just coming back from injury.
                              These players often get managed minutes initially.
        injury_duration_weeks: How long the player was out (longer = more managed return).
        fitness_level: 0.0 to 1.0 estimate of match fitness (1.0 = fully fit).
                      Players returning from long injuries start lower.

        # External match context
        days_since_last_match: Days since last competitive match (any competition).
                             Captures midweek European games, cup matches, internationals.
                             Shorter rest = higher rotation risk.
        played_minutes_last_match: Minutes played in the most recent match (any competition).
                                  90 mins 3 days ago = high fatigue.
        important_match_in_days: Days until the next important match (CL, cup final, derby).
                               Managers may rest players before big games.
        important_match_type: Description of the upcoming match (for logging/display).

        # Source tracking
        source: Where this context came from ("api", "manual", "scraper").
    """

    player_id: int
    gameweek: int | None = None

    # Availability / Injury
    chance_of_playing: int | None = None  # 0-100
    status: AvailabilityStatus = AvailabilityStatus.UNKNOWN
    news: str = ""
    returning_from_injury: bool = False
    injury_duration_weeks: float = 0.0
    fitness_level: float | None = None  # 0.0-1.0, None=unknown (assumed fit)

    # External match context
    days_since_last_match: float | None = None  # Any competition
    played_minutes_last_match: int | None = None  # Minutes in last match (any comp)
    important_match_in_days: float | None = None  # Days until next big match
    important_match_type: str = ""  # e.g., "Champions League QF"

    # Source
    source: str = "manual"

    @classmethod
    def from_fpl_player(cls, player_data: dict[str, Any], gameweek: int | None = None):
        """Create PlayerContext from FPL API bootstrap player data.

        Maps FPL API fields to our context structure:
        - chance_of_playing_next_round → chance_of_playing
        - status (a/d/i/s/u) → status enum
        - news → news
        """
        status_map = {
            "a": AvailabilityStatus.AVAILABLE,
            "d": AvailabilityStatus.DOUBTFUL,
            "i": AvailabilityStatus.INJURED,
            "s": AvailabilityStatus.SUSPENDED,
            "u": AvailabilityStatus.UNAVAILABLE,
        }

        chance = player_data.get("chance_of_playing_next_round")
        if chance is not None:
            chance = int(chance)

        return cls(
            player_id=player_data["id"],
            gameweek=gameweek,
            chance_of_playing=chance,
            status=status_map.get(player_data.get("status", ""), AvailabilityStatus.UNKNOWN),
            news=player_data.get("news", ""),
            source="api",
        )

    @classmethod
    def from_fpl_bootstrap(
        cls, bootstrap: dict[str, Any], gameweek: int | None = None
    ) -> dict[int, PlayerContext]:
        """Create contexts for all players from FPL bootstrap data.

        Returns dict mapping player_id → PlayerContext.
        """
        contexts = {}
        for player in bootstrap.get("elements", []):
            ctx = cls.from_fpl_player(player, gameweek=gameweek)
            contexts[ctx.player_id] = ctx
        return contexts

    def override(self, **kwargs) -> PlayerContext:
        """Create a new context with specific fields overridden.

        Usage:
            ctx = ctx.override(chance_of_playing=50, returning_from_injury=True)
        """
        data = {
            "player_id": self.player_id,
            "gameweek": self.gameweek,
            "chance_of_playing": self.chance_of_playing,
            "status": self.status,
            "news": self.news,
            "returning_from_injury": self.returning_from_injury,
            "injury_duration_weeks": self.injury_duration_weeks,
            "fitness_level": self.fitness_level,
            "days_since_last_match": self.days_since_last_match,
            "played_minutes_last_match": self.played_minutes_last_match,
            "important_match_in_days": self.important_match_in_days,
            "important_match_type": self.important_match_type,
            "source": "manual",  # Override marks source as manual
        }
        data.update(kwargs)
        return PlayerContext(**data)
