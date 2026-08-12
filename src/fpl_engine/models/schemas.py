"""Pydantic schemas for FPL domain entities."""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum

from pydantic import BaseModel


class Position(IntEnum):
    """FPL position codes."""

    GKP = 1
    DEF = 2
    MID = 3
    FWD = 4


class Team(BaseModel):
    """Premier League team."""

    id: int
    name: str
    short_name: str
    strength: int
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int


class Player(BaseModel):
    """FPL player summary (from bootstrap-static)."""

    id: int
    web_name: str
    first_name: str
    second_name: str
    team: int
    element_type: int  # Position enum value
    now_cost: int  # Price × 10
    total_points: int
    points_per_game: float
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    saves: int
    bonus: int
    bps: int
    form: float
    selected_by_percent: float
    transfers_in_event: int
    transfers_out_event: int
    # xG stats (available from 2023-24 onwards)
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    expected_goal_involvements: float = 0.0
    expected_goals_conceded: float = 0.0
    # Status
    status: str  # a=available, d=doubtful, i=injured, s=suspended, u=unavailable
    chance_of_playing_next_round: int | None = None
    news: str = ""


class Fixture(BaseModel):
    """A Premier League fixture."""

    id: int
    event: int | None  # Gameweek number (None if unscheduled)
    team_h: int
    team_a: int
    team_h_difficulty: int
    team_a_difficulty: int
    team_h_score: int | None = None
    team_a_score: int | None = None
    started: bool = False
    finished: bool = False
    kickoff_time: datetime | None = None


class GameweekHistory(BaseModel):
    """A single player's performance in one Gameweek (element-summary)."""

    element: int  # Player ID
    fixture: int
    round: int  # Gameweek
    opponent_team: int
    was_home: bool
    kickoff_time: datetime | None = None
    total_points: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    saves: int
    bonus: int
    bps: int
    yellow_cards: int
    red_cards: int
    penalties_saved: int
    penalties_missed: int
    own_goals: int
    # xG (available 2023-24+)
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    expected_goal_involvements: float = 0.0
    expected_goals_conceded: float = 0.0
    # Value/transfers
    value: int = 0  # Price at the time × 10
    selected: int = 0
    transfers_in: int = 0
    transfers_out: int = 0


class Gameweek(BaseModel):
    """Gameweek metadata."""

    id: int
    name: str
    deadline_time: datetime
    is_current: bool = False
    is_next: bool = False
    finished: bool = False
    highest_score: int | None = None
    average_score: int | None = None
    most_captained: int | None = None
    most_vice_captained: int | None = None


class UnderstatPlayer(BaseModel):
    """Player-level xG data from Understat."""

    player_name: str
    team: str
    games: int = 0
    minutes: int = 0
    goals: int = 0
    assists: int = 0
    xg: float = 0.0
    xa: float = 0.0
    npxg: float = 0.0  # Non-penalty xG
    xg_chain: float = 0.0
    xg_buildup: float = 0.0
    shots: int = 0
    key_passes: int = 0


class UnderstatShot(BaseModel):
    """Individual shot-level data from Understat."""

    id: int
    minute: int
    x: float  # Pitch coordinates (0-1)
    y: float
    xg: float
    player: str
    team: str
    result: str  # Goal, SavedShot, MissedShots, BlockedShot, etc.
    situation: str  # OpenPlay, FromCorner, SetPiece, DirectFreekick, Penalty
    season: str
    match_id: int
    player_id: int = 0


class SeasonMetadata(BaseModel):
    """Metadata about an FPL season."""

    season: str  # e.g. "2024-25"
    start_year: int
    total_gameweeks: int = 38
    current_gameweek: int | None = None
    is_active: bool = False
