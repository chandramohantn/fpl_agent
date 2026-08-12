"""Match difficulty and squad depth features.

Captures signals that directly influence minutes decisions:

1. Match difficulty (current, previous, next)
   - How hard is this fixture? Managers play their best XI in hard matches
     but may rotate in easy ones.
   - If the previous match was hard, players may be rested.
   - If the next match is hard, players may be saved.

2. Squad depth / position competition
   - How many viable competitors does a player have at their position?
   - If squad depth is high, rotation is more likely.
   - A player's "minutes share" within their position group indicates
     their standing in the pecking order.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Match Difficulty Features ───────────────────────────────────────────────


def add_match_difficulty(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    opponent_col: str = "opponent_team",
    home_col: str = "was_home",
) -> pd.DataFrame:
    """Add match difficulty rating for the current fixture.

    Uses opponent's overall strength (attack + defence) relative to league
    average to produce a single difficulty score. Higher = harder fixture.

    Also adds difficulty band (1-5) matching FPL's official FDR scale.

    Output columns:
        - match_difficulty: continuous score (centered around 1.0)
        - match_difficulty_band: 1 (easiest) to 5 (hardest)
    """
    from fpl_engine.features.fixture_difficulty import compute_team_strength

    result = df.copy()

    strength = compute_team_strength(fixtures_df, teams_df)
    if strength.empty:
        result["match_difficulty"] = np.nan
        result["match_difficulty_band"] = np.nan
        return result

    # Difficulty = opponent's overall attack + defence (both contribute to match difficulty)
    # Higher attack = more threatening, higher defence = harder to score against
    strength["overall_difficulty"] = (
        strength["overall_attack"] + strength["overall_defence"]
    ) / 2

    diff_map = strength.set_index("team_id")["overall_difficulty"]
    result["match_difficulty"] = result[opponent_col].map(diff_map)

    # Band: quantile-based 1-5
    result["match_difficulty_band"] = pd.qcut(
        result["match_difficulty"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(float)

    return result


def add_surrounding_difficulty(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    opponent_col: str = "opponent_team",
    group_col: str = "element",
    sort_col: str = "gameweek",
) -> pd.DataFrame:
    """Add difficulty of previous and next fixtures.

    These capture the "sandwich" effect:
    - Hard match before → player may be tired/rested
    - Hard match after → player may be saved for it

    Output columns:
        - prev_match_difficulty: difficulty of the previous fixture
        - next_match_difficulty: difficulty of the upcoming fixture
        - difficulty_change: current - previous (positive = harder now)
        - sandwich_score: max(prev, next) - current difficulty.
                         High value = current match is sandwiched between hard ones.
    """
    from fpl_engine.features.fixture_difficulty import compute_team_strength

    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    strength = compute_team_strength(fixtures_df, teams_df)
    if strength.empty:
        result["prev_match_difficulty"] = np.nan
        result["next_match_difficulty"] = np.nan
        result["difficulty_change"] = np.nan
        result["sandwich_score"] = np.nan
        return result

    strength["overall_difficulty"] = (
        strength["overall_attack"] + strength["overall_defence"]
    ) / 2
    diff_map = strength.set_index("team_id")["overall_difficulty"]

    # Current match difficulty
    result["_curr_diff"] = result[opponent_col].map(diff_map)

    # Previous match difficulty (shift forward = look at previous row)
    result["prev_match_difficulty"] = result.groupby(group_col)["_curr_diff"].shift(1)

    # Next match difficulty (shift backward = look at next row)
    result["next_match_difficulty"] = result.groupby(group_col)["_curr_diff"].shift(-1)

    # Derived
    result["difficulty_change"] = result["_curr_diff"] - result["prev_match_difficulty"]
    result["sandwich_score"] = (
        np.maximum(
            result["prev_match_difficulty"].fillna(0),
            result["next_match_difficulty"].fillna(0),
        )
        - result["_curr_diff"]
    )

    result = result.drop(columns=["_curr_diff"])
    return result


# ─── Squad Depth Features ────────────────────────────────────────────────────


def add_squad_depth(
    df: pd.DataFrame,
    group_col: str = "element",
    team_col: str = "team",
    position_col: str = "position",
    gameweek_col: str = "gameweek",
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    """Add squad depth and position competition features.

    For each player-GW, computes:
    - How many players in the same team+position played that GW
    - How many total players in the same team+position are in the squad
    - Player's share of available minutes at their position within the team

    Output columns:
        - squad_depth_position: count of players in same team+position in squad
        - competitors_played: how many at same team+position got minutes this GW
        - position_minutes_share: player's minutes / total position minutes for team
        - is_primary_choice: whether player got the most minutes at their position
    """
    result = df.copy()

    # Squad depth: total players registered at this team+position
    squad_size = result.groupby([team_col, position_col, gameweek_col])[group_col].transform(
        "nunique"
    )
    result["squad_depth_position"] = squad_size

    # Competitors who played: players at same team+position who got minutes
    played_flag = (result[minutes_col] > 0).astype(int)
    competitors_played = result.assign(_played=played_flag).groupby(
        [team_col, position_col, gameweek_col]
    )["_played"].transform("sum")
    result["competitors_played"] = competitors_played

    # Position minutes share: this player's minutes / total team position minutes
    total_pos_minutes = result.groupby(
        [team_col, position_col, gameweek_col]
    )[minutes_col].transform("sum")
    result["position_minutes_share"] = np.where(
        total_pos_minutes > 0,
        result[minutes_col] / total_pos_minutes,
        0.0,
    )

    # Is primary choice: did this player get the most minutes at their position?
    max_pos_minutes = result.groupby(
        [team_col, position_col, gameweek_col]
    )[minutes_col].transform("max")
    result["is_primary_choice"] = (
        (result[minutes_col] == max_pos_minutes) & (result[minutes_col] > 0)
    ).astype(int)

    return result


def add_rolling_squad_depth(
    df: pd.DataFrame,
    windows: list[int] = (3, 5),
    group_col: str = "element",
    sort_col: str = "gameweek",
) -> pd.DataFrame:
    """Add rolling position competition metrics.

    Instead of just current-GW squad depth (which includes the target),
    uses trailing metrics to avoid leakage:
    - Rolling average of competitors_played (how competitive is this position?)
    - Rolling position_minutes_share (player's recent pecking order standing)

    Output columns:
        - competitors_played_roll{window}
        - position_minutes_share_roll{window}
        - is_primary_choice_roll{window}
    """
    from fpl_engine.features.rolling import rolling_mean

    result = df.copy()

    # Compute rolling averages of squad depth metrics (with shift to avoid leakage)
    cols_to_roll = ["competitors_played", "position_minutes_share", "is_primary_choice"]
    available_cols = [c for c in cols_to_roll if c in result.columns]

    if available_cols:
        result = rolling_mean(
            result,
            columns=available_cols,
            windows=windows,
            group_col=group_col,
            sort_col=sort_col,
        )

    return result


# ─── Feature column names produced by this module ────────────────────────────

MATCH_DIFFICULTY_COLUMNS = [
    "match_difficulty",
    "match_difficulty_band",
    "prev_match_difficulty",
    "next_match_difficulty",
    "difficulty_change",
    "sandwich_score",
]

SQUAD_DEPTH_COLUMNS = [
    "squad_depth_position",
    "competitors_played",
    "position_minutes_share",
    "is_primary_choice",
    "competitors_played_roll3",
    "competitors_played_roll5",
    "position_minutes_share_roll3",
    "position_minutes_share_roll5",
    "is_primary_choice_roll3",
    "is_primary_choice_roll5",
]
