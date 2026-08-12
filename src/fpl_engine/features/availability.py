"""Availability and injury-related features.

Converts PlayerContext information into numeric features for the model.
Handles three scenarios:

1. API data available (chance_of_playing from FPL, news text)
2. Manual overrides (user sets fields for specific players)
3. No context available (features default to "unknown" values that the model handles)

Features produced:
- chance_of_playing_pct: 0.0-1.0 (from API or manual, NaN if unknown)
- is_injured: binary flag
- is_doubtful: binary flag
- returning_from_injury: binary flag
- injury_severity: 0.0-1.0 (longer absence → higher severity)
- fitness_estimate: 0.0-1.0 (1.0 = fully fit)
- days_since_last_match: float (any competition, NaN if unknown)
- fatigue_score: combines recent minutes + days rest
- rest_before_important_match: binary flag (rested if big match in 1-3 days)
- important_match_proximity: 0.0-1.0 (closer → higher)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fpl_engine.models.player_context import AvailabilityStatus, PlayerContext

logger = logging.getLogger(__name__)


def inject_player_context(
    df: pd.DataFrame,
    contexts: dict[int, PlayerContext],
    player_id_col: str = "element",
) -> pd.DataFrame:
    """Inject PlayerContext fields as columns in the DataFrame.

    For each player in the DataFrame, looks up their context and adds
    the corresponding feature columns. Players without context get NaN
    (which the model treats as "unknown").

    Args:
        df: Player-GW DataFrame.
        contexts: Dict mapping player_id → PlayerContext.
        player_id_col: Column containing player IDs.

    Returns:
        DataFrame with context-derived feature columns added.
    """
    result = df.copy()

    # Initialize columns with NaN (unknown)
    result["chance_of_playing_pct"] = np.nan
    result["is_injured"] = 0
    result["is_doubtful"] = 0
    result["is_suspended"] = 0
    result["returning_from_injury"] = 0
    result["injury_severity"] = 0.0
    result["fitness_estimate"] = np.nan
    result["ctx_days_since_last_match"] = np.nan
    result["ctx_minutes_last_match"] = np.nan
    result["fatigue_score"] = np.nan
    result["important_match_proximity"] = 0.0
    result["rest_before_important_match"] = 0

    if not contexts:
        return result

    # Build lookup arrays for vectorized assignment
    for idx, row in result.iterrows():
        pid = row[player_id_col]
        ctx = contexts.get(pid)
        if ctx is None:
            continue

        # Chance of playing (0-100 → 0.0-1.0)
        if ctx.chance_of_playing is not None:
            result.at[idx, "chance_of_playing_pct"] = ctx.chance_of_playing / 100.0

        # Status flags
        if ctx.status == AvailabilityStatus.INJURED:
            result.at[idx, "is_injured"] = 1
        elif ctx.status == AvailabilityStatus.DOUBTFUL:
            result.at[idx, "is_doubtful"] = 1
        elif ctx.status == AvailabilityStatus.SUSPENDED:
            result.at[idx, "is_suspended"] = 1

        # Returning from injury
        if ctx.returning_from_injury:
            result.at[idx, "returning_from_injury"] = 1

        # Injury severity (based on duration)
        if ctx.injury_duration_weeks > 0:
            # Normalize: 1 week = 0.1, 10+ weeks = 1.0
            result.at[idx, "injury_severity"] = min(ctx.injury_duration_weeks / 10.0, 1.0)

        # Fitness estimate
        if ctx.fitness_level is not None:
            result.at[idx, "fitness_estimate"] = ctx.fitness_level
        elif ctx.returning_from_injury:
            # Estimate fitness from injury duration
            # Longer absence → lower fitness on return
            result.at[idx, "fitness_estimate"] = max(0.3, 1.0 - ctx.injury_duration_weeks * 0.08)

        # Days since last match (any competition)
        if ctx.days_since_last_match is not None:
            result.at[idx, "ctx_days_since_last_match"] = ctx.days_since_last_match

        # Minutes in last match (any competition)
        if ctx.played_minutes_last_match is not None:
            result.at[idx, "ctx_minutes_last_match"] = ctx.played_minutes_last_match

        # Fatigue score: high minutes + low rest = high fatigue
        if ctx.days_since_last_match is not None and ctx.played_minutes_last_match is not None:
            # Fatigue = minutes_played / days_rest (higher = more fatigued)
            days = max(ctx.days_since_last_match, 0.5)  # Avoid division by zero
            result.at[idx, "fatigue_score"] = ctx.played_minutes_last_match / (days * 90)

        # Important match proximity
        if ctx.important_match_in_days is not None:
            # Closer the important match → higher the value
            # Within 3 days = high risk of rest
            if ctx.important_match_in_days <= 7:
                result.at[idx, "important_match_proximity"] = max(
                    0, 1.0 - ctx.important_match_in_days / 7.0
                )
            if ctx.important_match_in_days <= 3:
                result.at[idx, "rest_before_important_match"] = 1

    return result


def inject_player_context_vectorized(
    df: pd.DataFrame,
    contexts: dict[int, PlayerContext],
    player_id_col: str = "element",
) -> pd.DataFrame:
    """Vectorized version of inject_player_context for large DataFrames.

    More efficient for training datasets with many rows. Uses pandas
    merge instead of row-by-row iteration.

    Args:
        df: Player-GW DataFrame.
        contexts: Dict mapping player_id → PlayerContext.
        player_id_col: Column containing player IDs.

    Returns:
        DataFrame with context-derived feature columns added.
    """
    result = df.copy()

    if not contexts:
        # Return with all-NaN context columns
        for col in AVAILABILITY_FEATURE_COLUMNS:
            result[col] = np.nan if col not in ("is_injured", "is_doubtful",
                                                 "is_suspended", "returning_from_injury",
                                                 "rest_before_important_match") else 0
        return result

    # Build context DataFrame
    ctx_records = []
    for pid, ctx in contexts.items():
        record = {
            player_id_col: pid,
            "chance_of_playing_pct": (
                ctx.chance_of_playing / 100.0 if ctx.chance_of_playing is not None else np.nan
            ),
            "is_injured": 1 if ctx.status == AvailabilityStatus.INJURED else 0,
            "is_doubtful": 1 if ctx.status == AvailabilityStatus.DOUBTFUL else 0,
            "is_suspended": 1 if ctx.status == AvailabilityStatus.SUSPENDED else 0,
            "returning_from_injury": 1 if ctx.returning_from_injury else 0,
            "injury_severity": min(ctx.injury_duration_weeks / 10.0, 1.0),
            "fitness_estimate": ctx.fitness_level if ctx.fitness_level is not None else (
                max(0.3, 1.0 - ctx.injury_duration_weeks * 0.08)
                if ctx.returning_from_injury else np.nan
            ),
            "ctx_days_since_last_match": ctx.days_since_last_match,
            "ctx_minutes_last_match": ctx.played_minutes_last_match,
            "important_match_proximity": (
                max(0, 1.0 - ctx.important_match_in_days / 7.0)
                if ctx.important_match_in_days is not None and ctx.important_match_in_days <= 7
                else 0.0
            ),
            "rest_before_important_match": (
                1 if ctx.important_match_in_days is not None and ctx.important_match_in_days <= 3
                else 0
            ),
        }

        # Fatigue score
        if ctx.days_since_last_match is not None and ctx.played_minutes_last_match is not None:
            days = max(ctx.days_since_last_match, 0.5)
            record["fatigue_score"] = ctx.played_minutes_last_match / (days * 90)
        else:
            record["fatigue_score"] = np.nan

        ctx_records.append(record)

    ctx_df = pd.DataFrame(ctx_records)

    # Merge onto main DataFrame
    result = result.merge(ctx_df, on=player_id_col, how="left")

    # Fill missing flags with 0
    flag_cols = ["is_injured", "is_doubtful", "is_suspended",
                 "returning_from_injury", "rest_before_important_match"]
    for col in flag_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    # Fill important_match_proximity with 0 (no upcoming important match)
    if "important_match_proximity" in result.columns:
        result["important_match_proximity"] = result["important_match_proximity"].fillna(0.0)

    if "injury_severity" in result.columns:
        result["injury_severity"] = result["injury_severity"].fillna(0.0)

    return result


# Feature columns produced by this module
AVAILABILITY_FEATURE_COLUMNS = [
    "chance_of_playing_pct",
    "is_injured",
    "is_doubtful",
    "is_suspended",
    "returning_from_injury",
    "injury_severity",
    "fitness_estimate",
    "ctx_days_since_last_match",
    "ctx_minutes_last_match",
    "fatigue_score",
    "important_match_proximity",
    "rest_before_important_match",
]
