"""Per-90-minutes normalization utilities.

Converts raw counting stats into per-90 rates, which are the standard
normalization in football analytics. A player with 2 goals in 180 minutes
and a player with 2 goals in 900 minutes are very different — per-90
captures this.

Also handles minutes-based derived features like "minutes share"
(what percentage of available minutes has the player played).
"""

from __future__ import annotations

import pandas as pd


def compute_per90(
    df: pd.DataFrame,
    stat_columns: list[str],
    minutes_col: str = "minutes",
    min_minutes: int = 1,
    suffix: str = "_per90",
) -> pd.DataFrame:
    """Convert counting stats to per-90 rates.

    Formula: stat_per90 = stat / minutes * 90

    Args:
        df: Input DataFrame with stat columns and a minutes column.
        stat_columns: Columns to normalize (e.g., ["goals_scored", "assists"]).
        minutes_col: Name of the minutes column.
        min_minutes: Minimum minutes to compute rate (avoids division by small numbers).
        suffix: Suffix for new column names.

    Returns:
        DataFrame with new `{col}_per90` columns. Rows with < min_minutes
        get NaN for per-90 values.
    """
    result = df.copy()
    minutes = result[minutes_col]

    for col in stat_columns:
        if col not in result.columns:
            continue
        col_name = f"{col}{suffix}"
        result[col_name] = result[col].where(minutes >= min_minutes) / minutes * 90
        # Cap at reasonable values (e.g., 90 goals per 90 minutes is impossible)
        result[col_name] = result[col_name].clip(upper=10.0)

    return result


def compute_rolling_per90(
    df: pd.DataFrame,
    stat_columns: list[str],
    windows: list[int] = (3, 5, 10),
    minutes_col: str = "minutes",
    group_col: str = "element",
    sort_col: str = "gameweek",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute rolling per-90 rates over trailing windows.

    Instead of per-90 for a single GW (noisy), this computes
    sum(stat over window) / sum(minutes over window) * 90.

    This is the standard approach in football analytics for smoothed rates.

    Args:
        df: Input DataFrame.
        stat_columns: Columns to compute per-90 for.
        windows: Window sizes.
        minutes_col: Minutes column name.
        group_col: Player ID column.
        sort_col: Time ordering column.
        min_periods: Minimum observations.

    Returns:
        DataFrame with `{col}_per90_roll{window}` columns.
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    for col in stat_columns:
        if col not in result.columns:
            continue
        for window in windows:
            # Rolling sum of stat (shifted to exclude current)
            stat_sum = result.groupby(group_col)[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).sum()
            )
            # Rolling sum of minutes (shifted)
            mins_sum = result.groupby(group_col)[minutes_col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).sum()
            )

            col_name = f"{col}_per90_roll{window}"
            result[col_name] = (stat_sum / mins_sum * 90).where(mins_sum > 0)

    return result


def compute_minutes_share(
    df: pd.DataFrame,
    minutes_col: str = "minutes",
    max_minutes: int = 90,
) -> pd.DataFrame:
    """Compute minutes share (what fraction of the match the player played).

    Args:
        df: Input DataFrame.
        minutes_col: Minutes column.
        max_minutes: Maximum possible minutes (usually 90).

    Returns:
        DataFrame with `minutes_share` column (0.0 to 1.0).
    """
    result = df.copy()
    result["minutes_share"] = (result[minutes_col] / max_minutes).clip(0.0, 1.0)
    return result


def compute_availability_rate(
    df: pd.DataFrame,
    windows: list[int] = (5, 10, 38),
    minutes_col: str = "minutes",
    group_col: str = "element",
    sort_col: str = "gameweek",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute what fraction of recent GWs the player was available (minutes > 0).

    This is a key feature for minutes prediction — a player who has played
    in 9 of the last 10 GWs is much more likely to play than one who
    played in 3 of the last 10.

    Returns:
        DataFrame with `availability_rate{window}` columns (0.0 to 1.0).
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    for window in windows:
        col_name = f"availability_rate{window}"
        result[col_name] = result.groupby(group_col)[minutes_col].transform(
            lambda x: (x > 0).astype(float).shift(1).rolling(
                window=window, min_periods=min_periods
            ).mean()
        )

    return result
