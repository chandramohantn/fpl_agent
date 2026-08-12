"""Rolling window aggregation utilities.

Compute trailing statistics over the last N gameweeks for any player metric.
Used by prediction models to capture recent form.

All functions operate on a DataFrame sorted by (element, gameweek) and return
new columns with rolling aggregations. They never look into the future — only
past data relative to each row.
"""

from __future__ import annotations

import pandas as pd


def rolling_mean(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = (3, 5, 10),
    group_col: str = "element",
    sort_col: str = "gameweek",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute rolling mean of specified columns over past N gameweeks.

    For each player (group_col), computes a trailing mean of each column
    over the last `window` gameweeks. The current row is EXCLUDED (shift=1)
    to prevent data leakage.

    Args:
        df: Input DataFrame, must contain group_col and sort_col.
        columns: Stat columns to aggregate (e.g., ["minutes", "goals_scored"]).
        windows: Window sizes to compute (e.g., [3, 5, 10]).
        group_col: Column to group by (default: "element" = player ID).
        sort_col: Column to sort by within groups (default: "gameweek").
        min_periods: Minimum observations required for a valid result.

    Returns:
        DataFrame with new columns named `{col}_roll{window}` for each
        combination of column and window.
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    for col in columns:
        if col not in result.columns:
            continue
        grouped = result.groupby(group_col)[col]
        for window in windows:
            col_name = f"{col}_roll{window}"
            # shift(1) ensures we only use past data (no leakage)
            result[col_name] = grouped.transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean()
            )

    return result


def rolling_sum(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = (3, 5, 10),
    group_col: str = "element",
    sort_col: str = "gameweek",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute rolling sum of specified columns over past N gameweeks.

    Same semantics as rolling_mean but sums instead of averaging.
    Useful for counting events (goals, assists, starts).
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    for col in columns:
        if col not in result.columns:
            continue
        grouped = result.groupby(group_col)[col]
        for window in windows:
            col_name = f"{col}_sum{window}"
            result[col_name] = grouped.transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).sum()
            )

    return result


def rolling_std(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] = (5, 10),
    group_col: str = "element",
    sort_col: str = "gameweek",
    min_periods: int = 2,
) -> pd.DataFrame:
    """Compute rolling standard deviation (volatility) over past N gameweeks.

    Useful for capturing consistency vs. volatility of a player's output.
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    for col in columns:
        if col not in result.columns:
            continue
        grouped = result.groupby(group_col)[col]
        for window in windows:
            col_name = f"{col}_std{window}"
            result[col_name] = grouped.transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).std()
            )

    return result


def rolling_pct(
    df: pd.DataFrame,
    column: str,
    condition_fn,
    windows: list[int] = (3, 5, 10),
    group_col: str = "element",
    sort_col: str = "gameweek",
    min_periods: int = 1,
    output_name: str | None = None,
) -> pd.DataFrame:
    """Compute rolling percentage where a condition is true.

    Example: percentage of last 5 GWs where the player started (minutes >= 60).

    Args:
        df: Input DataFrame.
        column: Column to evaluate.
        condition_fn: Function applied to the column to produce a boolean.
                     e.g., lambda x: x >= 60
        windows: Window sizes.
        output_name: Base name for output columns. Defaults to column name.

    Returns:
        DataFrame with `{output_name}_pct{window}` columns.
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    name = output_name or column
    # Create boolean column
    bool_col = f"__{name}_bool"
    result[bool_col] = condition_fn(result[column]).astype(float)

    grouped = result.groupby(group_col)[bool_col]
    for window in windows:
        col_name = f"{name}_pct{window}"
        result[col_name] = grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=min_periods).mean()
        )

    result = result.drop(columns=[bool_col])
    return result


def cumulative_mean(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str = "element",
    sort_col: str = "gameweek",
) -> pd.DataFrame:
    """Compute expanding (season-to-date) mean for each player.

    Unlike rolling windows, this uses all past observations.
    Current row is excluded (shift=1).
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    for col in columns:
        if col not in result.columns:
            continue
        col_name = f"{col}_season_avg"
        result[col_name] = result.groupby(group_col)[col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

    return result
