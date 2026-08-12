"""Fixture context features.

Captures situational factors that affect player minutes and performance:
- Days rest between matches
- Fixture congestion (matches in a rolling window)
- Home/away indicator
- Season progression (early, mid, late)
- Double gameweek indicator
"""

from __future__ import annotations

import pandas as pd


def add_home_away(df: pd.DataFrame, was_home_col: str = "was_home") -> pd.DataFrame:
    """Add a numeric home/away indicator.

    Converts boolean was_home to integer (1=home, 0=away).
    """
    result = df.copy()
    result["is_home"] = result[was_home_col].astype(int)
    return result


def add_days_rest(
    df: pd.DataFrame,
    kickoff_col: str = "kickoff_time",
    group_col: str = "element",
    sort_col: str = "gameweek",
) -> pd.DataFrame:
    """Compute days since last match for each player.

    Players with more rest may be more likely to start. Players in
    congested periods (3 days rest) face rotation risk.

    Returns:
        DataFrame with `days_rest` column. First appearance gets NaN.
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    # Parse kickoff time if string
    if result[kickoff_col].dtype == object:
        result["_kickoff_dt"] = pd.to_datetime(result[kickoff_col], utc=True)
    else:
        result["_kickoff_dt"] = result[kickoff_col]

    # Days since last GW for each player
    result["days_rest"] = result.groupby(group_col)["_kickoff_dt"].transform(
        lambda x: x.diff().dt.total_seconds() / 86400
    )

    # Cap at reasonable values (first GW of season → NaN, which is fine)
    result["days_rest"] = result["days_rest"].clip(0, 30)

    result = result.drop(columns=["_kickoff_dt"])
    return result


def add_fixture_congestion(
    df: pd.DataFrame,
    windows: list[int] = (3, 5),
    group_col: str = "element",
    sort_col: str = "gameweek",
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    """Count how many matches a player played in the last N gameweeks.

    High congestion (3+ matches in 5 GWs with >60 mins) increases rotation risk.

    Returns:
        DataFrame with `matches_played_last{window}` columns.
    """
    result = df.copy()
    result = result.sort_values([group_col, sort_col])

    for window in windows:
        col_name = f"matches_played_last{window}"
        result[col_name] = result.groupby(group_col)[minutes_col].transform(
            lambda x: (x > 0).astype(int).shift(1).rolling(
                window=window, min_periods=1
            ).sum()
        )

    return result


def add_season_progress(
    df: pd.DataFrame,
    gameweek_col: str = "gameweek",
    total_gws: int = 38,
) -> pd.DataFrame:
    """Add season progress indicator (0.0 = start, 1.0 = end).

    Captures effects like: managers rotate more in Dec/Jan congestion,
    dead rubbers at end of season, etc.
    """
    result = df.copy()
    result["season_progress"] = result[gameweek_col] / total_gws
    return result


def add_double_gameweek(
    df: pd.DataFrame,
    group_col: str = "element",
    gameweek_col: str = "gameweek",
) -> pd.DataFrame:
    """Flag if a player has multiple fixtures in the same gameweek (DGW).

    In Double Gameweeks, players may play twice, affecting expected minutes.
    This counts appearances per player per GW.
    """
    result = df.copy()
    appearances = result.groupby([group_col, gameweek_col]).transform("size")
    result["is_dgw"] = (appearances > 1).astype(int)
    return result
