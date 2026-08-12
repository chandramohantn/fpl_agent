"""Opponent strength adjustment features.

Adjusts raw stats by the quality of opposition faced. A clean sheet against
Man City is more impressive than one against the bottom team. Similarly,
minutes prediction depends on opponent — managers may rotate against weaker
teams (using subs earlier) or play their strongest XI against top teams.
"""

from __future__ import annotations

import pandas as pd


def add_opponent_strength(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    opponent_col: str = "opponent_team",
) -> pd.DataFrame:
    """Add opponent's attack and defence strength to each row.

    Merges team strength ratings from the fixtures/teams data onto
    each player's GW record based on their opponent.

    Args:
        df: Player GW data with opponent_team column.
        fixtures_df: Completed fixtures (for computing strength).
        teams_df: Team metadata (with id, name).
        opponent_col: Column containing opponent team ID.

    Returns:
        DataFrame with added columns:
            - opp_attack_strength: Opponent's overall attacking strength
            - opp_defence_strength: Opponent's overall defensive strength
    """
    from fpl_engine.features.fixture_difficulty import compute_team_strength

    # Compute strengths from fixtures
    strength = compute_team_strength(fixtures_df, teams_df)

    if strength.empty:
        # No completed fixtures — return with NaN columns
        result = df.copy()
        result["opp_attack_strength"] = float("nan")
        result["opp_defence_strength"] = float("nan")
        return result

    # Build lookup maps
    attack_map = strength.set_index("team_id")["overall_attack"]
    defence_map = strength.set_index("team_id")["overall_defence"]

    result = df.copy()
    result["opp_attack_strength"] = result[opponent_col].map(attack_map)
    result["opp_defence_strength"] = result[opponent_col].map(defence_map)

    return result


def add_team_strength(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    team_col: str = "team",
) -> pd.DataFrame:
    """Add the player's own team attack and defence strength.

    This helps the model understand: a striker at a top team
    has more opportunities than one at a relegation team.

    Args:
        df: Player GW data with a team column (name or ID).
        fixtures_df: Completed fixtures.
        teams_df: Team metadata.
        team_col: Column containing the player's team.

    Returns:
        DataFrame with `team_attack_strength` and `team_defence_strength`.
    """
    from fpl_engine.features.fixture_difficulty import compute_team_strength

    strength = compute_team_strength(fixtures_df, teams_df)

    if strength.empty:
        result = df.copy()
        result["team_attack_strength"] = float("nan")
        result["team_defence_strength"] = float("nan")
        return result

    result = df.copy()

    # Handle team column being either ID or name
    if result[team_col].dtype in ("int64", "int32", "float64"):
        # Team column is ID-based
        attack_map = strength.set_index("team_id")["overall_attack"]
        defence_map = strength.set_index("team_id")["overall_defence"]
        result["team_attack_strength"] = result[team_col].map(attack_map)
        result["team_defence_strength"] = result[team_col].map(defence_map)
    else:
        # Team column is name-based
        attack_map = strength.set_index("team_name")["overall_attack"]
        defence_map = strength.set_index("team_name")["overall_defence"]
        result["team_attack_strength"] = result[team_col].map(attack_map)
        result["team_defence_strength"] = result[team_col].map(defence_map)

    return result


def compute_opponent_adjusted(
    df: pd.DataFrame,
    stat_columns: list[str],
    opp_strength_col: str = "opp_defence_strength",
) -> pd.DataFrame:
    """Adjust stats by opponent strength.

    For attacking stats (goals, xG), divide by opponent's defensive
    strength to normalize. A goal against a team with defence_strength=1.5
    (50% above average) is less indicative than one against 0.7 (30% below avg).

    Formula: adjusted_stat = raw_stat / opponent_strength

    This means higher values = better after adjustment.

    Args:
        df: DataFrame with stat columns and opponent strength column.
        stat_columns: Columns to adjust.
        opp_strength_col: Which opponent strength to divide by.

    Returns:
        DataFrame with `{col}_opp_adj` columns.
    """
    result = df.copy()

    for col in stat_columns:
        if col not in result.columns or opp_strength_col not in result.columns:
            continue
        col_name = f"{col}_opp_adj"
        result[col_name] = result[col] / result[opp_strength_col].replace(0, float("nan"))

    return result
