"""Fixture difficulty and team strength calculations.

Computes:
- Attack strength (goals scored relative to league average)
- Defence strength (goals conceded relative to league average)
- Home/away splits
- Fixture Difficulty Rating (FDR) from team strengths
- Rolling form-based strength
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_team_strength(
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute attack and defence strength for each team.

    Uses completed fixtures to calculate goals scored/conceded
    relative to the league average, split by home/away.

    Args:
        fixtures_df: Fixtures with team_h, team_a, team_h_score, team_a_score columns.
                    Must have finished=True filter applied or scores present.
        teams_df: Teams with at least id and name columns.

    Returns:
        DataFrame with columns:
            team_id, team_name,
            attack_strength_home, attack_strength_away,
            defence_strength_home, defence_strength_away,
            overall_attack, overall_defence
    """
    # Filter to completed fixtures only
    completed = fixtures_df.dropna(subset=["team_h_score", "team_a_score"]).copy()
    if completed.empty:
        logger.warning("No completed fixtures — returning empty strength table")
        return pd.DataFrame()

    completed["team_h_score"] = completed["team_h_score"].astype(int)
    completed["team_a_score"] = completed["team_a_score"].astype(int)

    # League averages
    avg_home_goals = completed["team_h_score"].mean()
    avg_away_goals = completed["team_a_score"].mean()

    # Per-team home attack: goals scored at home / league average home goals
    home_attack = (
        completed.groupby("team_h")["team_h_score"]
        .mean()
        .rename("attack_strength_home")
    )
    home_attack = home_attack / avg_home_goals

    # Per-team away attack: goals scored away / league average away goals
    away_attack = (
        completed.groupby("team_a")["team_a_score"]
        .mean()
        .rename("attack_strength_away")
    )
    away_attack = away_attack / avg_away_goals

    # Per-team home defence: goals conceded at home / league average away goals
    # (opponent scores = away goals against this team at home)
    home_defence = (
        completed.groupby("team_h")["team_a_score"]
        .mean()
        .rename("defence_strength_home")
    )
    home_defence = home_defence / avg_away_goals

    # Per-team away defence: goals conceded away / league average home goals
    away_defence = (
        completed.groupby("team_a")["team_h_score"]
        .mean()
        .rename("defence_strength_away")
    )
    away_defence = away_defence / avg_home_goals

    # Combine
    strength = pd.DataFrame({
        "attack_strength_home": home_attack,
        "attack_strength_away": away_attack,
        "defence_strength_home": home_defence,
        "defence_strength_away": away_defence,
    })
    strength.index.name = "team_id"
    strength = strength.reset_index()

    # Overall (weighted average of home + away)
    strength["overall_attack"] = (
        strength["attack_strength_home"] + strength["attack_strength_away"]
    ) / 2
    strength["overall_defence"] = (
        strength["defence_strength_home"] + strength["defence_strength_away"]
    ) / 2

    # Merge team names
    team_names = teams_df[["id", "name"]].rename(columns={"id": "team_id", "name": "team_name"})
    strength = strength.merge(team_names, on="team_id", how="left")

    return strength


def compute_fixture_difficulty(
    fixtures_df: pd.DataFrame,
    team_strength_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute FDR (Fixture Difficulty Rating) based on opponent strength.

    For each fixture, the difficulty for the home team is based on the
    away team's attack strength (harder opponents = higher FDR).
    Vice versa for the away team.

    FDR is scaled 1-5 to match the official FPL scale.

    Args:
        fixtures_df: All fixtures (upcoming or completed).
        team_strength_df: Output from compute_team_strength().

    Returns:
        Copy of fixtures_df with added columns:
            fdr_home: Difficulty for the home team (1-5)
            fdr_away: Difficulty for the away team (1-5)
    """
    result = fixtures_df.copy()

    # For home team: difficulty = opponent's away attack + own home defence weakness
    # Simplified: difficulty ∝ opponent's overall_attack
    attack_map = team_strength_df.set_index("team_id")["overall_attack"]
    defence_map = team_strength_df.set_index("team_id")["overall_defence"]

    # Home team faces away team's attack
    result["opponent_attack_for_home"] = result["team_a"].map(attack_map)
    result["opponent_attack_for_away"] = result["team_h"].map(attack_map)

    # Also consider own defensive weakness
    result["own_defence_home"] = result["team_h"].map(defence_map)
    result["own_defence_away"] = result["team_a"].map(defence_map)

    # Combined difficulty score (raw)
    result["raw_fdr_home"] = (
        result["opponent_attack_for_home"] * 0.6 + result["own_defence_home"] * 0.4
    )
    result["raw_fdr_away"] = (
        result["opponent_attack_for_away"] * 0.6 + result["own_defence_away"] * 0.4
    )

    # Scale to 1-5
    result["fdr_home"] = _scale_to_fdr(result["raw_fdr_home"])
    result["fdr_away"] = _scale_to_fdr(result["raw_fdr_away"])

    # Clean up intermediate columns
    result = result.drop(
        columns=[
            "opponent_attack_for_home",
            "opponent_attack_for_away",
            "own_defence_home",
            "own_defence_away",
            "raw_fdr_home",
            "raw_fdr_away",
        ]
    )

    return result


def _scale_to_fdr(series: pd.Series) -> pd.Series:
    """Scale a continuous difficulty score to integer FDR 1-5.

    Uses quantile-based binning to ensure reasonable distribution.
    """
    # Use rank-based percentile to assign FDR
    ranks = series.rank(pct=True)
    fdr = pd.cut(
        ranks,
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype(int)
    return fdr


def compute_rolling_strength(
    fixtures_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    window: int = 6,
) -> pd.DataFrame:
    """Compute rolling team strength based on recent form.

    Instead of full-season averages, uses last N matches to capture
    form changes (e.g., a team improving after a new signing).

    Args:
        fixtures_df: Completed fixtures, sorted by date/round.
        teams_df: Team metadata.
        window: Number of recent matches to consider.

    Returns:
        DataFrame with rolling attack/defence strength per team.
    """
    completed = fixtures_df.dropna(subset=["team_h_score", "team_a_score"]).copy()
    completed["team_h_score"] = completed["team_h_score"].astype(int)
    completed["team_a_score"] = completed["team_a_score"].astype(int)

    if completed.empty:
        return pd.DataFrame()

    # Sort by gameweek/event
    if "event" in completed.columns:
        completed = completed.sort_values("event")
    elif "kickoff_time" in completed.columns:
        completed = completed.sort_values("kickoff_time")

    # Build per-team match history
    records = []

    # Home matches
    for team_id, group in completed.groupby("team_h"):
        recent = group.tail(window)
        records.append({
            "team_id": team_id,
            "recent_goals_scored_home": recent["team_h_score"].mean(),
            "recent_goals_conceded_home": recent["team_a_score"].mean(),
            "home_matches": len(recent),
        })

    # Away matches
    away_records = {}
    for team_id, group in completed.groupby("team_a"):
        recent = group.tail(window)
        away_records[team_id] = {
            "recent_goals_scored_away": recent["team_a_score"].mean(),
            "recent_goals_conceded_away": recent["team_h_score"].mean(),
            "away_matches": len(recent),
        }

    # Combine
    df = pd.DataFrame(records).set_index("team_id")
    away_df = pd.DataFrame.from_dict(away_records, orient="index")
    away_df.index.name = "team_id"

    strength = df.join(away_df, how="outer").reset_index()

    # Merge team names
    team_names = teams_df[["id", "name"]].rename(columns={"id": "team_id", "name": "team_name"})
    strength = strength.merge(team_names, on="team_id", how="left")

    return strength
