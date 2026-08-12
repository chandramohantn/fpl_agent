"""Opposition detail features.

Provides granular opponent and team-level features derived from existing data:

1. Opposition defensive quality (from Understat team xGA, deep allowed)
2. Opposition GK save rate (from GW saves/goals conceded data)
3. Opposition pressing intensity (from Understat PPDA)
4. Penalty/set piece taker identification (from Understat penalty goals)
5. Defensive partnership stability (from DEF co-occurrence in lineups)

These features enhance the Goals, Assists, and Clean Sheets models by
capturing the specific defensive/attacking profile of the opposition rather
than using a single aggregated "opponent strength" number.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_engine.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


# ─── 1. Opposition Defensive Quality (from Understat) ────────────────────────


def build_opponent_defensive_profile(
    understat_cache_dir: str | Path,
    season: str,
) -> pd.DataFrame:
    """Build per-team defensive profile from Understat match-by-match data.

    For each team, computes season averages and rolling metrics of:
    - xGA (expected goals against — quality of chances they concede)
    - npxGA (non-penalty xGA)
    - Goals conceded
    - Deep completions allowed (passes into their final third)
    - PPDA (passes per defensive action — pressing intensity)

    Returns:
        DataFrame with one row per team, columns:
        team_title, xga_per_match, npxga_per_match, goals_conceded_per_match,
        deep_allowed_per_match, ppda, shots_quality_conceded
    """
    from fpl_engine.ingest.understat import SEASON_MAP

    year = SEASON_MAP.get(season, season)
    cache_file = Path(understat_cache_dir) / year / "league_data.json"

    if not cache_file.exists():
        logger.warning("Understat team data not found for %s", season)
        return pd.DataFrame()

    data = json.loads(cache_file.read_text())
    teams = data.get("teams", {})

    records = []
    for team_id, team_data in teams.items():
        title = team_data["title"]
        history = team_data.get("history", [])

        if not history:
            continue

        # Compute season averages
        xga_total = sum(float(m.get("xGA", 0)) for m in history)
        npxga_total = sum(float(m.get("npxGA", 0)) for m in history)
        goals_conceded_total = sum(int(m.get("missed", 0)) for m in history)
        deep_allowed_total = sum(int(m.get("deep_allowed", 0)) for m in history)

        # PPDA: passes per defensive action (lower = more pressing)
        ppda_att_total = sum(m.get("ppda", {}).get("att", 0) for m in history)
        ppda_def_total = sum(m.get("ppda", {}).get("def", 0) for m in history)

        n_matches = len(history)

        records.append({
            "team_title": title,
            "understat_team_id": team_id,
            "matches": n_matches,
            # Defensive quality (what they concede)
            "xga_per_match": xga_total / n_matches,
            "npxga_per_match": npxga_total / n_matches,
            "goals_conceded_per_match": goals_conceded_total / n_matches,
            "deep_allowed_per_match": deep_allowed_total / n_matches,
            # Pressing (lower PPDA = more aggressive pressing)
            "ppda": ppda_att_total / max(ppda_def_total, 1),
            # Derived: shot quality conceded (xGA per deep allowed)
            "shot_quality_conceded": xga_total / max(deep_allowed_total, 1),
        })

    profile = pd.DataFrame(records)
    logger.info("Built opponent defensive profile for %d teams", len(profile))
    return profile


def add_opponent_defensive_detail(
    df: pd.DataFrame,
    opponent_profile: pd.DataFrame,
    team_name_map: dict[int, str] | None = None,
    opponent_col: str = "opponent_team",
) -> pd.DataFrame:
    """Merge opponent defensive detail features onto player-GW data.

    Args:
        df: Player-GW DataFrame with opponent_team column.
        opponent_profile: Output of build_opponent_defensive_profile().
        team_name_map: Dict mapping FPL team_id → team name (for joining).
        opponent_col: Column with opponent team identifier.

    Returns:
        DataFrame with added columns: opp_xga_per_match, opp_npxga_per_match,
        opp_goals_conceded_pm, opp_deep_allowed_pm, opp_ppda, opp_shot_quality
    """
    result = df.copy()

    if opponent_profile.empty:
        for col in OPPOSITION_DEFENSIVE_COLUMNS:
            result[col] = np.nan
        return result

    # Build name-based lookup
    profile_map = opponent_profile.set_index("team_title")

    # If we have a team_id → name mapping, use it
    if team_name_map is not None:
        opp_names = result[opponent_col].map(team_name_map)
    else:
        # Assume opponent_col already contains team names
        opp_names = result[opponent_col]

    result["opp_xga_per_match"] = opp_names.map(
        profile_map["xga_per_match"]
    )
    result["opp_npxga_per_match"] = opp_names.map(
        profile_map["npxga_per_match"]
    )
    result["opp_goals_conceded_pm"] = opp_names.map(
        profile_map["goals_conceded_per_match"]
    )
    result["opp_deep_allowed_pm"] = opp_names.map(
        profile_map["deep_allowed_per_match"]
    )
    result["opp_ppda"] = opp_names.map(profile_map["ppda"])
    result["opp_shot_quality"] = opp_names.map(
        profile_map["shot_quality_conceded"]
    )

    return result


# ─── 2. Opposition GK Save Rate ─────────────────────────────────────────────


def compute_team_gk_save_rates(
    store: ParquetStore,
    season: str,
) -> pd.DataFrame:
    """Compute GK save rate for each team from GW data.

    Save rate = saves / (saves + goals_conceded)
    Higher = harder to score against this team's GK.

    Returns:
        DataFrame with: team, gk_save_rate, total_saves, total_shots_on_target
    """
    gw = store.load_gameweeks(season)
    gk = gw[(gw["position"] == "GK") & (gw["minutes"] >= 60)]

    team_gk = gk.groupby("team").agg(
        total_saves=("saves", "sum"),
        total_goals_conceded=("goals_conceded", "sum"),
        matches=("gameweek", "count"),
    ).reset_index()

    team_gk["shots_on_target_faced"] = (
        team_gk["total_saves"] + team_gk["total_goals_conceded"]
    )
    team_gk["gk_save_rate"] = (
        team_gk["total_saves"] / team_gk["shots_on_target_faced"].clip(lower=1)
    )
    team_gk["saves_per_match"] = team_gk["total_saves"] / team_gk["matches"].clip(lower=1)

    return team_gk[["team", "gk_save_rate", "saves_per_match"]]


def add_opponent_gk_save_rate(
    df: pd.DataFrame,
    gk_save_rates: pd.DataFrame,
    team_name_map: dict[int, str] | None = None,
    opponent_col: str = "opponent_team",
) -> pd.DataFrame:
    """Add opponent GK save rate to player-GW data.

    Higher opponent save rate = harder to score.
    """
    result = df.copy()

    if gk_save_rates.empty:
        result["opp_gk_save_rate"] = np.nan
        result["opp_gk_saves_per_match"] = np.nan
        return result

    save_rate_map = gk_save_rates.set_index("team")["gk_save_rate"]
    saves_pm_map = gk_save_rates.set_index("team")["saves_per_match"]

    if team_name_map is not None:
        opp_names = result[opponent_col].map(team_name_map)
    else:
        opp_names = result[opponent_col]

    result["opp_gk_save_rate"] = opp_names.map(save_rate_map)
    result["opp_gk_saves_per_match"] = opp_names.map(saves_pm_map)

    return result


# ─── 3. Penalty / Set Piece Taker Identification ────────────────────────────


def identify_penalty_takers(
    store: ParquetStore,
    season: str,
    understat_cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Identify penalty takers for each team.

    Uses Understat data (goals - npg = penalty goals) to find who takes pens.
    Falls back to FPL penalties_missed data if Understat unavailable.

    Returns:
        DataFrame with: player_name, team, penalty_goals, is_primary_pen_taker
    """
    # Try Understat first (most reliable)
    if understat_cache_dir:
        try:
            us = store.load_understat_players(season)
            us["goals"] = us["goals"].astype(int)
            us["npg"] = us["npg"].astype(int)
            us["penalty_goals"] = us["goals"] - us["npg"]

            pen_takers = us[us["penalty_goals"] > 0][
                ["player_name", "team_title", "penalty_goals"]
            ].copy()
            pen_takers = pen_takers.rename(columns={"team_title": "team"})

            # Primary pen taker per team = highest penalty goals
            pen_takers["is_primary_pen_taker"] = pen_takers.groupby("team")[
                "penalty_goals"
            ].transform("max") == pen_takers["penalty_goals"]

            logger.info("Identified %d penalty takers from Understat", len(pen_takers))
            return pen_takers
        except Exception as e:
            logger.warning("Understat pen taker identification failed: %s", e)

    # Fallback: use penalties_missed from GW data
    gw = store.load_gameweeks(season)
    pen_missed = gw[gw["penalties_missed"] > 0].groupby(["name", "team"]).agg(
        penalties_taken=("penalties_missed", "count")
    ).reset_index()
    pen_missed["is_primary_pen_taker"] = True
    pen_missed = pen_missed.rename(columns={"name": "player_name"})

    return pen_missed


def add_set_piece_flags(
    df: pd.DataFrame,
    penalty_takers: pd.DataFrame,
) -> pd.DataFrame:
    """Add flags indicating if a player is a penalty/set piece taker.

    Output columns:
        - is_penalty_taker: 1 if player takes penalties for their team
        - penalty_goals_season: number of penalty goals scored this season
    """
    result = df.copy()
    result["is_penalty_taker"] = 0
    result["penalty_goals_season"] = 0

    if penalty_takers.empty:
        return result

    # Match by player name + team
    for _, pen in penalty_takers.iterrows():
        mask = (result["name"] == pen["player_name"]) & (
            result["team"] == pen["team"]
        )
        if mask.any():
            result.loc[mask, "is_penalty_taker"] = (
                1 if pen.get("is_primary_pen_taker", False) else 0
            )
            result.loc[mask, "penalty_goals_season"] = pen.get("penalty_goals", 0)

    return result


# ─── 4. Defensive Partnership Stability ─────────────────────────────────────


def compute_defensive_stability(
    store: ParquetStore,
    season: str,
    min_minutes: int = 60,
    window: int = 10,
) -> pd.DataFrame:
    """Compute defensive partnership stability per team.

    Measures how settled a team's defence is by looking at how often
    the same defenders play together. A stable back 4 that has played
    together for 10+ games is much harder to break down than one with
    constant rotation.

    Metrics:
    - def_continuity: % of recent GWs where the same DEF combination played
    - def_partnership_score: average pairwise appearance overlap for starting DEFs
    - unique_def_combos: how many different DEF lineups in the last N GWs (fewer = more stable)

    Returns:
        DataFrame with: team, gameweek, def_continuity, def_partnership_score,
                       unique_def_combos_last{window}
    """
    gw = store.load_gameweeks(season)

    # Get DEFs who started (60+ min)
    defs = gw[(gw["position"] == "DEF") & (gw["minutes"] >= min_minutes)].copy()
    defs = defs.sort_values(["team", "gameweek"])

    records = []

    for team, team_data in defs.groupby("team"):
        gameweeks = sorted(team_data["gameweek"].unique())

        for gw_num in gameweeks:
            # DEFs who started this GW
            current_defs = set(
                team_data[team_data["gameweek"] == gw_num]["element"].tolist()
            )

            # Look back over the window
            past_gws = [g for g in gameweeks if g < gw_num][-window:]

            if len(past_gws) < 3:
                continue

            # Count unique DEF combinations in past window
            past_combos = []
            for pg in past_gws:
                combo = frozenset(
                    team_data[team_data["gameweek"] == pg]["element"].tolist()
                )
                past_combos.append(combo)

            unique_combos = len(set(past_combos))

            # Continuity: how often did the current DEF line play together recently?
            continuity_count = sum(1 for c in past_combos if c == frozenset(current_defs))
            continuity = continuity_count / len(past_combos)

            # Partnership score: pairwise overlap
            # Average Jaccard similarity between consecutive GW lineups
            if len(past_combos) >= 2:
                overlaps = []
                for i in range(1, len(past_combos)):
                    inter = len(past_combos[i] & past_combos[i - 1])
                    union = len(past_combos[i] | past_combos[i - 1])
                    if union > 0:
                        overlaps.append(inter / union)
                partnership_score = np.mean(overlaps) if overlaps else 0
            else:
                partnership_score = 0

            records.append({
                "team": team,
                "gameweek": gw_num,
                "def_continuity": continuity,
                "def_partnership_score": partnership_score,
                f"unique_def_combos_last{window}": unique_combos,
            })

    stability_df = pd.DataFrame(records)
    logger.info("Computed defensive stability for %d team-GW pairs", len(stability_df))
    return stability_df


def add_defensive_stability(
    df: pd.DataFrame,
    stability_df: pd.DataFrame,
    team_col: str = "team",
    gameweek_col: str = "gameweek",
) -> pd.DataFrame:
    """Merge defensive stability metrics onto player-GW data.

    For the CS model: merge on own team (how stable is MY defence?)
    For goals/assists models: could merge on opponent team (is their defence settled?)
    """
    result = df.copy()

    if stability_df.empty:
        result["def_continuity"] = np.nan
        result["def_partnership_score"] = np.nan
        return result

    result = result.merge(
        stability_df,
        on=[team_col, gameweek_col],
        how="left",
    )

    return result


def add_opponent_defensive_stability(
    df: pd.DataFrame,
    stability_df: pd.DataFrame,
    opponent_col: str = "opponent_team",
    gameweek_col: str = "gameweek",
    team_name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Add opponent's defensive stability (for goals/assists models).

    Low opponent stability = easier to score against.
    """
    result = df.copy()

    if stability_df.empty:
        result["opp_def_continuity"] = np.nan
        result["opp_def_partnership_score"] = np.nan
        return result

    # Map opponent to name if needed
    if team_name_map is not None:
        result["_opp_name"] = result[opponent_col].map(team_name_map)
    else:
        result["_opp_name"] = result[opponent_col]

    opp_stability = stability_df.rename(columns={
        "team": "_opp_name",
        "def_continuity": "opp_def_continuity",
        "def_partnership_score": "opp_def_partnership_score",
    })

    # Only keep relevant columns for merge
    merge_cols = ["_opp_name", gameweek_col, "opp_def_continuity", "opp_def_partnership_score"]
    available_merge = [c for c in merge_cols if c in opp_stability.columns]

    result = result.merge(
        opp_stability[available_merge],
        on=["_opp_name", gameweek_col],
        how="left",
    )
    result = result.drop(columns=["_opp_name"], errors="ignore")

    return result


# ─── Feature column names ────────────────────────────────────────────────────

OPPOSITION_DEFENSIVE_COLUMNS = [
    "opp_xga_per_match",
    "opp_npxga_per_match",
    "opp_goals_conceded_pm",
    "opp_deep_allowed_pm",
    "opp_ppda",
    "opp_shot_quality",
]

OPPOSITION_GK_COLUMNS = [
    "opp_gk_save_rate",
    "opp_gk_saves_per_match",
]

SET_PIECE_COLUMNS = [
    "is_penalty_taker",
    "penalty_goals_season",
]

DEFENSIVE_STABILITY_COLUMNS = [
    "def_continuity",
    "def_partnership_score",
]

OPPONENT_STABILITY_COLUMNS = [
    "opp_def_continuity",
    "opp_def_partnership_score",
]

# All columns produced by this module
ALL_OPPOSITION_FEATURE_COLUMNS = (
    OPPOSITION_DEFENSIVE_COLUMNS
    + OPPOSITION_GK_COLUMNS
    + SET_PIECE_COLUMNS
    + DEFENSIVE_STABILITY_COLUMNS
    + OPPONENT_STABILITY_COLUMNS
)
