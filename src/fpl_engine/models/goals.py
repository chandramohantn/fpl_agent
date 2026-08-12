"""Goals prediction model.

Predicts the probability distribution of goals scored per player per gameweek.

Approach: Poisson regression
- The data is near-perfectly Poisson (variance/mean ≈ 1.03-1.08)
- We predict lambda (expected goals rate) using LightGBM with Poisson objective
- Then derive P(0), P(1), P(2+) from the Poisson PMF

Key features:
- xG history (rolling per-90 expected goals)
- Shot volume (rolling shots per 90)
- Opponent defensive strength
- Match difficulty
- Team attack strength
- Home/away
- Minutes (more minutes = more opportunity)

Position-specific models:
- DEF: avg 0.037 goals/appearance — mostly set pieces
- MID: avg 0.111 goals/appearance — mixed
- FWD: avg 0.251 goals/appearance — primary scorers
- GK: excluded (essentially zero)

The model is conditioned on the player playing. The minutes model
determines P(playing), then this model determines P(goals | playing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fpl_engine.features.context import (
    add_home_away,
    add_season_progress,
)
from fpl_engine.features.match_context import (
    add_match_difficulty,
    add_surrounding_difficulty,
)
from fpl_engine.features.per90 import compute_rolling_per90
from fpl_engine.features.rolling import cumulative_mean, rolling_mean, rolling_sum
from fpl_engine.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

OUTFIELD_POSITIONS = ["DEF", "MID", "FWD"]


# ─── Feature construction ────────────────────────────────────────────────────


def build_goals_features(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute features for the goals prediction model.

    Features focus on attacking output, shot quality, and opponent weakness.

    Args:
        df: Raw GW data (must have minutes > 0 filtered before or after).
        fixtures_df: Season fixtures for computing opponent strength.
        teams_df: Season teams for computing strength.

    Returns:
        DataFrame with goals-specific features added.
    """
    logger.info("Building goals features for %d rows", len(df))

    result = df.copy()
    result = result.sort_values(["element", "gameweek"])

    # ─── xG features (the strongest predictor) ───────────────────────────

    if "expected_goals" in result.columns:
        # Rolling xG (raw cumulative)
        result = rolling_mean(result, columns=["expected_goals"], windows=[3, 5, 10])
        result = rolling_sum(result, columns=["expected_goals"], windows=[3, 5])

        # Rolling xG per 90 (minutes-adjusted rate)
        result = compute_rolling_per90(
            result, stat_columns=["expected_goals"], windows=[5, 10]
        )

        # Season average xG
        result = cumulative_mean(result, columns=["expected_goals"])

    # ─── Shots features ──────────────────────────────────────────────────

    # We can derive shots from xG presence or use goals_scored as proxy
    # Use goals_scored rolling as a feature too (hot streaks matter)
    result = rolling_mean(result, columns=["goals_scored"], windows=[3, 5, 10])
    result = rolling_sum(result, columns=["goals_scored"], windows=[5, 10])

    # Rolling goals per 90
    result = compute_rolling_per90(
        result, stat_columns=["goals_scored"], windows=[5, 10]
    )

    # Season average goals
    result = cumulative_mean(result, columns=["goals_scored"])

    # ─── Minutes context (more minutes = more opportunity) ───────────────

    result = rolling_mean(result, columns=["minutes"], windows=[3, 5])

    # ─── Fixture context ─────────────────────────────────────────────────

    result = add_home_away(result)
    result = add_season_progress(result)

    # Match difficulty (opponent strength)
    if fixtures_df is not None and teams_df is not None:
        result = add_match_difficulty(result, fixtures_df, teams_df)
        result = add_surrounding_difficulty(result, fixtures_df, teams_df)

    # ─── BPS / threat (if available) ────────────────────────────────────

    if "threat" in result.columns:
        result = rolling_mean(result, columns=["threat"], windows=[3, 5])

    # ─── Value (expensive players score more) ────────────────────────────

    if "value" in result.columns and "position" in result.columns:
        result["value_vs_pos_avg"] = result.groupby("position")["value"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 0.01)
        )

    logger.info("Goals features built. Shape: %s", result.shape)
    return result


# ─── Feature column definitions ──────────────────────────────────────────────

GOALS_FEATURE_COLUMNS = [
    # xG features (strongest predictors)
    "expected_goals_roll3",
    "expected_goals_roll5",
    "expected_goals_roll10",
    "expected_goals_sum3",
    "expected_goals_sum5",
    "expected_goals_per90_roll5",
    "expected_goals_per90_roll10",
    "expected_goals_season_avg",
    # Goals history
    "goals_scored_roll3",
    "goals_scored_roll5",
    "goals_scored_roll10",
    "goals_scored_sum5",
    "goals_scored_sum10",
    "goals_scored_per90_roll5",
    "goals_scored_per90_roll10",
    "goals_scored_season_avg",
    # Minutes context
    "minutes_roll3",
    "minutes_roll5",
    # Fixture context
    "is_home",
    "season_progress",
    "match_difficulty",
    "prev_match_difficulty",
    "next_match_difficulty",
    # Threat
    "threat_roll3",
    "threat_roll5",
    # Value
    "value_vs_pos_avg",
]


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_goals_training_dataset(
    store: ParquetStore,
    seasons: list[str] | None = None,
    min_gw: int = 4,
    min_minutes: int = 1,
) -> pd.DataFrame:
    """Build training dataset for the goals model.

    Only includes rows where the player actually played (minutes > 0).
    The goals model is conditioned on participation — the minutes model
    handles P(playing) separately.

    Args:
        store: ParquetStore instance.
        seasons: Seasons to include.
        min_gw: Minimum GW (drops early GWs with insufficient history).
        min_minutes: Minimum minutes to include a row.

    Returns:
        Training DataFrame with features and target (goals_scored).
    """
    if seasons is None:
        seasons = store.list_seasons("gameweeks")

    logger.info("Building goals training dataset from seasons: %s", seasons)

    frames = []
    for season in seasons:
        logger.info("Processing season %s", season)
        df = store.load_gameweeks(season)
        df["season"] = season

        # Load fixtures/teams for this season (for match difficulty)
        try:
            fixtures_df = store.load_fixtures(season)
            teams_df = store.load_teams(season)
        except FileNotFoundError:
            fixtures_df = None
            teams_df = None

        # Build features on ALL data (before filtering to played)
        # This ensures rolling windows use the full history
        df = build_goals_features(df, fixtures_df, teams_df)

        # Now filter to played rows only
        df = df[df["minutes"] >= min_minutes]

        # Drop early GWs
        df = df[df["gameweek"] >= min_gw]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    logger.info(
        "Goals dataset: %d rows, %d columns, seasons=%s",
        len(dataset), len(dataset.columns), seasons,
    )
    return dataset


# ─── Model wrapper ───────────────────────────────────────────────────────────


@dataclass
class GoalsModel:
    """Position-specific goals prediction model using Poisson regression.

    Predicts lambda (expected goals rate) per player-GW, then derives
    P(0), P(1), P(2+) from the Poisson PMF.

    Usage:
        model = GoalsModel.load("models/goals_v1")
        predictions = model.predict(featured_df)
        # Returns: element, position, lambda, p_0, p_1, p_2plus
    """

    models: dict[str, Any] = field(default_factory=dict)  # position → LGBMRegressor
    feature_columns: list[str] = field(default_factory=list)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict goals distribution for all players.

        Args:
            df: Featured DataFrame (output of build_goals_features).
                Should be filtered to players who will play.

        Returns:
            DataFrame with: element, position, lambda_goals, p_0, p_1, p_2plus
        """
        results = []

        for pos in OUTFIELD_POSITIONS:
            pos_data = df[df["position"] == pos]
            if pos_data.empty or pos not in self.models:
                continue

            model = self.models[pos]
            features = [c for c in self.feature_columns if c in pos_data.columns]
            x_input = pos_data[features].fillna(0)

            # Predict lambda (Poisson rate)
            lambda_pred = model.predict(x_input)
            lambda_pred = np.clip(lambda_pred, 0.001, 5.0)  # Bound predictions

            # Derive probabilities from Poisson PMF
            p_0 = poisson.pmf(0, lambda_pred)
            p_1 = poisson.pmf(1, lambda_pred)
            p_2plus = 1.0 - p_0 - p_1

            pos_result = pd.DataFrame({
                "element": pos_data["element"].values,
                "position": pos,
                "lambda_goals": lambda_pred,
                "p_0_goals": p_0,
                "p_1_goal": p_1,
                "p_2plus_goals": p_2plus,
            })
            results.append(pos_result)

        if not results:
            return pd.DataFrame(
                columns=[
                    "element", "position", "lambda_goals",
                    "p_0_goals", "p_1_goal", "p_2plus_goals",
                ]
            )

        return pd.concat(results, ignore_index=True)

    def save(self, path: str | Path) -> None:
        """Save model artifacts."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path / "models.pkl")
        joblib.dump(self.feature_columns, path / "feature_columns.pkl")
        logger.info("GoalsModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> GoalsModel:
        """Load model artifacts."""
        import joblib

        path = Path(path)
        return cls(
            models=joblib.load(path / "models.pkl"),
            feature_columns=joblib.load(path / "feature_columns.pkl"),
        )
