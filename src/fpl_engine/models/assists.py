"""Assists prediction model.

Predicts the probability distribution of assists per player per gameweek.

Approach: Poisson regression (same as goals model)
- Data is Poisson-distributed (variance/mean ≈ 1.03-1.09)
- Predict lambda (expected assists rate) using LightGBM with Poisson objective
- Derive P(0), P(1), P(2+) from Poisson PMF

Key differences from goals model:
- xA (expected assists) is the primary predictor instead of xG
- Creativity metric is important (FPL's measure of chance creation)
- Assists are more evenly distributed across positions than goals
  (DEF 5.6%, MID 10.6%, FWD 8.6% — vs goals DEF 3.5%, MID 10.2%, FWD 21.5%)
- FWDs assist less than MIDs (unlike goals where FWDs dominate)

Conditioned on the player playing (minutes model handles P(playing)).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fpl_engine.features.context import add_home_away, add_season_progress
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


def build_assists_features(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute features for the assists prediction model.

    Features focus on chance creation, creativity, and team attacking context.
    """
    logger.info("Building assists features for %d rows", len(df))

    result = df.copy()
    result = result.sort_values(["element", "gameweek"])

    # ─── xA features (primary predictor for assists) ─────────────────────

    if "expected_assists" in result.columns:
        result = rolling_mean(result, columns=["expected_assists"], windows=[3, 5, 10])
        result = rolling_sum(result, columns=["expected_assists"], windows=[3, 5])
        result = compute_rolling_per90(
            result, stat_columns=["expected_assists"], windows=[5, 10]
        )
        result = cumulative_mean(result, columns=["expected_assists"])

    # ─── Assists history ─────────────────────────────────────────────────

    result = rolling_mean(result, columns=["assists"], windows=[3, 5, 10])
    result = rolling_sum(result, columns=["assists"], windows=[5, 10])
    result = compute_rolling_per90(
        result, stat_columns=["assists"], windows=[5, 10]
    )
    result = cumulative_mean(result, columns=["assists"])

    # ─── Creativity (FPL's chance creation metric) ───────────────────────

    if "creativity" in result.columns:
        result = rolling_mean(result, columns=["creativity"], windows=[3, 5])

    # ─── Team attack context (assists need goals from teammates) ─────────

    if "expected_goal_involvements" in result.columns:
        result = rolling_mean(
            result, columns=["expected_goal_involvements"], windows=[5, 10]
        )

    # ─── Minutes context ─────────────────────────────────────────────────

    result = rolling_mean(result, columns=["minutes"], windows=[3, 5])

    # ─── Fixture context ─────────────────────────────────────────────────

    result = add_home_away(result)
    result = add_season_progress(result)

    if fixtures_df is not None and teams_df is not None:
        result = add_match_difficulty(result, fixtures_df, teams_df)
        result = add_surrounding_difficulty(result, fixtures_df, teams_df)

    # ─── Value ───────────────────────────────────────────────────────────

    if "value" in result.columns and "position" in result.columns:
        result["value_vs_pos_avg"] = result.groupby("position")["value"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 0.01)
        )

    logger.info("Assists features built. Shape: %s", result.shape)
    return result


# ─── Feature column definitions ──────────────────────────────────────────────

ASSISTS_FEATURE_COLUMNS = [
    # xA features
    "expected_assists_roll3",
    "expected_assists_roll5",
    "expected_assists_roll10",
    "expected_assists_sum3",
    "expected_assists_sum5",
    "expected_assists_per90_roll5",
    "expected_assists_per90_roll10",
    "expected_assists_season_avg",
    # Assists history
    "assists_roll3",
    "assists_roll5",
    "assists_roll10",
    "assists_sum5",
    "assists_sum10",
    "assists_per90_roll5",
    "assists_per90_roll10",
    "assists_season_avg",
    # Creativity
    "creativity_roll3",
    "creativity_roll5",
    # Team attack (goal involvements)
    "expected_goal_involvements_roll5",
    "expected_goal_involvements_roll10",
    # Minutes
    "minutes_roll3",
    "minutes_roll5",
    # Fixture context
    "is_home",
    "season_progress",
    "match_difficulty",
    "prev_match_difficulty",
    "next_match_difficulty",
    # Value
    "value_vs_pos_avg",
]


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_assists_training_dataset(
    store: ParquetStore,
    seasons: list[str] | None = None,
    min_gw: int = 4,
    min_minutes: int = 1,
) -> pd.DataFrame:
    """Build training dataset for the assists model.

    Only includes rows where the player played (minutes > 0).
    """
    if seasons is None:
        seasons = store.list_seasons("gameweeks")

    logger.info("Building assists training dataset from seasons: %s", seasons)

    frames = []
    for season in seasons:
        logger.info("Processing season %s", season)
        df = store.load_gameweeks(season)
        df["season"] = season

        try:
            fixtures_df = store.load_fixtures(season)
            teams_df = store.load_teams(season)
        except FileNotFoundError:
            fixtures_df = None
            teams_df = None

        df = build_assists_features(df, fixtures_df, teams_df)
        df = df[df["minutes"] >= min_minutes]
        df = df[df["gameweek"] >= min_gw]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    logger.info(
        "Assists dataset: %d rows, %d columns",
        len(dataset), len(dataset.columns),
    )
    return dataset


# ─── Model wrapper ───────────────────────────────────────────────────────────


@dataclass
class AssistsModel:
    """Position-specific assists prediction using Poisson regression.

    Predicts lambda (expected assists rate) per player-GW, then derives
    P(0), P(1), P(2+) from the Poisson PMF.
    """

    models: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict assists distribution for all players."""
        results = []

        for pos in OUTFIELD_POSITIONS:
            pos_data = df[df["position"] == pos]
            if pos_data.empty or pos not in self.models:
                continue

            model = self.models[pos]
            features = [c for c in self.feature_columns if c in pos_data.columns]
            x_input = pos_data[features].fillna(0)

            lambda_pred = model.predict(x_input)
            lambda_pred = np.clip(lambda_pred, 0.001, 5.0)

            p_0 = poisson.pmf(0, lambda_pred)
            p_1 = poisson.pmf(1, lambda_pred)
            p_2plus = 1.0 - p_0 - p_1

            pos_result = pd.DataFrame({
                "element": pos_data["element"].values,
                "position": pos,
                "lambda_assists": lambda_pred,
                "p_0_assists": p_0,
                "p_1_assist": p_1,
                "p_2plus_assists": p_2plus,
            })
            results.append(pos_result)

        if not results:
            return pd.DataFrame(
                columns=[
                    "element", "position", "lambda_assists",
                    "p_0_assists", "p_1_assist", "p_2plus_assists",
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
        logger.info("AssistsModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> AssistsModel:
        """Load model artifacts."""
        import joblib

        path = Path(path)
        return cls(
            models=joblib.load(path / "models.pkl"),
            feature_columns=joblib.load(path / "feature_columns.pkl"),
        )
