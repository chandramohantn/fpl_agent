"""Bonus points prediction model.

Predicts the expected bonus points (0, 1, 2, or 3) per player per gameweek.

Key insight: Bonus is a DERIVATIVE of in-match performance. The BPS (Bonus Points
System) awards bonus to the top 3 BPS scorers in each match. BPS itself is
determined by goals, assists, clean sheets, saves, passes, tackles, etc.

Therefore, bonus prediction is best approached as:
1. Predict BPS (or a proxy) from the same features that predict goals/assists/CS
2. Derive P(bonus | BPS) from the distribution

Alternatively (simpler, what we do here):
- Use historical BPS patterns, goals/assists predictions, and match context
- LightGBM ordinal/regression to predict expected bonus directly

Data characteristics:
- 89.5% get 0 bonus (only top 3 per match get bonus)
- Mean: 0.21 per appearance
- BPS-bonus correlation: 0.693
- Goals-bonus correlation: 0.622
- FWDs get most bonus (16.7% rate), DEFs least (9.0%)

Approach: Regression (predict expected bonus 0-3) using LightGBM
- The target is treated as continuous (0-3) for simplicity
- This gives us E[bonus] directly, which is what we need for expected points
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_engine.features.context import add_home_away
from fpl_engine.features.match_context import add_match_difficulty
from fpl_engine.features.per90 import compute_rolling_per90
from fpl_engine.features.rolling import cumulative_mean, rolling_mean
from fpl_engine.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

OUTFIELD_POSITIONS = ["DEF", "MID", "FWD"]


# ─── Feature construction ────────────────────────────────────────────────────


def build_bonus_features(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute features for the bonus prediction model.

    Since bonus is derived from in-match performance, the best predictors
    are the same stats that predict goals, assists, and clean sheets:
    - xG, xA, xGI (attacking output → BPS)
    - BPS history (direct predictor)
    - Clean sheets (GK/DEF bonus source)
    - Goals/assists history (the #1 way to earn bonus)
    """
    logger.info("Building bonus features for %d rows", len(df))

    result = df.copy()
    result = result.sort_values(["element", "gameweek"])

    # ─── BPS history (most direct predictor) ─────────────────────────────

    if "bps" in result.columns:
        result = rolling_mean(result, columns=["bps"], windows=[3, 5, 10])
        result = cumulative_mean(result, columns=["bps"])

    # ─── Bonus history ───────────────────────────────────────────────────

    result = rolling_mean(result, columns=["bonus"], windows=[3, 5, 10])
    result = cumulative_mean(result, columns=["bonus"])

    # ─── xG/xA/xGI (attacking output drives BPS) ────────────────────────

    if "expected_goals" in result.columns:
        result = rolling_mean(result, columns=["expected_goals"], windows=[5, 10])
        result = compute_rolling_per90(
            result, stat_columns=["expected_goals"], windows=[5]
        )

    if "expected_assists" in result.columns:
        result = rolling_mean(result, columns=["expected_assists"], windows=[5, 10])

    if "expected_goal_involvements" in result.columns:
        result = rolling_mean(
            result, columns=["expected_goal_involvements"], windows=[5, 10]
        )

    # ─── Goals and assists (strongest bonus predictors) ──────────────────

    result = rolling_mean(result, columns=["goals_scored"], windows=[5, 10])
    result = rolling_mean(result, columns=["assists"], windows=[5, 10])

    # ─── ICT index (FPL's own performance metric) ────────────────────────

    if "ict_index" in result.columns:
        result = rolling_mean(result, columns=["ict_index"], windows=[3, 5])

    # ─── Minutes & context ───────────────────────────────────────────────

    result = rolling_mean(result, columns=["minutes"], windows=[3, 5])
    result = add_home_away(result)

    if fixtures_df is not None and teams_df is not None:
        result = add_match_difficulty(result, fixtures_df, teams_df)

    # ─── Value ───────────────────────────────────────────────────────────

    if "value" in result.columns and "position" in result.columns:
        result["value_vs_pos_avg"] = result.groupby("position")["value"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 0.01)
        )

    logger.info("Bonus features built. Shape: %s", result.shape)
    return result


# ─── Feature columns ─────────────────────────────────────────────────────────

BONUS_FEATURE_COLUMNS = [
    # BPS history (strongest signal)
    "bps_roll3",
    "bps_roll5",
    "bps_roll10",
    "bps_season_avg",
    # Bonus history
    "bonus_roll3",
    "bonus_roll5",
    "bonus_roll10",
    "bonus_season_avg",
    # xG/xA (attacking output → bonus)
    "expected_goals_roll5",
    "expected_goals_roll10",
    "expected_goals_per90_roll5",
    "expected_assists_roll5",
    "expected_assists_roll10",
    "expected_goal_involvements_roll5",
    "expected_goal_involvements_roll10",
    # Actual performance
    "goals_scored_roll5",
    "goals_scored_roll10",
    "assists_roll5",
    "assists_roll10",
    # ICT
    "ict_index_roll3",
    "ict_index_roll5",
    # Context
    "minutes_roll3",
    "minutes_roll5",
    "is_home",
    "match_difficulty",
    # Value
    "value_vs_pos_avg",
]


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_bonus_training_dataset(
    store: ParquetStore,
    seasons: list[str] | None = None,
    min_gw: int = 4,
    min_minutes: int = 1,
) -> pd.DataFrame:
    """Build training dataset for the bonus model.

    Includes all players who played (minutes > 0).
    """
    if seasons is None:
        seasons = store.list_seasons("gameweeks")

    logger.info("Building bonus training dataset from seasons: %s", seasons)

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

        df = build_bonus_features(df, fixtures_df, teams_df)
        df = df[(df["minutes"] >= min_minutes) & (df["gameweek"] >= min_gw)]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    logger.info("Bonus dataset: %d rows, %d columns", len(dataset), len(dataset.columns))
    return dataset


# ─── Model wrapper ───────────────────────────────────────────────────────────


@dataclass
class BonusModel:
    """Bonus points prediction model.

    Position-specific regression models that predict E[bonus] (0-3).
    GKs included since they earn bonus from saves and clean sheets.

    Output: expected bonus per player (continuous 0-3).
    """

    models: dict[str, Any] = field(default_factory=dict)  # position → LGBMRegressor
    feature_columns: list[str] = field(default_factory=list)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict expected bonus for all players.

        Returns:
            DataFrame with: element, position, expected_bonus
        """
        results = []
        positions = OUTFIELD_POSITIONS + ["GK"]

        for pos in positions:
            pos_data = df[df["position"] == pos]
            if pos_data.empty or pos not in self.models:
                continue

            model = self.models[pos]
            features = [c for c in self.feature_columns if c in pos_data.columns]
            x_input = pos_data[features].fillna(0)

            bonus_pred = model.predict(x_input)
            bonus_pred = np.clip(bonus_pred, 0, 3)

            pos_result = pd.DataFrame({
                "element": pos_data["element"].values,
                "position": pos,
                "expected_bonus": bonus_pred,
            })
            results.append(pos_result)

        if not results:
            return pd.DataFrame(columns=["element", "position", "expected_bonus"])

        return pd.concat(results, ignore_index=True)

    def save(self, path: str | Path) -> None:
        """Save model artifacts."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path / "models.pkl")
        joblib.dump(self.feature_columns, path / "feature_columns.pkl")
        logger.info("BonusModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> BonusModel:
        """Load model artifacts."""
        import joblib

        path = Path(path)
        return cls(
            models=joblib.load(path / "models.pkl"),
            feature_columns=joblib.load(path / "feature_columns.pkl"),
        )
