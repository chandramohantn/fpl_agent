"""Saves prediction model.

Predicts the expected number of saves for a goalkeeper per gameweek.

Key characteristics:
- GK-only model (outfield players make 0 saves in FPL)
- Mean ~3.0 saves per appearance, slightly overdispersed (var/mean = 1.27)
- FPL scoring: every 3 saves = 1 point
- Away GKs make more saves (3.29 vs 2.79 at home)
- Strong correlation with xGC (0.475) — facing more shots = more saves

Approach: Poisson regression (LightGBM)
- Predict lambda (expected saves)
- Slight overdispersion is acceptable for Poisson (ratio 1.27 is mild)
- Derive P(0-2 saves), P(3-5), P(6-8), P(9+) for FPL points

Conditioned on the GK playing 60+ minutes (minutes model handles that).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import poisson

from fpl_engine.features.context import add_home_away
from fpl_engine.features.rolling import cumulative_mean, rolling_mean
from fpl_engine.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


# ─── Feature construction ────────────────────────────────────────────────────


def build_saves_features(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute features for the saves prediction model.

    Features focus on:
    - How many saves the GK has been making (rolling history)
    - How many shots the team faces (xGC as proxy)
    - Opponent attacking quality
    - Home/away (away GKs face more shots)
    """
    logger.info("Building saves features for %d rows", len(df))

    result = df.copy()
    result = result.sort_values(["element", "gameweek"])

    # ─── Saves history ───────────────────────────────────────────────────

    result = rolling_mean(result, columns=["saves"], windows=[3, 5, 10])
    result = cumulative_mean(result, columns=["saves"])

    # ─── xGC (expected goals conceded = proxy for shots faced) ───────────

    if "expected_goals_conceded" in result.columns:
        result = rolling_mean(
            result, columns=["expected_goals_conceded"], windows=[3, 5, 10]
        )
        result = cumulative_mean(result, columns=["expected_goals_conceded"])

    # ─── Goals conceded history ──────────────────────────────────────────

    if "goals_conceded" in result.columns:
        result = rolling_mean(result, columns=["goals_conceded"], windows=[3, 5])

    # ─── Opponent attack strength ────────────────────────────────────────

    if fixtures_df is not None and teams_df is not None:
        from fpl_engine.features.fixture_difficulty import compute_team_strength

        strength = compute_team_strength(fixtures_df, teams_df)
        if not strength.empty:
            opp_attack = strength.set_index("team_id")["overall_attack"]
            result["opp_attack_strength"] = result["opponent_team"].map(opp_attack)

    # ─── Home/away ───────────────────────────────────────────────────────

    result = add_home_away(result)

    logger.info("Saves features built. Shape: %s", result.shape)
    return result


# ─── Feature columns ─────────────────────────────────────────────────────────

SAVES_FEATURE_COLUMNS = [
    # Saves history
    "saves_roll3",
    "saves_roll5",
    "saves_roll10",
    "saves_season_avg",
    # xGC (shots faced proxy)
    "expected_goals_conceded_roll3",
    "expected_goals_conceded_roll5",
    "expected_goals_conceded_roll10",
    "expected_goals_conceded_season_avg",
    # Goals conceded
    "goals_conceded_roll3",
    "goals_conceded_roll5",
    # Opponent
    "opp_attack_strength",
    # Context
    "is_home",
]


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_saves_training_dataset(
    store: ParquetStore,
    seasons: list[str] | None = None,
    min_gw: int = 4,
) -> pd.DataFrame:
    """Build training dataset for the saves model.

    GK-only, filtered to 60+ minute appearances.
    """
    if seasons is None:
        seasons = store.list_seasons("gameweeks")

    logger.info("Building saves training dataset from seasons: %s", seasons)

    frames = []
    for season in seasons:
        logger.info("Processing season %s", season)
        df = store.load_gameweeks(season)
        df["season"] = season

        # GK only
        df = df[df["position"] == "GK"]

        try:
            fixtures_df = store.load_fixtures(season)
            teams_df = store.load_teams(season)
        except FileNotFoundError:
            fixtures_df = None
            teams_df = None

        df = build_saves_features(df, fixtures_df, teams_df)

        # Filter to 60+ min, GW >= min_gw
        df = df[(df["minutes"] >= 60) & (df["gameweek"] >= min_gw)]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    logger.info("Saves dataset: %d rows, %d columns", len(dataset), len(dataset.columns))
    return dataset


# ─── Model wrapper ───────────────────────────────────────────────────────────


@dataclass
class SavesModel:
    """Saves prediction model for goalkeepers.

    Predicts lambda (expected saves), then derives:
    - FPL save points distribution: P(0 pts), P(1 pt), P(2 pts), P(3+ pts)
    - Raw saves distribution: P(0-2), P(3-5), P(6-8), P(9+)
    """

    model: Any = None  # LGBMRegressor (Poisson)
    feature_columns: list[str] = field(default_factory=list)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict saves distribution for goalkeepers.

        Args:
            df: Featured DataFrame (GK rows only, output of build_saves_features).

        Returns:
            DataFrame with: element, lambda_saves, expected_save_pts,
                           p_0pts, p_1pt, p_2pts, p_3plus_pts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        features = [c for c in self.feature_columns if c in df.columns]
        x_input = df[features].fillna(0)

        lambda_pred = self.model.predict(x_input)
        lambda_pred = np.clip(lambda_pred, 0.1, 15.0)

        # Derive FPL save points distribution
        # 0 pts = 0-2 saves, 1 pt = 3-5 saves, 2 pts = 6-8, 3+ pts = 9+
        p_0_2 = poisson.cdf(2, lambda_pred)
        p_3_5 = poisson.cdf(5, lambda_pred) - poisson.cdf(2, lambda_pred)
        p_6_8 = poisson.cdf(8, lambda_pred) - poisson.cdf(5, lambda_pred)
        p_9plus = 1.0 - poisson.cdf(8, lambda_pred)

        # Expected save points
        expected_pts = p_3_5 * 1 + p_6_8 * 2 + p_9plus * 3

        return pd.DataFrame({
            "element": df["element"].values,
            "lambda_saves": lambda_pred,
            "expected_save_pts": expected_pts,
            "p_0pts_saves": p_0_2,
            "p_1pt_saves": p_3_5,
            "p_2pts_saves": p_6_8,
            "p_3plus_pts_saves": p_9plus,
        })

    def save(self, path: str | Path) -> None:
        """Save model artifacts."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.pkl")
        joblib.dump(self.feature_columns, path / "feature_columns.pkl")
        logger.info("SavesModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> SavesModel:
        """Load model artifacts."""
        import joblib

        path = Path(path)
        return cls(
            model=joblib.load(path / "model.pkl"),
            feature_columns=joblib.load(path / "feature_columns.pkl"),
        )
