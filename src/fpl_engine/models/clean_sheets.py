"""Clean sheets prediction model.

Predicts the probability of a clean sheet for a player's team in a given GW.

Key insight: Clean sheets are a TEAM-LEVEL event. All players on a team
either keep a CS or don't. So we model this as:
    P(team keeps clean sheet | fixture)

Then for individual players:
    P(player gets CS points) = P(team CS) × P(player plays 60+ min)

The minutes model provides P(60+ min), this model provides P(team CS).

Approach: Binary classification (Logistic / LightGBM)
- Target: 0 or 1 (team conceded or didn't)
- Features: team defensive strength, opponent attack strength, home/away,
  rolling xGC, recent CS rate, match difficulty

FPL scoring context:
- GK: 4 pts for CS
- DEF: 4 pts for CS
- MID: 1 pt for CS
- FWD: 0 pts for CS (but still useful for simulation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_engine.features.context import add_home_away, add_season_progress
from fpl_engine.features.match_context import (
    add_match_difficulty,
    add_surrounding_difficulty,
)
from fpl_engine.features.rolling import cumulative_mean, rolling_mean, rolling_sum
from fpl_engine.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


# ─── Feature construction ────────────────────────────────────────────────────


def build_cs_features(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute features for the clean sheet prediction model.

    Since CS is team-level, features focus on the team's defensive record
    and the opponent's attacking threat. We compute features at
    the player-GW level (since that's what we have), but they capture
    team-level signals.
    """
    logger.info("Building CS features for %d rows", len(df))

    result = df.copy()
    result = result.sort_values(["element", "gameweek"])

    # ─── xGC features (team concession quality) ──────────────────────────

    if "expected_goals_conceded" in result.columns:
        result = rolling_mean(
            result, columns=["expected_goals_conceded"], windows=[3, 5, 10]
        )
        result = cumulative_mean(result, columns=["expected_goals_conceded"])

    # ─── Goals conceded history ──────────────────────────────────────────

    if "goals_conceded" in result.columns:
        result = rolling_mean(result, columns=["goals_conceded"], windows=[3, 5, 10])
        result = rolling_sum(result, columns=["goals_conceded"], windows=[3, 5])
        result = cumulative_mean(result, columns=["goals_conceded"])

    # ─── Clean sheet history (rolling CS rate) ───────────────────────────

    if "clean_sheets" in result.columns:
        result = rolling_mean(result, columns=["clean_sheets"], windows=[3, 5, 10])
        result = cumulative_mean(result, columns=["clean_sheets"])

    # ─── Fixture context ─────────────────────────────────────────────────

    result = add_home_away(result)
    result = add_season_progress(result)

    if fixtures_df is not None and teams_df is not None:
        result = add_match_difficulty(result, fixtures_df, teams_df)
        result = add_surrounding_difficulty(result, fixtures_df, teams_df)

    # ─── Minutes context (CS requires 60+ min) ──────────────────────────

    result = rolling_mean(result, columns=["minutes"], windows=[3, 5])

    logger.info("CS features built. Shape: %s", result.shape)
    return result


# ─── Feature column definitions ──────────────────────────────────────────────

CS_FEATURE_COLUMNS = [
    # Matchup-specific (most important)
    "opp_attack_strength",
    "own_defence_strength",
    # xGC (expected goals conceded — opponent quality)
    "expected_goals_conceded_roll3",
    "expected_goals_conceded_roll5",
    "expected_goals_conceded_roll10",
    "expected_goals_conceded_season_avg",
    # Goals conceded history
    "goals_conceded_roll3",
    "goals_conceded_roll5",
    "goals_conceded_roll10",
    "goals_conceded_season_avg",
    # Clean sheet history
    "clean_sheets_roll3",
    "clean_sheets_roll5",
    "clean_sheets_roll10",
    "clean_sheets_season_avg",
    # Fixture context
    "is_home",
]


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_cs_training_dataset(
    store: ParquetStore,
    seasons: list[str] | None = None,
    min_gw: int = 4,
    min_minutes: int = 60,
) -> pd.DataFrame:
    """Build training dataset for the CS model.

    Filters to players who played 60+ minutes (CS eligibility).
    This is because P(CS | < 60 min) = 0 by FPL rules, so those rows
    are irrelevant for CS prediction.

    Args:
        store: ParquetStore instance.
        seasons: Seasons to include.
        min_gw: Minimum GW.
        min_minutes: Minimum minutes (default 60 — CS eligibility).
    """
    if seasons is None:
        seasons = store.list_seasons("gameweeks")

    logger.info("Building CS training dataset from seasons: %s", seasons)

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

        # Build features on all data (rolling needs full history)
        df = build_cs_features(df, fixtures_df, teams_df)

        # Filter to 60+ min (CS eligible)
        df = df[df["minutes"] >= min_minutes]
        df = df[df["gameweek"] >= min_gw]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    logger.info("CS dataset: %d rows, %d columns", len(dataset), len(dataset.columns))
    return dataset


# ─── Model wrapper ───────────────────────────────────────────────────────────


@dataclass
class CleanSheetModel:
    """Clean sheet prediction model.

    Since CS is position-independent (team event), a single model is used
    for all positions. The model predicts P(team CS) for each player-fixture.

    For FPL points calculation:
        CS points = P(team CS) × P(60+ min) × position_multiplier
        where multiplier = 4 (GK/DEF), 1 (MID), 0 (FWD)
    """

    model: Any = None  # LGBMClassifier
    feature_columns: list[str] = field(default_factory=list)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict CS probability for all players.

        Args:
            df: Featured DataFrame (output of build_cs_features).
                Should be filtered to players expected to play 60+.

        Returns:
            DataFrame with: element, position, p_clean_sheet
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        features = [c for c in self.feature_columns if c in df.columns]
        x_input = df[features].fillna(0)

        probs = self.model.predict_proba(x_input)[:, 1]  # P(CS=1)

        return pd.DataFrame({
            "element": df["element"].values,
            "position": df["position"].values if "position" in df.columns else "UNK",
            "p_clean_sheet": probs,
        })

    def save(self, path: str | Path) -> None:
        """Save model artifacts."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.pkl")
        joblib.dump(self.feature_columns, path / "feature_columns.pkl")
        logger.info("CleanSheetModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> CleanSheetModel:
        """Load model artifacts."""
        import joblib

        path = Path(path)
        return cls(
            model=joblib.load(path / "model.pkl"),
            feature_columns=joblib.load(path / "feature_columns.pkl"),
        )
