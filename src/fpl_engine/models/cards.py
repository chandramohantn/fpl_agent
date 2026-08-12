"""Cards (yellow/red) prediction model.

Predicts the probability of receiving a yellow or red card per gameweek.

Key characteristics:
- Yellow cards: 13.2% rate, binary classification
- Red cards: 0.45% rate — too rare to model with ML, use flat position rate
- DEFs get the most yellows (15.0%), FWDs least (9.8%)
- Away players get more yellows (14.3% vs 12.2%)
- More minutes = more exposure (15.6% for 60+ min vs 8.1% for subs)
- FPL: Yellow = -1 pt, Red = -3 pts

Approach: Binary classification (LightGBM) for yellow cards
- Position-specific models (card rates differ by position)
- Red cards use position-level flat rate (too rare for modeling)

Conditioned on the player playing (minutes model handles that).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_engine.features.context import add_home_away, add_season_progress
from fpl_engine.features.match_context import add_match_difficulty
from fpl_engine.features.rolling import cumulative_mean, rolling_mean, rolling_sum
from fpl_engine.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

OUTFIELD_POSITIONS = ["DEF", "MID", "FWD"]

# Red card rates by position (too rare to model, use historical rates)
RED_CARD_RATES = {
    "GK": 0.0009,
    "DEF": 0.0066,
    "MID": 0.0040,
    "FWD": 0.0024,
}


# ─── Feature construction ────────────────────────────────────────────────────


def build_cards_features(
    df: pd.DataFrame,
    fixtures_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute features for the cards prediction model.

    Card-prone players tend to be consistent — a player who gets
    booked frequently will continue to do so. Features focus on:
    - Historical card rate
    - Position
    - Home/away (away = more cards)
    - Match difficulty (harder matches = more fouls)
    - Minutes (more exposure = more risk)
    """
    logger.info("Building cards features for %d rows", len(df))

    result = df.copy()
    result = result.sort_values(["element", "gameweek"])

    # ─── Card history (strongest predictor) ──────────────────────────────

    result = rolling_mean(result, columns=["yellow_cards"], windows=[5, 10])
    result = rolling_sum(result, columns=["yellow_cards"], windows=[5, 10])
    result = cumulative_mean(result, columns=["yellow_cards"])

    # ─── Minutes (more minutes = more card risk) ─────────────────────────

    result = rolling_mean(result, columns=["minutes"], windows=[3, 5])

    # ─── Fixture context ─────────────────────────────────────────────────

    result = add_home_away(result)
    result = add_season_progress(result)

    if fixtures_df is not None and teams_df is not None:
        result = add_match_difficulty(result, fixtures_df, teams_df)

    # ─── Value (proxy for player importance — top players may be careful) ─

    if "value" in result.columns and "position" in result.columns:
        result["value_vs_pos_avg"] = result.groupby("position")["value"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 0.01)
        )

    logger.info("Cards features built. Shape: %s", result.shape)
    return result


# ─── Feature columns ─────────────────────────────────────────────────────────

CARDS_FEATURE_COLUMNS = [
    # Card history (strongest signal — card-prone players are consistent)
    "yellow_cards_roll5",
    "yellow_cards_roll10",
    "yellow_cards_sum5",
    "yellow_cards_sum10",
    "yellow_cards_season_avg",
    # Minutes (more exposure)
    "minutes_roll3",
    "minutes_roll5",
    # Context
    "is_home",
    "season_progress",
    "match_difficulty",
    # Value
    "value_vs_pos_avg",
]


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_cards_training_dataset(
    store: ParquetStore,
    seasons: list[str] | None = None,
    min_gw: int = 4,
    min_minutes: int = 1,
) -> pd.DataFrame:
    """Build training dataset for the cards model.

    Includes all players who played (minutes > 0), since even subs can
    get booked (8.1% card rate for subs).
    """
    if seasons is None:
        seasons = store.list_seasons("gameweeks")

    logger.info("Building cards training dataset from seasons: %s", seasons)

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

        df = build_cards_features(df, fixtures_df, teams_df)
        df = df[(df["minutes"] >= min_minutes) & (df["gameweek"] >= min_gw)]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    logger.info("Cards dataset: %d rows, %d columns", len(dataset), len(dataset.columns))
    return dataset


# ─── Model wrapper ───────────────────────────────────────────────────────────


@dataclass
class CardsModel:
    """Cards prediction model.

    Position-specific binary classifiers for yellow cards.
    Red cards use position-level flat rates (too rare for ML).

    Output: P(yellow card), P(red card) per player.
    Expected card points = P(YC) × (-1) + P(RC) × (-3)
    """

    models: dict[str, Any] = field(default_factory=dict)  # position → LGBMClassifier
    feature_columns: list[str] = field(default_factory=list)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict card probabilities for all players.

        Returns:
            DataFrame with: element, position, p_yellow, p_red, expected_card_pts
        """
        results = []

        for pos in OUTFIELD_POSITIONS:
            pos_data = df[df["position"] == pos]
            if pos_data.empty or pos not in self.models:
                continue

            model = self.models[pos]
            features = [c for c in self.feature_columns if c in pos_data.columns]
            x_input = pos_data[features].fillna(0)

            p_yellow = model.predict_proba(x_input)[:, 1]
            p_red = RED_CARD_RATES.get(pos, 0.004)

            expected_pts = p_yellow * (-1) + p_red * (-3)

            pos_result = pd.DataFrame({
                "element": pos_data["element"].values,
                "position": pos,
                "p_yellow_card": p_yellow,
                "p_red_card": p_red,
                "expected_card_pts": expected_pts,
            })
            results.append(pos_result)

        # GK (simple — very low card rate)
        gk_data = df[df["position"] == "GK"]
        if not gk_data.empty:
            gk_result = pd.DataFrame({
                "element": gk_data["element"].values,
                "position": "GK",
                "p_yellow_card": 0.075,  # Historical rate
                "p_red_card": RED_CARD_RATES["GK"],
                "expected_card_pts": 0.075 * (-1) + RED_CARD_RATES["GK"] * (-3),
            })
            results.append(gk_result)

        if not results:
            return pd.DataFrame(
                columns=["element", "position", "p_yellow_card",
                         "p_red_card", "expected_card_pts"]
            )

        return pd.concat(results, ignore_index=True)

    def save(self, path: str | Path) -> None:
        """Save model artifacts."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, path / "models.pkl")
        joblib.dump(self.feature_columns, path / "feature_columns.pkl")
        logger.info("CardsModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> CardsModel:
        """Load model artifacts."""
        import joblib

        path = Path(path)
        return cls(
            models=joblib.load(path / "models.pkl"),
            feature_columns=joblib.load(path / "feature_columns.pkl"),
        )
