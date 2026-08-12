"""Minutes prediction model v2.

Enhancements over v1:
- Position-specific models (DEF, MID, FWD — GKPs use simpler logic)
- Injury/availability features from PlayerContext
- External match context (fatigue, important upcoming matches)
- Isotonic regression calibration layer
- Flexible API/manual override support via PlayerContext

Target variable: minutes_category
    0 = Did not play (0 minutes)
    1 = Came off bench / was subbed (1-59 minutes)
    2 = Played full or near-full match (60+ minutes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_engine.features.availability import (
    AVAILABILITY_FEATURE_COLUMNS,
    inject_player_context_vectorized,
)
from fpl_engine.features.context import (
    add_days_rest,
    add_double_gameweek,
    add_fixture_congestion,
    add_home_away,
    add_season_progress,
)
from fpl_engine.features.per90 import compute_availability_rate
from fpl_engine.features.rolling import (
    cumulative_mean,
    rolling_mean,
    rolling_pct,
    rolling_sum,
)
from fpl_engine.models.player_context import PlayerContext
from fpl_engine.storage.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

# Target classes
MINUTES_DID_NOT_PLAY = 0
MINUTES_SUB = 1  # 1-59 minutes
MINUTES_FULL = 2  # 60+ minutes

# Positions that get their own model
OUTFIELD_POSITIONS = ["DEF", "MID", "FWD"]


# ─── Target construction ─────────────────────────────────────────────────────


def build_minutes_target(df: pd.DataFrame, minutes_col: str = "minutes") -> pd.DataFrame:
    """Create the 3-class target variable from raw minutes."""
    result = df.copy()
    result["minutes_category"] = pd.cut(
        result[minutes_col],
        bins=[-1, 0, 59, 200],
        labels=[MINUTES_DID_NOT_PLAY, MINUTES_SUB, MINUTES_FULL],
    ).astype(int)
    return result


# ─── Feature construction ────────────────────────────────────────────────────


def build_minutes_features(
    df: pd.DataFrame,
    player_contexts: dict[int, PlayerContext] | None = None,
) -> pd.DataFrame:
    """Compute all features for the minutes prediction model.

    Args:
        df: Raw GW data with columns: element, gameweek, minutes, starts,
            position, was_home, kickoff_time, value, selected, opponent_team.
        player_contexts: Optional dict of player_id → PlayerContext for
                        injecting availability/injury/context features.
                        Can be from API or manually set.

    Returns:
        DataFrame with all feature columns added.
    """
    logger.info("Building minutes features for %d rows", len(df))

    result = df.copy()
    result = result.sort_values(["element", "gameweek"])

    # ─── Rolling minutes features ────────────────────────────────────────

    result = rolling_mean(result, columns=["minutes"], windows=[3, 5, 10])

    if "starts" in result.columns:
        result = rolling_mean(result, columns=["starts"], windows=[3, 5, 10])
        result = rolling_sum(result, columns=["starts"], windows=[3, 5])

    # % of recent GWs where player got 60+ mins
    result = rolling_pct(
        result,
        column="minutes",
        condition_fn=lambda x: x >= 60,
        windows=[3, 5, 10],
        output_name="started_60plus",
    )

    # % of recent GWs where player got any minutes
    result = rolling_pct(
        result,
        column="minutes",
        condition_fn=lambda x: x > 0,
        windows=[3, 5, 10],
        output_name="played_any",
    )

    # ─── Availability ────────────────────────────────────────────────────

    result = compute_availability_rate(result, windows=[5, 10, 38])
    result = cumulative_mean(result, columns=["minutes"])

    # ─── Context features ────────────────────────────────────────────────

    result = add_home_away(result)
    result = add_days_rest(result)
    result = add_fixture_congestion(result, windows=[3, 5])
    result = add_season_progress(result)
    result = add_double_gameweek(result)

    # ─── Position encoding ───────────────────────────────────────────────

    if "position" in result.columns:
        position_dummies = pd.get_dummies(result["position"], prefix="pos")
        result = pd.concat([result, position_dummies], axis=1)

    # ─── Value (price as proxy for nailedness) ───────────────────────────

    if "value" in result.columns and "position" in result.columns:
        result["value_vs_pos_avg"] = result.groupby("position")["value"].transform(
            lambda x: (x - x.mean()) / max(x.std(), 0.01)
        )
    elif "value" in result.columns:
        std = result["value"].std()
        result["value_vs_pos_avg"] = (
            result["value"] - result["value"].mean()
        ) / max(std, 0.01)

    # ─── PlayerContext / availability features ───────────────────────────

    if player_contexts:
        result = inject_player_context_vectorized(
            result, player_contexts, player_id_col="element"
        )

    logger.info("Features built. Shape: %s", result.shape)
    return result


# ─── Feature column definitions ──────────────────────────────────────────────

# Base features (available during both training and inference)
BASE_FEATURE_COLUMNS = [
    # Rolling minutes
    "minutes_roll3",
    "minutes_roll5",
    "minutes_roll10",
    # Rolling starts
    "starts_roll3",
    "starts_roll5",
    "starts_roll10",
    "starts_sum3",
    "starts_sum5",
    # Start / play percentages
    "started_60plus_pct3",
    "started_60plus_pct5",
    "started_60plus_pct10",
    "played_any_pct3",
    "played_any_pct5",
    "played_any_pct10",
    # Availability
    "availability_rate5",
    "availability_rate10",
    "availability_rate38",
    # Season average
    "minutes_season_avg",
    # Context
    "is_home",
    "days_rest",
    "matches_played_last3",
    "matches_played_last5",
    "season_progress",
    "is_dgw",
    # Value
    "value_vs_pos_avg",
]

# Context features (from PlayerContext — available at inference when context is provided)
CONTEXT_FEATURE_COLUMNS = AVAILABILITY_FEATURE_COLUMNS

# All features (base + context)
ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_training_dataset(
    store: ParquetStore,
    seasons: list[str] | None = None,
    min_gw: int = 4,
) -> pd.DataFrame:
    """Build the full training dataset for position-specific minutes models.

    Processes each season independently (no cross-season leakage),
    then concatenates. Target + features are computed per-season.

    Note: PlayerContext features are NOT available for training (we don't have
    historical API snapshots). The model learns to use BASE_FEATURE_COLUMNS
    during training, and CONTEXT_FEATURE_COLUMNS provide additional signal
    at inference time when available.

    Args:
        store: ParquetStore instance.
        seasons: Seasons to include (defaults to all available).
        min_gw: Minimum GW to include (drops early GWs with insufficient history).

    Returns:
        Complete training DataFrame with features and target.
    """
    if seasons is None:
        seasons = store.list_seasons("gameweeks")

    logger.info("Building training dataset from seasons: %s", seasons)

    frames = []
    for season in seasons:
        logger.info("Processing season %s", season)
        df = store.load_gameweeks(season)
        df["season"] = season

        df = build_minutes_target(df)
        df = build_minutes_features(df, player_contexts=None)  # No context for training

        # Drop early gameweeks
        df = df[df["gameweek"] >= min_gw]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    logger.info(
        "Training dataset: %d rows, %d columns, seasons=%s",
        len(dataset), len(dataset.columns), seasons,
    )
    return dataset


def split_by_position(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split dataset into position-specific subsets.

    Returns dict: {"DEF": df_def, "MID": df_mid, "FWD": df_fwd}
    GKPs are excluded (handled by simple rules).
    """
    splits = {}
    for pos in OUTFIELD_POSITIONS:
        mask = dataset["position"] == pos
        if mask.any():
            splits[pos] = dataset[mask].copy()
            logger.info("  %s: %d rows", pos, len(splits[pos]))
    return splits


# ─── Model wrapper ───────────────────────────────────────────────────────────


@dataclass
class MinutesModelV2:
    """Position-specific minutes prediction with calibration.

    Holds separate LightGBM models for DEF, MID, FWD, plus
    isotonic regression calibrators for each.

    GKPs use a simple rule: if availability_rate > 0.8, predict 90 min;
    otherwise predict 0. GKPs rarely come off the bench (sub GKP is ~0.5% of cases).
    """

    models: dict[str, Any] = field(default_factory=dict)  # position → LGBMClassifier
    calibrators: dict[str, Any] = field(default_factory=dict)  # position → calibrator per class
    feature_columns: list[str] = field(default_factory=list)

    def predict_proba(
        self,
        df: pd.DataFrame,
        player_contexts: dict[int, PlayerContext] | None = None,
    ) -> pd.DataFrame:
        """Predict minutes probabilities for all players.

        Args:
            df: Current season GW data (used to compute features).
            player_contexts: Optional per-player context overrides.

        Returns:
            DataFrame with columns: element, p_no_play, p_sub, p_full, position
        """
        # Build features
        featured = build_minutes_features(df, player_contexts=player_contexts)

        # Get latest row per player (most recent GW)
        latest = featured.sort_values("gameweek").groupby("element").last().reset_index()

        results = []

        for pos in OUTFIELD_POSITIONS:
            pos_data = latest[latest["position"] == pos]
            if pos_data.empty or pos not in self.models:
                continue

            model = self.models[pos]
            features = [c for c in self.feature_columns if c in pos_data.columns]
            x_input = pos_data[features].fillna(0)

            # Raw probabilities
            probs = model.predict_proba(x_input)

            # Apply calibration if available
            if pos in self.calibrators:
                probs = self._apply_calibration(probs, pos)

            pos_result = pd.DataFrame({
                "element": pos_data["element"].values,
                "position": pos,
                "p_no_play": probs[:, 0],
                "p_sub": probs[:, 1],
                "p_full": probs[:, 2],
            })
            results.append(pos_result)

        # Handle GKPs
        gk_data = latest[latest["position"] == "GK"]
        if not gk_data.empty:
            gk_result = self._predict_gk(gk_data)
            results.append(gk_result)

        if not results:
            return pd.DataFrame(columns=["element", "position", "p_no_play", "p_sub", "p_full"])

        return pd.concat(results, ignore_index=True)

    def _predict_gk(self, gk_data: pd.DataFrame) -> pd.DataFrame:
        """Simple rule-based prediction for goalkeepers.

        GKPs almost always play 90 or 0. Sub appearances are ~0.5%.
        """
        probs = np.zeros((len(gk_data), 3))

        for i, (_, row) in enumerate(gk_data.iterrows()):
            # Use availability rate as primary signal
            avail = row.get("availability_rate5", row.get("availability_rate10", 0.5))
            if pd.isna(avail):
                avail = 0.5

            # Chance of playing override
            cop = row.get("chance_of_playing_pct", np.nan)
            if not pd.isna(cop):
                avail = cop  # Use API signal if available

            # GKPs: very bimodal (play 90 or don't play)
            probs[i, 0] = 1.0 - avail  # P(0 min)
            probs[i, 1] = 0.02  # P(sub) — nearly impossible for GKPs
            probs[i, 2] = avail - 0.02  # P(60+ min)

            # Normalize
            probs[i] = probs[i].clip(0) / probs[i].clip(0).sum()

        return pd.DataFrame({
            "element": gk_data["element"].values,
            "position": "GK",
            "p_no_play": probs[:, 0],
            "p_sub": probs[:, 1],
            "p_full": probs[:, 2],
        })

    def _apply_calibration(self, probs: np.ndarray, position: str) -> np.ndarray:
        """Apply isotonic regression calibration per class."""
        calibrator = self.calibrators[position]
        calibrated = np.zeros_like(probs)

        for cls_idx in range(3):
            if cls_idx in calibrator:
                calibrated[:, cls_idx] = calibrator[cls_idx].transform(probs[:, cls_idx])
            else:
                calibrated[:, cls_idx] = probs[:, cls_idx]

        # Re-normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    def save(self, path: str | Path) -> None:
        """Save model artifacts to directory."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.models, path / "models.pkl")
        joblib.dump(self.calibrators, path / "calibrators.pkl")
        joblib.dump(self.feature_columns, path / "feature_columns.pkl")
        logger.info("MinutesModelV2 saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> MinutesModelV2:
        """Load model artifacts from directory."""
        import joblib

        path = Path(path)
        return cls(
            models=joblib.load(path / "models.pkl"),
            calibrators=joblib.load(path / "calibrators.pkl"),
            feature_columns=joblib.load(path / "feature_columns.pkl"),
        )
