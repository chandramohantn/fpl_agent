# ruff: noqa: E501
"""Train and evaluate the FPL component models from processed historical data.

The default temporal split follows ``docs/models``:

* train: 2023-24
* minutes calibration: 2024-25
* test: 2025-26

Artifacts are written below ``models/`` in the locations consumed by the
Model Management page and the model wrapper ``load()`` methods.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fpl_engine.models.assists import (
    ASSISTS_FEATURE_COLUMNS,
    AssistsModel,
    build_assists_training_dataset,
)
from fpl_engine.models.bonus import (
    BONUS_FEATURE_COLUMNS,
    BonusModel,
    build_bonus_training_dataset,
)
from fpl_engine.models.cards import (
    CARDS_FEATURE_COLUMNS,
    CardsModel,
    build_cards_training_dataset,
)
from fpl_engine.models.clean_sheets import (
    CS_FEATURE_COLUMNS,
    CleanSheetModel,
    build_cs_training_dataset,
)
from fpl_engine.models.goals import (
    GOALS_FEATURE_COLUMNS,
    GoalsModel,
    build_goals_training_dataset,
)
from fpl_engine.models.minutes import (
    BASE_FEATURE_COLUMNS,
    MINUTES_DID_NOT_PLAY,
    MINUTES_FULL,
    MINUTES_SUB,
    MinutesModelV2,
)
from fpl_engine.models.minutes import (
    build_training_dataset as build_minutes_training_dataset,
)
from fpl_engine.models.saves import (
    SAVES_FEATURE_COLUMNS,
    SavesModel,
    build_saves_training_dataset,
)
from fpl_engine.storage.parquet_store import ParquetStore

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTFIELD_POSITIONS = ("DEF", "MID", "FWD")
ALL_MODELS = ("minutes", "goals", "assists", "clean_sheets", "saves", "cards", "bonus")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _feature_matrix(dataset: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a numeric, ordered feature matrix, filling unavailable features with zero."""
    return (
        dataset.reindex(columns=columns, fill_value=0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


def _season_subset(dataset: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    return dataset[dataset["season"].isin(seasons)].copy()


def _require_rows(dataset: pd.DataFrame, description: str) -> None:
    if dataset.empty:
        raise ValueError(f"No rows available for {description}.")


def _save_metrics(model_dir: Path, metrics: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _regressor() -> LGBMRegressor:
    """Poisson configuration documented for the count component models."""
    return LGBMRegressor(
        objective="poisson",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=20,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )


def _classifier() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )


def train_minutes(
    store: ParquetStore, train_seasons: list[str], calibration_season: str, test_season: str
) -> dict[str, Any]:
    dataset = build_minutes_training_dataset(
        store, seasons=[*train_seasons, calibration_season, test_season]
    )
    train = _season_subset(dataset, train_seasons)
    calibration = _season_subset(dataset, [calibration_season])
    test = _season_subset(dataset, [test_season])
    models: dict[str, LGBMClassifier] = {}
    calibrators: dict[str, dict[int, IsotonicRegression]] = {}
    metrics: dict[str, Any] = {
        "train_seasons": train_seasons,
        "calibration_season": calibration_season,
        "test_season": test_season,
        "positions": {},
    }

    for position in OUTFIELD_POSITIONS:
        position_train = train[train["position"] == position]
        position_calibration = calibration[calibration["position"] == position]
        position_test = test[test["position"] == position]
        _require_rows(position_train, f"Minutes {position} training")
        _require_rows(position_calibration, f"Minutes {position} calibration")
        _require_rows(position_test, f"Minutes {position} test")

        model = _classifier()
        model.fit(
            _feature_matrix(position_train, BASE_FEATURE_COLUMNS),
            position_train["minutes_category"],
        )
        if set(model.classes_) != {MINUTES_DID_NOT_PLAY, MINUTES_SUB, MINUTES_FULL}:
            raise ValueError(
                f"Minutes {position} training does not contain all three target classes."
            )

        calibration_probs = model.predict_proba(
            _feature_matrix(position_calibration, BASE_FEATURE_COLUMNS)
        )
        calibrators[position] = {}
        for class_index, target_class in enumerate(model.classes_):
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(
                calibration_probs[:, class_index],
                (position_calibration["minutes_category"] == target_class).astype(int),
            )
            calibrators[position][int(target_class)] = calibrator

        test_probs = model.predict_proba(_feature_matrix(position_test, BASE_FEATURE_COLUMNS))
        calibrated = np.column_stack(
            [
                calibrators[position][int(target_class)].transform(test_probs[:, index])
                for index, target_class in enumerate(model.classes_)
            ]
        )
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        models[position] = model
        metrics["positions"][position] = {
            "accuracy": float(
                accuracy_score(position_test["minutes_category"], calibrated.argmax(axis=1))
            ),
            "rows": len(position_test),
        }

    path = MODEL_DIR / "minutes_v2"
    MinutesModelV2(
        models=models, calibrators=calibrators, feature_columns=BASE_FEATURE_COLUMNS
    ).save(path)
    _save_metrics(path, metrics)
    return metrics


def train_position_regressor(
    dataset_builder: Callable[..., pd.DataFrame],
    wrapper: type[GoalsModel] | type[AssistsModel] | type[BonusModel],
    feature_columns: list[str],
    target: str,
    artifact_folder: str,
    store: ParquetStore,
    train_seasons: list[str],
    test_season: str,
    positions: tuple[str, ...],
) -> dict[str, Any]:
    dataset = dataset_builder(store, seasons=[*train_seasons, test_season])
    train = _season_subset(dataset, train_seasons)
    test = _season_subset(dataset, [test_season])
    models: dict[str, LGBMRegressor] = {}
    metrics: dict[str, Any] = {
        "train_seasons": train_seasons,
        "test_season": test_season,
        "positions": {},
    }

    for position in positions:
        position_train = train[train["position"] == position]
        position_test = test[test["position"] == position]
        _require_rows(position_train, f"{artifact_folder} {position} training")
        _require_rows(position_test, f"{artifact_folder} {position} test")
        model = _regressor()
        model.fit(_feature_matrix(position_train, feature_columns), position_train[target])
        prediction = np.clip(
            model.predict(_feature_matrix(position_test, feature_columns)), 0, None
        )
        models[position] = model
        metrics["positions"][position] = {
            "mae": float(mean_absolute_error(position_test[target], prediction)),
            "predicted_mean": float(prediction.mean()),
            "actual_mean": float(position_test[target].mean()),
            "rows": len(position_test),
        }

    path = MODEL_DIR / artifact_folder
    wrapper(models=models, feature_columns=feature_columns).save(path)
    _save_metrics(path, metrics)
    return metrics


def train_clean_sheets(
    store: ParquetStore, train_seasons: list[str], test_season: str
) -> dict[str, Any]:
    dataset = build_cs_training_dataset(store, seasons=[*train_seasons, test_season])
    train = _season_subset(dataset, train_seasons)
    test = _season_subset(dataset, [test_season])
    _require_rows(train, "clean sheet training")
    _require_rows(test, "clean sheet test")
    target_train = (train["clean_sheets"] > 0).astype(int)
    target_test = (test["clean_sheets"] > 0).astype(int)
    model = _classifier()
    model.fit(_feature_matrix(train, CS_FEATURE_COLUMNS), target_train)
    probabilities = model.predict_proba(_feature_matrix(test, CS_FEATURE_COLUMNS))[:, 1]
    metrics = {
        "train_seasons": train_seasons,
        "test_season": test_season,
        "auc": float(roc_auc_score(target_test, probabilities)),
        "brier": float(brier_score_loss(target_test, probabilities)),
        "rows": len(test),
    }
    path = MODEL_DIR / "clean_sheets_v1"
    CleanSheetModel(model=model, feature_columns=CS_FEATURE_COLUMNS).save(path)
    _save_metrics(path, metrics)
    return metrics


def train_saves(store: ParquetStore, train_seasons: list[str], test_season: str) -> dict[str, Any]:
    dataset = build_saves_training_dataset(store, seasons=[*train_seasons, test_season])
    train = _season_subset(dataset, train_seasons)
    test = _season_subset(dataset, [test_season])
    _require_rows(train, "saves training")
    _require_rows(test, "saves test")
    model = _regressor()
    model.fit(_feature_matrix(train, SAVES_FEATURE_COLUMNS), train["saves"])
    prediction = np.clip(model.predict(_feature_matrix(test, SAVES_FEATURE_COLUMNS)), 0, None)
    metrics = {
        "train_seasons": train_seasons,
        "test_season": test_season,
        "mae": float(mean_absolute_error(test["saves"], prediction)),
        "predicted_mean": float(prediction.mean()),
        "actual_mean": float(test["saves"].mean()),
        "rows": len(test),
    }
    path = MODEL_DIR / "saves_v1"
    SavesModel(model=model, feature_columns=SAVES_FEATURE_COLUMNS).save(path)
    _save_metrics(path, metrics)
    return metrics


def train_cards(store: ParquetStore, train_seasons: list[str], test_season: str) -> dict[str, Any]:
    dataset = build_cards_training_dataset(store, seasons=[*train_seasons, test_season])
    train = _season_subset(dataset, train_seasons)
    test = _season_subset(dataset, [test_season])
    models: dict[str, LGBMClassifier] = {}
    metrics: dict[str, Any] = {
        "train_seasons": train_seasons,
        "test_season": test_season,
        "positions": {},
    }
    for position in OUTFIELD_POSITIONS:
        position_train = train[train["position"] == position]
        position_test = test[test["position"] == position]
        _require_rows(position_train, f"Cards {position} training")
        _require_rows(position_test, f"Cards {position} test")
        target_train = (position_train["yellow_cards"] > 0).astype(int)
        target_test = (position_test["yellow_cards"] > 0).astype(int)
        model = _classifier()
        model.fit(_feature_matrix(position_train, CARDS_FEATURE_COLUMNS), target_train)
        probabilities = model.predict_proba(_feature_matrix(position_test, CARDS_FEATURE_COLUMNS))[
            :, 1
        ]
        models[position] = model
        metrics["positions"][position] = {
            "auc": float(roc_auc_score(target_test, probabilities)),
            "brier": float(brier_score_loss(target_test, probabilities)),
            "rows": len(position_test),
        }
    path = MODEL_DIR / "cards_v1"
    CardsModel(models=models, feature_columns=CARDS_FEATURE_COLUMNS).save(path)
    _save_metrics(path, metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FPL component prediction models")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--train-seasons", nargs="+", default=["2023-24"])
    parser.add_argument("--calibration-season", default="2024-25")
    parser.add_argument("--test-season", default="2025-26")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the required data without training or writing models",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = ParquetStore(base_dir=DATA_DIR)
    required_seasons = set(args.train_seasons + [args.test_season])
    if "minutes" in args.models:
        required_seasons.add(args.calibration_season)
    missing = sorted(required_seasons - set(store.list_seasons("gameweeks")))
    if missing:
        raise SystemExit(f"Missing gameweek data for: {', '.join(missing)}")

    logger.info("Training models: %s", ", ".join(args.models))
    logger.info(
        "Train=%s, calibration=%s, test=%s",
        args.train_seasons,
        args.calibration_season,
        args.test_season,
    )
    if args.dry_run:
        logger.info("Dry run passed: all required gameweek seasons are available.")
        return

    trainers: dict[str, Callable[[], dict[str, Any]]] = {
        "minutes": lambda: train_minutes(
            store, args.train_seasons, args.calibration_season, args.test_season
        ),
        "goals": lambda: train_position_regressor(
            build_goals_training_dataset,
            GoalsModel,
            GOALS_FEATURE_COLUMNS,
            "goals_scored",
            "goals_v1",
            store,
            args.train_seasons,
            args.test_season,
            OUTFIELD_POSITIONS,
        ),
        "assists": lambda: train_position_regressor(
            build_assists_training_dataset,
            AssistsModel,
            ASSISTS_FEATURE_COLUMNS,
            "assists",
            "assists_v1",
            store,
            args.train_seasons,
            args.test_season,
            OUTFIELD_POSITIONS,
        ),
        "clean_sheets": lambda: train_clean_sheets(store, args.train_seasons, args.test_season),
        "saves": lambda: train_saves(store, args.train_seasons, args.test_season),
        "cards": lambda: train_cards(store, args.train_seasons, args.test_season),
        "bonus": lambda: train_position_regressor(
            build_bonus_training_dataset,
            BonusModel,
            BONUS_FEATURE_COLUMNS,
            "bonus",
            "bonus_v1",
            store,
            args.train_seasons,
            args.test_season,
            ("GK", *OUTFIELD_POSITIONS),
        ),
    }
    for model_name in args.models:
        metrics = trainers[model_name]()
        logger.info("%s metrics: %s", model_name, json.dumps(metrics))


if __name__ == "__main__":
    main()
