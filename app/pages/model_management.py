"""Model Management page — retrain models, view metrics."""

import subprocess
import sys
from pathlib import Path

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "train_models.py"
MODEL_OPTIONS = {
    "Minutes": "minutes",
    "Goals": "goals",
    "Assists": "assists",
    "Clean Sheets": "clean_sheets",
    "Saves": "saves",
    "Cards": "cards",
    "Bonus": "bonus",
}


def load_pipeline_config() -> tuple[dict[str, list[str]] | None, str | None]:
    """Load pipeline season selections for display in the frontend."""
    try:
        with PIPELINE_CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError:
        return None, f"Pipeline config not found: {PIPELINE_CONFIG_PATH}"
    except yaml.YAMLError as exc:
        return None, f"Invalid pipeline YAML: {exc}"

    if not isinstance(config, dict):
        return None, "Pipeline config must contain a YAML mapping."

    selections = {}
    for source in ("historical", "understat"):
        seasons = config.get(source, {}).get("seasons")
        if not isinstance(seasons, list) or not all(isinstance(season, str) for season in seasons):
            return None, f"Invalid {source}.seasons configuration."
        selections[source] = seasons

    return selections, None


def find_stored_seasons(domain_dir: Path) -> list[str]:
    """Return season names from a partitioned Parquet domain."""
    if not domain_dir.exists():
        return []
    return sorted(
        directory.name.replace("season=", "")
        for directory in domain_dir.iterdir()
        if directory.is_dir() and directory.name.startswith("season=")
    )


def _run_training(
    model_names: list[str],
    training_seasons: list[str],
    calibration_season: str,
    test_season: str,
) -> None:
    """Run the real training script and present its result in the UI."""
    if not model_names:
        st.error("Choose at least one model to train.")
        return
    if not training_seasons:
        st.error("Choose at least one training season.")
        return
    if calibration_season == test_season or {
        calibration_season,
        test_season,
    }.intersection(training_seasons):
        st.error("Training, calibration, and test seasons must not overlap.")
        return
    if not TRAINING_SCRIPT.exists():
        st.error(f"Training script not found: {TRAINING_SCRIPT}")
        return

    command = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--models",
        *(MODEL_OPTIONS[name] for name in model_names),
        "--train-seasons",
        *training_seasons,
        "--calibration-season",
        calibration_season,
        "--test-season",
        test_season,
    ]
    with st.spinner("Training models. This can take a few minutes..."):
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    with st.expander("Training log", expanded=completed.returncode != 0):
        st.code(output or "No output produced.")

    if completed.returncode == 0:
        st.success(
            f"Trained and saved {len(model_names)} model(s). "
            "Open Model Status to confirm the generated artifacts."
        )
    else:
        st.error("Training failed. No success status has been reported.")


def render():
    st.title("🔧 Model Management")

    tab1, tab2, tab3 = st.tabs(["Model Status", "Retrain", "Data Status"])

    # ─── Tab 1: Model Status ─────────────────────────────────────────────

    with tab1:
        st.subheader("Trained Models")

        models_dir = PROJECT_ROOT / "models"
        model_info = [
            ("Minutes v2", "minutes_v2", "3-class classification (per position)",
             "DEF 79.6%, MID 77.9%, FWD 73.8%"),
            ("Goals v1", "goals_v1", "Poisson regression (per position)",
             "Calibrated within 1-2%"),
            ("Assists v1", "assists_v1", "Poisson regression (per position)",
             "Calibrated within 1-2%"),
            ("Clean Sheets v1", "clean_sheets_v1", "Binary classification",
             "AUC 0.627, Brier 0.198"),
            ("Saves v1", "saves_v1", "Poisson regression (GK only)",
             "FPL pts calibrated within 2%"),
            ("Cards v1", "cards_v1", "Binary classification (per position)",
             "Rates calibrated within 1-2%"),
            ("Bonus v1", "bonus_v1", "Regression (per position)",
             "Mean within 0.01-0.04"),
        ]

        for name, folder, model_type, metric in model_info:
            model_path = models_dir / folder
            artifacts = sorted(model_path.glob("*.pkl")) if model_path.exists() else []
            exists = bool(artifacts)
            status = "✅" if exists else "❌"

            with st.expander(f"{status} **{name}** — {model_type}"):
                if exists:
                    st.markdown(f"**Metric:** {metric}")
                    st.markdown(f"**Files:** {', '.join(file.name for file in artifacts)}")
                    size = sum(file.stat().st_size for file in artifacts) / 1024
                    st.markdown(f"**Size:** {size:.1f} KB")
                    metrics_path = model_path / "metrics.json"
                    if metrics_path.exists():
                        st.caption(f"Training metrics: {metrics_path.read_text(encoding='utf-8')}")
                else:
                    st.warning("No trained model artifacts found. Use the Retrain tab.")

    # ─── Tab 2: Retrain ──────────────────────────────────────────────────

    with tab2:
        st.subheader("Retrain Models")

        st.markdown(
            "Retrain prediction models using historical data. "
            "This is useful after a new season's data is loaded."
        )

        # Season selector
        available_seasons = ["2023-24", "2024-25", "2025-26"]
        training_seasons = st.multiselect(
            "Training seasons",
            available_seasons,
            default=["2023-24"],
            help="Must not overlap with calibration or test seasons.",
        )
        calibration_season = st.selectbox(
            "Calibration season (Minutes model)",
            available_seasons,
            index=1,
            help="Used for isotonic calibration of Minutes probabilities.",
        )
        test_season = st.selectbox(
            "Test season (for evaluation)",
            available_seasons,
            index=2,
        )

        st.markdown("---")

        # Model selector
        models_to_train = st.multiselect(
            "Models to retrain",
            list(MODEL_OPTIONS),
            default=list(MODEL_OPTIONS),
        )

        st.markdown("---")
        st.caption(
            "Training uses the documented LightGBM hyperparameters and saves artifacts under `models/`."
        )

        if st.button("🚀 Start Training", type="primary"):
            _run_training(models_to_train, training_seasons, calibration_season, test_season)

    # ─── Tab 3: Data Status ──────────────────────────────────────────────

    with tab3:
        st.subheader("Data Status")

        data_dir = PROJECT_ROOT / "data"
        processed_dir = data_dir / "processed"
        pipeline_config, config_error = load_pipeline_config()

        st.markdown("**Pipeline configuration:**")
        if config_error:
            st.warning(config_error)
        else:
            configured_data = {
                "Source": ["Historical FPL", "Understat xG"],
                "Configured seasons": [
                    ", ".join(pipeline_config["historical"]),
                    ", ".join(pipeline_config["understat"]),
                ],
                "Downloaded seasons": [
                    ", ".join(find_stored_seasons(processed_dir / "gameweeks")) or "None",
                    ", ".join(
                        find_stored_seasons(processed_dir / "understat" / "players")
                    )
                    or "None",
                ],
            }
            st.dataframe(configured_data, hide_index=True, use_container_width=True)

        if processed_dir.exists():
            st.markdown("**Processed data (Parquet store):**")

            domains = ["players", "gameweeks", "fixtures", "teams"]
            for domain in domains:
                domain_dir = processed_dir / domain
                if domain_dir.exists():
                    seasons = find_stored_seasons(domain_dir)
                    st.markdown(f"- **{domain}:** {', '.join(seasons)}")

            # Understat
            us_dir = processed_dir / "understat" / "players"
            if us_dir.exists():
                us_seasons = find_stored_seasons(us_dir)
                st.markdown(f"- **Understat xG:** {', '.join(us_seasons)}")

            # FBref
            fbref_dir = data_dir / "raw" / "fbref" / "processed"
            if fbref_dir.exists():
                fbref_seasons = sorted(
                    d.name for d in fbref_dir.iterdir() if d.is_dir()
                )
                st.markdown(f"- **FBref:** {', '.join(fbref_seasons)}")

            # Total size
            total_size = sum(
                f.stat().st_size for f in processed_dir.rglob("*.parquet")
            )
            st.metric("Total processed data", f"{total_size / (1024*1024):.2f} MB")
        else:
            st.warning(
                "No processed data found. Run `python scripts/run_pipeline.py` first."
            )

        st.markdown("---")
        st.markdown("**Actions:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data (current season)"):
                st.info("Run: `python scripts/refresh.py`")
        with col2:
            if st.button("📥 Full Pipeline (configured seasons)"):
                st.info("Run: `python scripts/run_pipeline.py`")
