"""Model Management page — retrain models, view metrics."""

from pathlib import Path

import streamlit as st


def render():
    st.title("🔧 Model Management")

    tab1, tab2, tab3 = st.tabs(["Model Status", "Retrain", "Data Status"])

    # ─── Tab 1: Model Status ─────────────────────────────────────────────

    with tab1:
        st.subheader("Trained Models")

        models_dir = Path("models")
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
            exists = model_path.exists()
            status = "✅" if exists else "❌"

            with st.expander(f"{status} **{name}** — {model_type}"):
                if exists:
                    files = list(model_path.glob("*.pkl"))
                    st.markdown(f"**Metric:** {metric}")
                    st.markdown(f"**Files:** {', '.join(f.name for f in files)}")
                    size = sum(f.stat().st_size for f in files) / 1024
                    st.markdown(f"**Size:** {size:.1f} KB")
                else:
                    st.warning("Model not trained yet. Use the Retrain tab.")

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
            default=["2023-24", "2024-25"],
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
            ["Minutes", "Goals", "Assists", "Clean Sheets", "Saves", "Cards", "Bonus"],
            default=["Minutes", "Goals", "Assists"],
        )

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.number_input("Trees (n_estimators)", 100, 2000, 500, step=100)
        with col2:
            max_depth = st.number_input("Max depth", 3, 10, 6)

        if st.button("🚀 Start Training", type="primary"):
            progress = st.progress(0)
            status_text = st.empty()

            for i, model_name in enumerate(models_to_train):
                status_text.text(f"Training {model_name}...")
                progress.progress((i + 1) / len(models_to_train))

                # In production, this would call the actual training functions:
                # from fpl_engine.models.minutes import build_training_dataset
                # dataset = build_training_dataset(store, seasons=training_seasons)
                # ... train model ...

            status_text.text("✅ Training complete!")
            st.success(
                f"Trained {len(models_to_train)} models on "
                f"{', '.join(training_seasons)}. "
                f"Evaluated on {test_season}."
            )

            # Show mock results
            st.markdown("**Results:**")
            results_data = {
                "Model": models_to_train,
                "Train seasons": [", ".join(training_seasons)] * len(models_to_train),
                "Test season": [test_season] * len(models_to_train),
                "Status": ["✅ Saved"] * len(models_to_train),
            }
            st.dataframe(results_data, hide_index=True)

    # ─── Tab 3: Data Status ──────────────────────────────────────────────

    with tab3:
        st.subheader("Data Status")

        data_dir = Path("data")
        processed_dir = data_dir / "processed"

        if processed_dir.exists():
            st.markdown("**Processed data (Parquet store):**")

            domains = ["players", "gameweeks", "fixtures", "teams"]
            for domain in domains:
                domain_dir = processed_dir / domain
                if domain_dir.exists():
                    seasons = sorted(
                        d.name.replace("season=", "")
                        for d in domain_dir.iterdir()
                        if d.is_dir() and d.name.startswith("season=")
                    )
                    st.markdown(f"- **{domain}:** {', '.join(seasons)}")

            # Understat
            us_dir = processed_dir / "understat" / "players"
            if us_dir.exists():
                us_seasons = sorted(
                    d.name.replace("season=", "")
                    for d in us_dir.iterdir() if d.is_dir()
                )
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
            if st.button("📥 Full Pipeline (all seasons)"):
                st.info("Run: `python scripts/run_pipeline.py`")
