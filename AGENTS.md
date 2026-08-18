# FPL Engine contributor guide

## Purpose

FPL Engine is a Python 3.12+ Fantasy Premier League decision-support system.
It ingests official FPL, historical, and Understat data; trains component
models; simulates player outcomes; and exposes recommendations in Streamlit.

## Repository map

- `src/fpl_engine/` — core package. Keep domain logic here, not in Streamlit pages.
  - `ingest/` — FPL API, historical, and Understat ingestion.
  - `storage/` — Parquet storage abstraction.
  - `features/` — feature engineering and fixture difficulty.
  - `models/` — inference models and training-dataset builders.
  - `simulation/` — player simulation and manual-context override handling.
  - `squad/`, `optimization/`, `planning/`, `agent/` — squad selection and decision logic.
- `scripts/` — executable operational entry points.
- `app/main.py` — Streamlit entry point and the only application router.
- `app/pages/` — page rendering functions. Keep them thin and session-state focused.
- `config/pipeline.yaml` — configured historical and Understat seasons for ingestion.
- `data/` — raw cache and processed Parquet data; gitignored.
- `models/` — generated trained artifacts and metrics; gitignored.
- `docs/` — architecture and model specifications. Read the relevant document before
  changing a model, simulation rule, optimizer, or pipeline behavior.
- `tests/` — pytest tests.

## Common commands

Use `uv`; do not depend on a globally activated virtual environment.

```bash
# Install development dependencies
uv sync --extra dev

# Start the frontend
uv run streamlit run app/main.py

# Fetch the configured historical and Understat seasons plus current FPL API data
uv run python scripts/run_pipeline.py

# Refresh live-season FPL data
uv run python scripts/refresh.py

# Train all component models
uv run python scripts/train_models.py

# Confirm training inputs and splits without creating artifacts
uv run python scripts/train_models.py --dry-run

# Run checks
uv run pytest
uv run ruff check .
uv run ruff format .
```

Run targeted tests while iterating, then run the relevant wider test suite before handoff.

## Data, seasons, and training

- Treat season strings as `YYYY-YY` (for example, `2026-27`).
- Update `config/pipeline.yaml` when changing the historical or Understat seasons.
  The pipeline reads this file; do not duplicate season lists in call sites.
- The Official FPL API supplies the active season. Historical and Understat data
  are partitioned by season under `data/processed/`.
- Never commit data caches, Parquet files, model `.pkl` artifacts, metrics generated
  by local training, or credentials.
- Keep training, calibration, and test seasons temporally ordered and non-overlapping.
  `scripts/train_models.py` defaults to train `2023-24`, calibrate `2024-25`, and
  test `2025-26`; change all split arguments deliberately when moving the window.
- Model definitions live in `src/fpl_engine/models/`; `scripts/train_models.py`
  orchestrates training and writes artifacts that the frontend checks.

## Frontend conventions

- Keep navigation in `app/main.py`. The project uses a custom sidebar router.
  `.streamlit/config.toml` disables Streamlit's automatic `pages/` navigation to
  prevent duplicate links.
- Pages expose `render()` and use `st.session_state` for the current squad,
  simulation output, and user choices.
- Any change that invalidates player predictions or simulation inputs must clear
  stale `sim_results` so displayed recommendations cannot be outdated.
- Manual inputs must affect the simulation inputs and must visibly explain their
  effect; do not present a saved override as active unless it changes a downstream result.
- Formation, captain, and transfer recommendations are advisory. Preserve a user
  path to select an alternative legal option and show the relevant trade-off.

## Engineering conventions

- Prefer typed functions, small pure helpers, and dataclasses/Pydantic models for
  domain data.
- Keep FPL rules explicit: a 15-player squad uses quotas of 2 GK, 5 DEF, 5 MID,
  and 3 FWD; starting XIs have one goalkeeper and a legal outfield formation.
- Guard network ingestion independently by source and season so one failure does
  not discard other available data.
- Use deterministic seeds for simulations and tests where reproducibility matters.
- Do not silently substitute placeholder data in production flows. Surface missing
  data, missing model artifacts, and failed training clearly in the UI or CLI.
- Use `pathlib.Path` for filesystem access and resolve project-relative paths from
  the project root rather than the current working directory.

## Before handing off a change

1. Add or update tests for changed business logic.
2. Run focused tests and lint/format checks appropriate to the files changed.
3. For UI changes, start Streamlit or otherwise verify the affected session-state flow.
4. State whether ingestion, training, or model artifacts were actually run; do not
   claim models were trained unless artifacts were produced successfully.
5. Call out required configuration, data refresh, or retraining steps for the user.
