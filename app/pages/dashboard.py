"""Dashboard page — view predictions, squad, and recommendations."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from fpl_engine.simulation.overrides import OverrideEffect, apply_manual_overrides
from fpl_engine.simulation.player_sim import PlayerPrediction, simulate_player_match_batch
from fpl_engine.squad.formation import recommend_formations
from fpl_engine.squad.manager import SQUAD_FILE

SQUAD_SESSION_KEYS = (
    "squad_predictions",
    "player_names",
    "player_prices",
    "squad_budget",
    "sim_results",
    "override_effects",
    "player_contexts",
    "squad_manager",
)
CURRENT_SEASON = "2026-27"
SQUAD_BUDGET = 1000  # £100.0m, in FPL price units
MAX_PER_CLUB = 3
POSITION_QUOTAS = {1: 2, 2: 5, 3: 5, 4: 3}
POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render():
    st.title("🏠 Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Squad Overview", "Simulations", "Captain Comparison", "Formation Recommendation"]
    )

    # ─── Tab 1: Squad Overview ───────────────────────────────────────────

    with tab1:
        st.subheader("Your Squad")

        # Check if squad exists in session state
        if "squad_predictions" not in st.session_state:
            st.info(
                "No squad loaded. Go to **Planning** to set up your squad, "
                "or load a current-season sample squad below."
            )
            if st.button("Load Current-Season Sample Squad"):
                try:
                    _load_sample_squad()
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
        else:
            _render_squad_table()

            with st.expander("Edit current squad"):
                _render_squad_editor()

            with st.expander("Clear current squad"):
                st.warning(
                    "This removes the squad, simulations, manual player inputs, and saved squad state."
                )
                confirmed = st.checkbox(
                    "I understand that this cannot be undone.",
                    key="confirm_clear_squad",
                )
                st.button(
                    "Clear squad",
                    type="primary",
                    disabled=not confirmed,
                    on_click=_clear_current_squad,
                )

    # ─── Tab 2: Simulations ──────────────────────────────────────────────

    with tab2:
        st.subheader("Player Simulations")

        if "squad_predictions" in st.session_state:
            active_overrides = st.session_state.get("player_contexts", {})
            if active_overrides:
                st.info(
                    f"{len(active_overrides)} manual override(s) will be applied to minutes probabilities."
                )
            n_sims = st.slider("Number of simulations", 1000, 50000, 10000, step=1000)
            if st.button("Run Simulations", type="primary"):
                _run_simulations(n_sims)

            if "sim_results" in st.session_state:
                _render_simulation_results()
        else:
            st.info("Load a squad first (Squad Overview tab).")

    # ─── Tab 3: Captain Comparison ───────────────────────────────────────

    with tab3:
        st.subheader("Captain Comparison")

        if "sim_results" in st.session_state:
            _render_captain_comparison()
        else:
            st.info("Run simulations first (Simulations tab).")

    # ─── Tab 4: Formation Recommendation ────────────────────────────────

    with tab4:
        st.subheader("Best Formation")

        if "sim_results" in st.session_state:
            _render_formation_recommendation()
        else:
            st.info("Run simulations first (Simulations tab).")


def _load_legacy_sample_squad():
    """Load a sample squad for demonstration.

    This represents a realistic £100m squad (15 players):
    - 2 GK, 5 DEF, 5 MID, 3 FWD
    - Total cost: £100.0m (£0.0m in the bank)
    - Max 3 per club
    """
    squad_info = [
        # GKs (2) — total: £9.0m
        (1, "Raya", "GK", "Arsenal", 50),  # £5.0m
        (2, "Henderson", "GK", "Crystal Palace", 40),  # £4.0m
        # DEFs (5) — total: £25.0m
        (3, "Gabriel", "DEF", "Arsenal", 55),  # £5.5m
        (4, "Saliba", "DEF", "Arsenal", 53),  # £5.3m
        (5, "TAA", "DEF", "Liverpool", 60),  # £6.0m
        (6, "Estupinan", "DEF", "Brighton", 45),  # £4.5m
        (7, "Munoz", "DEF", "Crystal Palace", 42),  # £4.2m — (enabler)
        # MIDs (5) — total: £38.5m
        (8, "Salah", "MID", "Liverpool", 125),  # £12.5m (premium)
        (9, "Palmer", "MID", "Chelsea", 95),  # £9.5m (premium)
        (10, "Mbeumo", "MID", "Brentford", 70),  # £7.0m (mid-price)
        (11, "Nkunku", "MID", "Chelsea", 55),  # £5.5m (mid-price)
        (12, "Rogers", "MID", "Aston Villa", 45),  # £4.5m (enabler)
        # FWDs (3) — total: £26.0m
        (13, "Haaland", "FWD", "Man City", 135),  # £13.5m (premium)
        (14, "Watkins", "FWD", "Aston Villa", 75),  # £7.5m (mid-price)
        (15, "Welbeck", "FWD", "Brighton", 55),  # £5.5m (budget)
    ]
    # Total: 50+40+55+53+60+45+42+125+95+70+55+45+135+75+55 = 1000
    # Budget: 1000 (£100m). Bank remaining: 0

    total_cost = sum(p[4] for p in squad_info)
    bank = 1000 - total_cost

    # Build predictions with fixture context
    predictions = [
        PlayerPrediction(
            element=1,
            position="GK",
            team="Arsenal",
            opponent="Ipswich",
            is_home=True,
            p_no_play=0.02,
            p_sub=0.01,
            p_full=0.97,
            lambda_goals=0.0,
            lambda_assists=0.0,
            p_clean_sheet=0.40,
            lambda_saves=2.5,
            p_yellow_card=0.03,
            p_red_card=0.001,
            expected_bonus=0.20,
            lambda_goals_conceded=0.9,
        ),
        PlayerPrediction(
            element=2,
            position="GK",
            team="Crystal Palace",
            opponent="West Ham",
            is_home=False,
            p_no_play=0.03,
            p_sub=0.01,
            p_full=0.96,
            lambda_goals=0.0,
            lambda_assists=0.0,
            p_clean_sheet=0.20,
            lambda_saves=3.5,
            p_yellow_card=0.04,
            p_red_card=0.001,
            expected_bonus=0.15,
            lambda_goals_conceded=1.5,
        ),
        PlayerPrediction(
            element=3,
            position="DEF",
            team="Arsenal",
            opponent="Ipswich",
            is_home=True,
            p_no_play=0.05,
            p_sub=0.05,
            p_full=0.90,
            lambda_goals=0.05,
            lambda_assists=0.06,
            p_clean_sheet=0.40,
            p_yellow_card=0.12,
            expected_bonus=0.18,
            lambda_goals_conceded=0.9,
        ),
        PlayerPrediction(
            element=4,
            position="DEF",
            team="Arsenal",
            opponent="Ipswich",
            is_home=True,
            p_no_play=0.05,
            p_sub=0.05,
            p_full=0.90,
            lambda_goals=0.04,
            lambda_assists=0.05,
            p_clean_sheet=0.40,
            p_yellow_card=0.10,
            expected_bonus=0.15,
            lambda_goals_conceded=0.9,
        ),
        PlayerPrediction(
            element=5,
            position="DEF",
            team="Liverpool",
            opponent="Burnley",
            is_home=False,
            p_no_play=0.08,
            p_sub=0.07,
            p_full=0.85,
            lambda_goals=0.03,
            lambda_assists=0.12,
            p_clean_sheet=0.30,
            p_yellow_card=0.08,
            expected_bonus=0.12,
            lambda_goals_conceded=1.2,
        ),
        PlayerPrediction(
            element=6,
            position="DEF",
            team="Brighton",
            opponent="Everton",
            is_home=True,
            p_no_play=0.10,
            p_sub=0.10,
            p_full=0.80,
            lambda_goals=0.02,
            lambda_assists=0.04,
            p_clean_sheet=0.25,
            p_yellow_card=0.10,
            expected_bonus=0.10,
            lambda_goals_conceded=1.3,
        ),
        PlayerPrediction(
            element=7,
            position="DEF",
            team="Crystal Palace",
            opponent="West Ham",
            is_home=False,
            p_no_play=0.10,
            p_sub=0.12,
            p_full=0.78,
            lambda_goals=0.02,
            lambda_assists=0.03,
            p_clean_sheet=0.18,
            p_yellow_card=0.14,
            expected_bonus=0.08,
            lambda_goals_conceded=1.5,
        ),
        PlayerPrediction(
            element=8,
            position="MID",
            team="Liverpool",
            opponent="Burnley",
            is_home=False,
            p_no_play=0.04,
            p_sub=0.08,
            p_full=0.88,
            lambda_goals=0.5,
            lambda_assists=0.3,
            p_clean_sheet=0.30,
            p_yellow_card=0.07,
            expected_bonus=0.35,
            lambda_goals_conceded=1.2,
        ),
        PlayerPrediction(
            element=9,
            position="MID",
            team="Chelsea",
            opponent="Forest",
            is_home=True,
            p_no_play=0.05,
            p_sub=0.10,
            p_full=0.85,
            lambda_goals=0.6,
            lambda_assists=0.25,
            p_clean_sheet=0.25,
            p_yellow_card=0.09,
            expected_bonus=0.30,
            lambda_goals_conceded=1.1,
        ),
        PlayerPrediction(
            element=10,
            position="MID",
            team="Brentford",
            opponent="Wolves",
            is_home=True,
            p_no_play=0.06,
            p_sub=0.10,
            p_full=0.84,
            lambda_goals=0.25,
            lambda_assists=0.15,
            p_clean_sheet=0.20,
            p_yellow_card=0.11,
            expected_bonus=0.15,
            lambda_goals_conceded=1.3,
        ),
        PlayerPrediction(
            element=11,
            position="MID",
            team="Chelsea",
            opponent="Forest",
            is_home=True,
            p_no_play=0.12,
            p_sub=0.15,
            p_full=0.73,
            lambda_goals=0.30,
            lambda_assists=0.10,
            p_clean_sheet=0.25,
            p_yellow_card=0.08,
            expected_bonus=0.12,
            lambda_goals_conceded=1.1,
        ),
        PlayerPrediction(
            element=12,
            position="MID",
            team="Aston Villa",
            opponent="Brighton",
            is_home=False,
            p_no_play=0.15,
            p_sub=0.20,
            p_full=0.65,
            lambda_goals=0.08,
            lambda_assists=0.05,
            p_clean_sheet=0.18,
            p_yellow_card=0.12,
            expected_bonus=0.05,
            lambda_goals_conceded=1.4,
        ),
        PlayerPrediction(
            element=13,
            position="FWD",
            team="Man City",
            opponent="Southampton",
            is_home=True,
            p_no_play=0.03,
            p_sub=0.05,
            p_full=0.92,
            lambda_goals=1.0,
            lambda_assists=0.2,
            p_clean_sheet=0.15,
            p_yellow_card=0.08,
            expected_bonus=0.45,
            lambda_goals_conceded=1.4,
        ),
        PlayerPrediction(
            element=14,
            position="FWD",
            team="Aston Villa",
            opponent="Brighton",
            is_home=False,
            p_no_play=0.07,
            p_sub=0.10,
            p_full=0.83,
            lambda_goals=0.35,
            lambda_assists=0.15,
            p_clean_sheet=0.18,
            p_yellow_card=0.09,
            expected_bonus=0.18,
            lambda_goals_conceded=1.4,
        ),
        PlayerPrediction(
            element=15,
            position="FWD",
            team="Brighton",
            opponent="Everton",
            is_home=True,
            p_no_play=0.30,
            p_sub=0.25,
            p_full=0.45,
            lambda_goals=0.15,
            lambda_assists=0.08,
            p_clean_sheet=0.25,
            p_yellow_card=0.07,
            expected_bonus=0.08,
            lambda_goals_conceded=1.3,
        ),
    ]

    names = {p[0]: p[1] for p in squad_info}
    prices = {p[0]: p[4] for p in squad_info}

    st.session_state["squad_predictions"] = predictions
    st.session_state["player_names"] = names
    st.session_state["player_prices"] = prices
    st.session_state["squad_budget"] = {"total": 1000, "spent": total_cost, "bank": bank}


def _load_sample_squad():
    """Load a legal 2026-27 sample from the current processed FPL data."""
    predictions, names, prices, total_cost = _build_current_season_sample_squad()
    st.session_state["squad_predictions"] = predictions
    st.session_state["player_names"] = names
    st.session_state["player_prices"] = prices
    st.session_state["squad_budget"] = {
        "total": SQUAD_BUDGET,
        "spent": total_cost,
        "bank": SQUAD_BUDGET - total_cost,
    }


def _build_current_season_sample_squad() -> (
    tuple[list[PlayerPrediction], dict[int, str], dict[int, int], int]
):
    """Build a legal, deterministic squad from the current FPL player pool."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    players_path = processed_dir / "players" / f"season={CURRENT_SEASON}" / "players.parquet"
    teams_path = processed_dir / "teams" / f"season={CURRENT_SEASON}" / "teams.parquet"
    fixtures_path = processed_dir / "fixtures" / f"season={CURRENT_SEASON}" / "fixtures.parquet"
    if not all(path.exists() for path in (players_path, teams_path, fixtures_path)):
        raise ValueError(
            f"Current-season data for {CURRENT_SEASON} is missing. "
            "Run `uv run python scripts/refresh.py` first."
        )

    players = pd.read_parquet(players_path).copy()
    teams = pd.read_parquet(teams_path)
    fixtures = pd.read_parquet(fixtures_path)
    players["now_cost"] = pd.to_numeric(players["now_cost"], errors="coerce")
    players["selection_score"] = (
        pd.to_numeric(players["ep_next"], errors="coerce").fillna(0) * 5
        + pd.to_numeric(players["points_per_game"], errors="coerce").fillna(0) * 2
        + pd.to_numeric(players["form"], errors="coerce").fillna(0)
    )
    eligible = players[
        players["can_select"].fillna(False)
        & players["status"].eq("a")
        & players["element_type"].isin(POSITION_QUOTAS)
        & players["now_cost"].gt(0)
    ].copy()

    selected: list[dict] = []
    team_counts: dict[int, int] = {}
    for position, quota in POSITION_QUOTAS.items():
        candidates = eligible[eligible["element_type"].eq(position)].sort_values(
            ["now_cost", "selection_score", "id"], ascending=[True, False, True]
        )
        for player in candidates.to_dict("records"):
            team_id = int(player["team"])
            if team_counts.get(team_id, 0) >= MAX_PER_CLUB:
                continue
            selected.append(player)
            team_counts[team_id] = team_counts.get(team_id, 0) + 1
            if sum(p["element_type"] == position for p in selected) == quota:
                break
        if sum(p["element_type"] == position for p in selected) != quota:
            raise ValueError(f"Could not find {quota} eligible {POSITION_NAMES[position]} players.")

    _upgrade_squad_within_budget(selected, eligible, team_counts)
    total_cost = int(sum(player["now_cost"] for player in selected))
    if total_cost > SQUAD_BUDGET:
        raise ValueError("Could not construct a sample squad within the £100.0m budget.")

    team_names = dict(zip(teams["id"].astype(int), teams["name"], strict=True))
    predictions = [_build_sample_prediction(player, fixtures, team_names) for player in selected]
    names = {int(player["id"]): str(player["web_name"]) for player in selected}
    prices = {int(player["id"]): int(player["now_cost"]) for player in selected}
    return predictions, names, prices, total_cost


def _upgrade_squad_within_budget(
    selected: list[dict], eligible: pd.DataFrame, team_counts: dict[int, int]
) -> None:
    """Spend spare budget on higher-ranked players without breaking FPL constraints."""
    while True:
        total_cost = sum(player["now_cost"] for player in selected)
        best_upgrade: tuple[float, int, dict] | None = None
        for index, current in enumerate(selected):
            selected_ids = {player["id"] for player in selected}
            candidates = eligible[eligible["element_type"].eq(current["element_type"])]
            for candidate in candidates.to_dict("records"):
                if candidate["id"] in selected_ids:
                    continue
                extra_cost = candidate["now_cost"] - current["now_cost"]
                score_gain = candidate["selection_score"] - current["selection_score"]
                candidate_team = int(candidate["team"])
                current_team = int(current["team"])
                projected_team_count = team_counts.get(candidate_team, 0) - int(
                    candidate_team == current_team
                )
                if (
                    extra_cost <= 0
                    or total_cost + extra_cost > SQUAD_BUDGET
                    or score_gain <= 0
                    or projected_team_count >= MAX_PER_CLUB
                ):
                    continue
                value_gain = score_gain / extra_cost
                if best_upgrade is None or value_gain > best_upgrade[0]:
                    best_upgrade = (value_gain, index, candidate)
        if best_upgrade is None:
            return

        _, index, replacement = best_upgrade
        current = selected[index]
        team_counts[int(current["team"])] -= 1
        replacement_team = int(replacement["team"])
        team_counts[replacement_team] = team_counts.get(replacement_team, 0) + 1
        selected[index] = replacement


def _build_sample_prediction(
    player: dict, fixtures: pd.DataFrame, team_names: dict[int, str]
) -> PlayerPrediction:
    """Create a simple fixture-aware prediction for a real current-season player."""
    team_id = int(player["team"])
    upcoming = fixtures[
        fixtures["event"].notna()
        & ~fixtures["finished"]
        & ((fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id))
    ].sort_values(["event", "kickoff_time"])
    if upcoming.empty:
        opponent, is_home, difficulty = "TBC", True, 3
    else:
        fixture = upcoming.iloc[0]
        is_home = int(fixture["team_h"]) == team_id
        opponent_id = int(fixture["team_a"] if is_home else fixture["team_h"])
        opponent = team_names.get(opponent_id, f"Team {opponent_id}")
        difficulty = int(fixture["team_h_difficulty"] if is_home else fixture["team_a_difficulty"])

    position = int(player["element_type"])
    goal_rate = {1: 0.0, 2: 0.06, 3: 0.14, 4: 0.28}[position]
    assist_rate = {1: 0.0, 2: 0.05, 3: 0.12, 4: 0.08}[position]
    clean_sheet = max(0.08, min(0.45, 0.42 - 0.06 * (difficulty - 2)))
    return PlayerPrediction(
        element=int(player["id"]),
        position=POSITION_NAMES[position],
        team=team_names.get(team_id, f"Team {team_id}"),
        opponent=opponent,
        is_home=is_home,
        p_no_play=0.03,
        p_sub=0.07,
        p_full=0.90,
        lambda_goals=goal_rate,
        lambda_assists=assist_rate,
        p_clean_sheet=clean_sheet,
        lambda_saves=3.0 if position == 1 else 0.0,
        p_yellow_card=0.08,
        p_red_card=0.001,
        expected_bonus=max(0.05, min(0.5, float(player["selection_score"]) / 20)),
        lambda_goals_conceded=0.8 + 0.2 * difficulty,
    )


def _render_squad_editor():
    """Render controls for removing and adding current-season players."""
    preds = st.session_state["squad_predictions"]
    names = st.session_state.get("player_names", {})
    prices = st.session_state.get("player_prices", {})
    budget = st.session_state.get("squad_budget", {})
    bank = budget.get("bank", SQUAD_BUDGET - sum(prices.get(p.element, 0) for p in preds))

    st.caption(
        "Changes use the current 2026–27 player pool and reset simulations and saved squad-management state."
    )
    remove_col, add_col = st.columns(2)
    with remove_col:
        st.markdown("**Remove player**")
        remove_options = {
            f"{names.get(p.element, f'#{p.element}')} · {p.position} · £{prices.get(p.element, 0) / 10:.1f}m": p.element
            for p in preds
        }
        selected_label = st.selectbox(
            "Current player", list(remove_options), key="dashboard_remove_player"
        )
        if st.button("Remove", key="dashboard_remove_button"):
            _remove_dashboard_player(remove_options[selected_label])
            st.rerun()

    with add_col:
        st.markdown("**Add player**")
        try:
            candidates, fixtures, team_names = _eligible_additions(preds, prices, bank)
        except ValueError as exc:
            st.warning(str(exc))
            return

        if not candidates:
            st.info("Remove a player first, or free more budget, to add an eligible replacement.")
            return

        add_options = {
            f"{player['web_name']} · {POSITION_NAMES[int(player['element_type'])]} · "
            f"{team_names[int(player['team'])]} · £{player['now_cost'] / 10:.1f}m": player
            for player in candidates
        }
        add_label = st.selectbox(
            "Eligible current-season player", list(add_options), key="dashboard_add_player"
        )
        if st.button("Add", key="dashboard_add_button"):
            _add_dashboard_player(add_options[add_label], fixtures, team_names)
            st.rerun()


def _eligible_additions(
    preds: list[PlayerPrediction], prices: dict[int, int], bank: int
) -> tuple[list[dict], pd.DataFrame, dict[int, str]]:
    """Return players who can be added without violating FPL squad constraints."""
    processed_dir = PROJECT_ROOT / "data" / "processed"
    players_path = processed_dir / "players" / f"season={CURRENT_SEASON}" / "players.parquet"
    teams_path = processed_dir / "teams" / f"season={CURRENT_SEASON}" / "teams.parquet"
    fixtures_path = processed_dir / "fixtures" / f"season={CURRENT_SEASON}" / "fixtures.parquet"
    if not all(path.exists() for path in (players_path, teams_path, fixtures_path)):
        raise ValueError(
            f"Current-season data for {CURRENT_SEASON} is missing. "
            "Run `uv run python scripts/refresh.py` first."
        )

    players = pd.read_parquet(players_path).copy()
    teams = pd.read_parquet(teams_path)
    fixtures = pd.read_parquet(fixtures_path)
    players["now_cost"] = pd.to_numeric(players["now_cost"], errors="coerce")
    players["selection_score"] = (
        pd.to_numeric(players["ep_next"], errors="coerce").fillna(0) * 5
        + pd.to_numeric(players["points_per_game"], errors="coerce").fillna(0) * 2
        + pd.to_numeric(players["form"], errors="coerce").fillna(0)
    )
    team_names = dict(zip(teams["id"].astype(int), teams["name"], strict=True))
    selected_ids = {prediction.element for prediction in preds}
    position_counts = {
        position: sum(p.position == position for p in preds) for position in POSITION_NAMES.values()
    }
    team_counts = {team: sum(p.team == team for p in preds) for team in team_names.values()}

    eligible = players[
        players["can_select"].fillna(False)
        & players["status"].eq("a")
        & players["element_type"].isin(POSITION_QUOTAS)
        & players["now_cost"].gt(0)
        & players["now_cost"].le(bank)
        & ~players["id"].isin(selected_ids)
    ]
    candidates = []
    for player in eligible.sort_values(
        ["selection_score", "web_name"], ascending=[False, True]
    ).to_dict("records"):
        position = POSITION_NAMES[int(player["element_type"])]
        team_name = team_names[int(player["team"])]
        if (
            position_counts[position] < POSITION_QUOTAS[int(player["element_type"])]
            and team_counts.get(team_name, 0) < MAX_PER_CLUB
        ):
            candidates.append(player)
    return candidates, fixtures, team_names


def _remove_dashboard_player(element: int) -> None:
    """Remove a player and invalidate data derived from the previous squad."""
    st.session_state["squad_predictions"] = [
        prediction
        for prediction in st.session_state["squad_predictions"]
        if prediction.element != element
    ]
    names = dict(st.session_state.get("player_names", {}))
    prices = dict(st.session_state.get("player_prices", {}))
    names.pop(element, None)
    prices.pop(element, None)
    st.session_state["player_names"] = names
    st.session_state["player_prices"] = prices
    _finalize_squad_edit()


def _add_dashboard_player(player: dict, fixtures: pd.DataFrame, team_names: dict[int, str]) -> None:
    """Add an eligible player and invalidate data derived from the previous squad."""
    prediction = _build_sample_prediction(player, fixtures, team_names)
    st.session_state["squad_predictions"] = [*st.session_state["squad_predictions"], prediction]
    names = dict(st.session_state.get("player_names", {}))
    prices = dict(st.session_state.get("player_prices", {}))
    names[prediction.element] = str(player["web_name"])
    prices[prediction.element] = int(player["now_cost"])
    st.session_state["player_names"] = names
    st.session_state["player_prices"] = prices
    _finalize_squad_edit()


def _finalize_squad_edit() -> None:
    """Recalculate budget and discard state that belongs to the prior squad."""
    preds = st.session_state["squad_predictions"]
    prices = st.session_state["player_prices"]
    spent = sum(prices.get(prediction.element, 0) for prediction in preds)
    st.session_state["squad_budget"] = {
        "total": SQUAD_BUDGET,
        "spent": spent,
        "bank": SQUAD_BUDGET - spent,
    }
    st.session_state.pop("sim_results", None)
    st.session_state.pop("player_contexts", None)
    st.session_state.pop("squad_manager", None)
    if SQUAD_FILE.exists():
        SQUAD_FILE.unlink()


def _clear_current_squad():
    """Remove the active and persisted squad state."""
    for key in SQUAD_SESSION_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("confirm_clear_squad", None)

    if SQUAD_FILE.exists():
        SQUAD_FILE.unlink()

    st.toast("Squad cleared.")


def _render_squad_table():
    """Display squad as a table with prices and budget."""
    preds = st.session_state["squad_predictions"]
    names = st.session_state.get("player_names", {})
    prices = st.session_state.get("player_prices", {})
    budget = st.session_state.get("squad_budget", {})

    # Show budget summary
    if budget:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Squad Cost", f"£{budget['spent']/10:.1f}m")
        with col2:
            st.metric("In The Bank", f"£{budget['bank']/10:.1f}m")
        with col3:
            st.metric("Total Budget", f"£{budget['total']/10:.1f}m")

    rows = []
    for p in preds:
        price = prices.get(p.element, 0)
        rows.append(
            {
                "Player": names.get(p.element, f"#{p.element}"),
                "Pos": p.position,
                "Team": p.team,
                "Price": f"£{price/10:.1f}m" if price else "—",
                "vs": p.opponent,
                "Home": "🏠" if p.is_home else "✈️",
                "P(play)": f"{(1-p.p_no_play)*100:.0f}%",
                "λ Goals": f"{p.lambda_goals:.2f}",
                "λ Assists": f"{p.lambda_assists:.2f}",
                "P(CS)": f"{p.p_clean_sheet*100:.0f}%",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=36 * (len(df) + 1) + 3,
        column_config={
            "Player": st.column_config.TextColumn("Player", help="Player name.", width="medium"),
            "Pos": st.column_config.TextColumn(
                "Pos",
                help="FPL position: goalkeeper, defender, midfielder, or forward.",
                width="small",
            ),
            "Team": st.column_config.TextColumn(
                "Team", help="Player's current club.", width="medium"
            ),
            "Price": st.column_config.TextColumn(
                "Price", help="Current FPL price in millions of pounds.", width="small"
            ),
            "vs": st.column_config.TextColumn(
                "vs", help="Opponent in the next scheduled fixture.", width="medium"
            ),
            "Home": st.column_config.TextColumn(
                "Home", help="🏠 means home; ✈️ means away.", width="small"
            ),
            "P(play)": st.column_config.TextColumn(
                "P(play)",
                help="Estimated chance that the player appears in the match.",
                width="small",
            ),
            "λ Goals": st.column_config.TextColumn(
                "λ Goals", help="Expected goals rate used by the sample simulation.", width="small"
            ),
            "λ Assists": st.column_config.TextColumn(
                "λ Assists",
                help="Expected assists rate used by the sample simulation.",
                width="small",
            ),
            "P(CS)": st.column_config.TextColumn(
                "P(CS)", help="Estimated probability of a clean sheet.", width="small"
            ),
        },
    )


def _run_simulations(n_sims: int):
    """Run Monte Carlo simulations for all squad players."""
    preds, effects = apply_manual_overrides(
        st.session_state["squad_predictions"],
        st.session_state.get("player_contexts", {}),
    )
    results = {}

    progress = st.progress(0)
    for i, pred in enumerate(preds):
        points = simulate_player_match_batch(pred, n_simulations=n_sims, seed=42 + i)
        results[pred.element] = {
            "points": points,
            "mean": points.mean(),
            "std": points.std(),
            "median": np.median(points),
            "p10": np.percentile(points, 10),
            "p90": np.percentile(points, 90),
            "p_blank": (points <= 2).mean(),
            "p_return": (points >= 5).mean(),
            "p_haul": (points >= 10).mean(),
        }
        progress.progress((i + 1) / len(preds))

    st.session_state["sim_results"] = results
    st.session_state["override_effects"] = effects
    suffix = f" with {len(effects)} manual override(s) applied" if effects else ""
    st.success(f"✅ Simulated {n_sims:,} outcomes for {len(preds)} players{suffix}")


def _render_simulation_results():
    """Display simulation results."""
    results = st.session_state["sim_results"]
    names = st.session_state.get("player_names", {})

    effects: list[OverrideEffect] = st.session_state.get("override_effects", [])
    if effects:
        st.markdown("**Applied manual overrides**")
        st.dataframe(
            [
                {
                    "Player": names.get(effect.element, f"#{effect.element}"),
                    "P(play) before": f"{effect.baseline_play_probability:.0%}",
                    "P(play) after": f"{effect.adjusted_play_probability:.0%}",
                    "Reason": ", ".join(effect.reasons),
                }
                for effect in effects
            ],
            hide_index=True,
            use_container_width=True,
        )

    rows = []
    for eid, r in results.items():
        rows.append(
            {
                "Player": names.get(eid, f"#{eid}"),
                "xPts": f"{r['mean']:.2f}",
                "Std": f"{r['std']:.2f}",
                "P10": f"{r['p10']:.1f}",
                "Median": f"{r['median']:.1f}",
                "P90": f"{r['p90']:.1f}",
                "P(blank)": f"{r['p_blank']*100:.0f}%",
                "P(return)": f"{r['p_return']*100:.0f}%",
                "P(haul)": f"{r['p_haul']*100:.0f}%",
            }
        )

    df = pd.DataFrame(rows).sort_values("xPts", ascending=False)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", help="Player name"),
            "xPts": st.column_config.TextColumn(
                "xPts",
                help="Expected points — average across all simulations. The primary metric for player value this GW",
            ),
            "Std": st.column_config.TextColumn(
                "Std",
                help="Standard deviation — how much the outcome varies. High Std = unpredictable (could haul or blank)",
            ),
            "P10": st.column_config.TextColumn(
                "P10",
                help="10th percentile — in a bad GW, the player still scores at least this. The 'floor'",
            ),
            "Median": st.column_config.TextColumn(
                "Median",
                help="50th percentile — the most likely single outcome. Half the time they score above, half below",
            ),
            "P90": st.column_config.TextColumn(
                "P90",
                help="90th percentile — in a great GW, the player can reach this. The 'ceiling'",
            ),
            "P(blank)": st.column_config.TextColumn(
                "P(blank)",
                help="Probability of scoring ≤2 points (appearance only or didn't play). Lower is safer",
            ),
            "P(return)": st.column_config.TextColumn(
                "P(return)",
                help="Probability of scoring ≥5 points (likely got a goal, assist, or CS). Higher is better",
            ),
            "P(haul)": st.column_config.TextColumn(
                "P(haul)",
                help="Probability of scoring ≥10 points (multiple returns — goal+assist, brace, etc). Key for captaincy",
            ),
        },
    )


def _render_formation_recommendation():
    """Recommend the highest-scoring valid starting formation."""
    predictions = st.session_state["squad_predictions"]
    results = st.session_state["sim_results"]
    names = st.session_state.get("player_names", {})
    positions = {prediction.element: prediction.position for prediction in predictions}
    recommendations = recommend_formations(predictions, results)

    if not recommendations:
        st.warning("The current squad does not contain enough players for a valid FPL formation.")
        return

    best = recommendations[0]
    formation_options = [recommendation.formation for recommendation in recommendations]
    selected_formation = st.selectbox(
        "Formation to use",
        formation_options,
        key="formation_recommendation_choice",
        help="The top option is the model recommendation. Choose any other legal formation to overrule it.",
    )
    selected = next(
        recommendation
        for recommendation in recommendations
        if recommendation.formation == selected_formation
    )
    is_manual_override = selected.formation != best.formation
    captain_name = names.get(selected.captain, f"#{selected.captain}")
    st.success(
        f"🎯 **{'Selected' if is_manual_override else 'Recommended'} formation: "
        f"{selected.formation}** — expected XI points: **{selected.expected_points:.2f}**"
    )
    st.caption(
        f"Captain suggestion: {captain_name}. This ranks every legal formation using "
        "the expected points from the latest simulation."
    )
    if is_manual_override:
        st.info(
            f"Manual formation override active. It is {selected.expected_points - best.expected_points:.2f} "
            "expected points below the top-ranked formation."
        )

    st.markdown("**Starting XI**")
    starting_rows = [
        {
            "Player": f"{names.get(element, f'#{element}')} {'👑 (C)' if element == selected.captain else ''}",
            "Position": positions.get(element, ""),
            "xPts": f"{results[element]['mean']:.2f}",
        }
        for element in selected.starting
    ]
    st.dataframe(starting_rows, hide_index=True, use_container_width=True)

    st.markdown("**Bench order**")
    bench_rows = [
        {
            "Priority": priority,
            "Player": names.get(element, f"#{element}"),
            "Position": positions.get(element, ""),
            "xPts": f"{results[element]['mean']:.2f}",
        }
        for priority, element in enumerate(selected.bench, start=1)
    ]
    st.dataframe(bench_rows, hide_index=True, use_container_width=True)

    st.markdown("**Other legal formations**")
    alternatives = pd.DataFrame(
        [
            {
                "Formation": recommendation.formation,
                "Expected XI points": f"{recommendation.expected_points:.2f}",
                "Difference from best": f"{recommendation.expected_points - best.expected_points:+.2f}",
            }
            for recommendation in recommendations
        ]
    )
    st.dataframe(alternatives, hide_index=True, use_container_width=True)


def _render_captain_comparison():
    """Show captain comparison."""
    results = st.session_state["sim_results"]
    names = st.session_state.get("player_names", {})

    rows = []
    for eid, r in results.items():
        doubled = r["points"] * 2
        rows.append(
            {
                "Player": names.get(eid, f"#{eid}"),
                "E[2×pts]": f"{doubled.mean():.1f}",
                "Std": f"{doubled.std():.1f}",
                "P(haul≥20)": f"{(doubled >= 20).mean()*100:.0f}%",
                "P(blank≤4)": f"{(doubled <= 4).mean()*100:.0f}%",
                "P90": f"{np.percentile(doubled, 90):.0f}",
                "_sort": doubled.mean(),
            }
        )

    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn(
                "Player", help="Player name — candidates are from your starting XI only"
            ),
            "E[2×pts]": st.column_config.TextColumn(
                "E[2×pts]",
                help="Expected DOUBLED points if captained. This is the primary metric — pick the highest",
            ),
            "Std": st.column_config.TextColumn(
                "Std",
                help="Volatility of doubled points. High Std = boom-or-bust captain. Good for chasing, risky for protecting rank",
            ),
            "P(haul≥20)": st.column_config.TextColumn(
                "P(haul≥20)",
                help="Probability of captain scoring 20+ doubled points (10+ actual). A massive haul. Best differential metric",
            ),
            "P(blank≤4)": st.column_config.TextColumn(
                "P(blank≤4)",
                help="Probability of captain scoring ≤4 doubled points (≤2 actual). The risk of a wasted armband. Lower is safer",
            ),
            "P90": st.column_config.TextColumn(
                "P90",
                help="90th percentile of doubled points — the ceiling if things go well. Higher = more explosive upside",
            ),
        },
    )

    # Highlight recommendation
    best = max(results.items(), key=lambda x: x[1]["mean"])
    best_name = names.get(best[0], f"#{best[0]}")
    st.success(f"🎯 **Recommended Captain: {best_name}** (highest expected doubled points)")
