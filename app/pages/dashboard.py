"""Dashboard page — view predictions, squad, and recommendations."""

import numpy as np
import pandas as pd
import streamlit as st
from fpl_engine.simulation.player_sim import PlayerPrediction, simulate_player_match_batch


def render():
    st.title("🏠 Dashboard")

    tab1, tab2, tab3 = st.tabs(["Squad Overview", "Simulations", "Captain Comparison"])

    # ─── Tab 1: Squad Overview ───────────────────────────────────────────

    with tab1:
        st.subheader("Your Squad")

        # Check if squad exists in session state
        if "squad_predictions" not in st.session_state:
            st.info(
                "No squad loaded. Go to **Planning** to set up your squad, "
                "or load sample data below."
            )
            if st.button("Load Sample Squad"):
                _load_sample_squad()
                st.rerun()
        else:
            _render_squad_table()

    # ─── Tab 2: Simulations ──────────────────────────────────────────────

    with tab2:
        st.subheader("Player Simulations")

        if "squad_predictions" in st.session_state:
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


def _load_sample_squad():
    """Load a sample squad for demonstration.

    This represents a realistic £100m squad (15 players):
    - 2 GK, 5 DEF, 5 MID, 3 FWD
    - Total cost: £100.0m (£0.0m in the bank)
    - Max 3 per club
    """
    squad_info = [
        # GKs (2) — total: £9.0m
        (1, "Raya", "GK", "Arsenal", 50),         # £5.0m
        (2, "Henderson", "GK", "Crystal Palace", 40),  # £4.0m
        # DEFs (5) — total: £25.0m
        (3, "Gabriel", "DEF", "Arsenal", 55),      # £5.5m
        (4, "Saliba", "DEF", "Arsenal", 53),       # £5.3m
        (5, "TAA", "DEF", "Liverpool", 60),        # £6.0m
        (6, "Estupinan", "DEF", "Brighton", 45),   # £4.5m
        (7, "Munoz", "DEF", "Crystal Palace", 42), # £4.2m — (enabler)
        # MIDs (5) — total: £38.5m
        (8, "Salah", "MID", "Liverpool", 125),     # £12.5m (premium)
        (9, "Palmer", "MID", "Chelsea", 95),       # £9.5m (premium)
        (10, "Mbeumo", "MID", "Brentford", 70),   # £7.0m (mid-price)
        (11, "Nkunku", "MID", "Chelsea", 55),      # £5.5m (mid-price)
        (12, "Rogers", "MID", "Aston Villa", 45),  # £4.5m (enabler)
        # FWDs (3) — total: £26.0m
        (13, "Haaland", "FWD", "Man City", 135),   # £13.5m (premium)
        (14, "Watkins", "FWD", "Aston Villa", 75), # £7.5m (mid-price)
        (15, "Welbeck", "FWD", "Brighton", 55),    # £5.5m (budget)
    ]
    # Total: 50+40+55+53+60+45+42+125+95+70+55+45+135+75+55 = 1000
    # Budget: 1000 (£100m). Bank remaining: 0

    total_cost = sum(p[4] for p in squad_info)
    bank = 1000 - total_cost

    # Build predictions with fixture context
    predictions = [
        PlayerPrediction(element=1, position="GK", team="Arsenal", opponent="Ipswich", is_home=True,
                        p_no_play=0.02, p_sub=0.01, p_full=0.97,
                        lambda_goals=0.0, lambda_assists=0.0,
                        p_clean_sheet=0.40, lambda_saves=2.5,
                        p_yellow_card=0.03, p_red_card=0.001,
                        expected_bonus=0.20, lambda_goals_conceded=0.9),
        PlayerPrediction(element=2, position="GK", team="Crystal Palace", opponent="West Ham", is_home=False,
                        p_no_play=0.03, p_sub=0.01, p_full=0.96,
                        lambda_goals=0.0, lambda_assists=0.0,
                        p_clean_sheet=0.20, lambda_saves=3.5,
                        p_yellow_card=0.04, p_red_card=0.001,
                        expected_bonus=0.15, lambda_goals_conceded=1.5),
        PlayerPrediction(element=3, position="DEF", team="Arsenal", opponent="Ipswich", is_home=True,
                        p_no_play=0.05, p_sub=0.05, p_full=0.90,
                        lambda_goals=0.05, lambda_assists=0.06,
                        p_clean_sheet=0.40, p_yellow_card=0.12,
                        expected_bonus=0.18, lambda_goals_conceded=0.9),
        PlayerPrediction(element=4, position="DEF", team="Arsenal", opponent="Ipswich", is_home=True,
                        p_no_play=0.05, p_sub=0.05, p_full=0.90,
                        lambda_goals=0.04, lambda_assists=0.05,
                        p_clean_sheet=0.40, p_yellow_card=0.10,
                        expected_bonus=0.15, lambda_goals_conceded=0.9),
        PlayerPrediction(element=5, position="DEF", team="Liverpool", opponent="Burnley", is_home=False,
                        p_no_play=0.08, p_sub=0.07, p_full=0.85,
                        lambda_goals=0.03, lambda_assists=0.12,
                        p_clean_sheet=0.30, p_yellow_card=0.08,
                        expected_bonus=0.12, lambda_goals_conceded=1.2),
        PlayerPrediction(element=6, position="DEF", team="Brighton", opponent="Everton", is_home=True,
                        p_no_play=0.10, p_sub=0.10, p_full=0.80,
                        lambda_goals=0.02, lambda_assists=0.04,
                        p_clean_sheet=0.25, p_yellow_card=0.10,
                        expected_bonus=0.10, lambda_goals_conceded=1.3),
        PlayerPrediction(element=7, position="DEF", team="Crystal Palace", opponent="West Ham", is_home=False,
                        p_no_play=0.10, p_sub=0.12, p_full=0.78,
                        lambda_goals=0.02, lambda_assists=0.03,
                        p_clean_sheet=0.18, p_yellow_card=0.14,
                        expected_bonus=0.08, lambda_goals_conceded=1.5),
        PlayerPrediction(element=8, position="MID", team="Liverpool", opponent="Burnley", is_home=False,
                        p_no_play=0.04, p_sub=0.08, p_full=0.88,
                        lambda_goals=0.5, lambda_assists=0.3,
                        p_clean_sheet=0.30, p_yellow_card=0.07,
                        expected_bonus=0.35, lambda_goals_conceded=1.2),
        PlayerPrediction(element=9, position="MID", team="Chelsea", opponent="Forest", is_home=True,
                        p_no_play=0.05, p_sub=0.10, p_full=0.85,
                        lambda_goals=0.6, lambda_assists=0.25,
                        p_clean_sheet=0.25, p_yellow_card=0.09,
                        expected_bonus=0.30, lambda_goals_conceded=1.1),
        PlayerPrediction(element=10, position="MID", team="Brentford", opponent="Wolves", is_home=True,
                        p_no_play=0.06, p_sub=0.10, p_full=0.84,
                        lambda_goals=0.25, lambda_assists=0.15,
                        p_clean_sheet=0.20, p_yellow_card=0.11,
                        expected_bonus=0.15, lambda_goals_conceded=1.3),
        PlayerPrediction(element=11, position="MID", team="Chelsea", opponent="Forest", is_home=True,
                        p_no_play=0.12, p_sub=0.15, p_full=0.73,
                        lambda_goals=0.30, lambda_assists=0.10,
                        p_clean_sheet=0.25, p_yellow_card=0.08,
                        expected_bonus=0.12, lambda_goals_conceded=1.1),
        PlayerPrediction(element=12, position="MID", team="Aston Villa", opponent="Brighton", is_home=False,
                        p_no_play=0.15, p_sub=0.20, p_full=0.65,
                        lambda_goals=0.08, lambda_assists=0.05,
                        p_clean_sheet=0.18, p_yellow_card=0.12,
                        expected_bonus=0.05, lambda_goals_conceded=1.4),
        PlayerPrediction(element=13, position="FWD", team="Man City", opponent="Southampton", is_home=True,
                        p_no_play=0.03, p_sub=0.05, p_full=0.92,
                        lambda_goals=1.0, lambda_assists=0.2,
                        p_clean_sheet=0.15, p_yellow_card=0.08,
                        expected_bonus=0.45, lambda_goals_conceded=1.4),
        PlayerPrediction(element=14, position="FWD", team="Aston Villa", opponent="Brighton", is_home=False,
                        p_no_play=0.07, p_sub=0.10, p_full=0.83,
                        lambda_goals=0.35, lambda_assists=0.15,
                        p_clean_sheet=0.18, p_yellow_card=0.09,
                        expected_bonus=0.18, lambda_goals_conceded=1.4),
        PlayerPrediction(element=15, position="FWD", team="Brighton", opponent="Everton", is_home=True,
                        p_no_play=0.30, p_sub=0.25, p_full=0.45,
                        lambda_goals=0.15, lambda_assists=0.08,
                        p_clean_sheet=0.25, p_yellow_card=0.07,
                        expected_bonus=0.08, lambda_goals_conceded=1.3),
    ]

    names = {p[0]: p[1] for p in squad_info}
    prices = {p[0]: p[4] for p in squad_info}

    st.session_state["squad_predictions"] = predictions
    st.session_state["player_names"] = names
    st.session_state["player_prices"] = prices
    st.session_state["squad_budget"] = {"total": 1000, "spent": total_cost, "bank": bank}


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
        st.markdown("---")

    rows = []
    for p in preds:
        price = prices.get(p.element, 0)
        rows.append({
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
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", help="Player name"),
            "Pos": st.column_config.TextColumn("Pos", help="Position: GK, DEF, MID, or FWD"),
            "Team": st.column_config.TextColumn("Team", help="Player's club"),
            "Price": st.column_config.TextColumn("Price", help="Current price in millions (£). Total squad must be ≤ £100m"),
            "vs": st.column_config.TextColumn("vs", help="Opponent team this gameweek"),
            "Home": st.column_config.TextColumn("Home", help="🏠 = Home fixture, ✈️ = Away fixture. Home teams tend to score more"),
            "P(play)": st.column_config.TextColumn("P(play)", help="Probability the player features in this match (any minutes). Based on minutes model"),
            "λ Goals": st.column_config.TextColumn("λ Goals", help="Poisson rate for goals. λ=0.5 means ~39% chance of scoring. λ=1.0 means ~63% chance"),
            "λ Assists": st.column_config.TextColumn("λ Assists", help="Poisson rate for assists. Higher = more likely to assist. Depends on creativity and opponent"),
            "P(CS)": st.column_config.TextColumn("P(CS)", help="Probability of team keeping a clean sheet. Only gives FPL points if player plays 60+ mins. GK/DEF=4pts, MID=1pt"),
        },
    )


def _run_simulations(n_sims: int):
    """Run Monte Carlo simulations for all squad players."""
    preds = st.session_state["squad_predictions"]
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
    st.success(f"✅ Simulated {n_sims:,} outcomes for {len(preds)} players")


def _render_simulation_results():
    """Display simulation results."""
    results = st.session_state["sim_results"]
    names = st.session_state.get("player_names", {})

    rows = []
    for eid, r in results.items():
        rows.append({
            "Player": names.get(eid, f"#{eid}"),
            "xPts": f"{r['mean']:.2f}",
            "Std": f"{r['std']:.2f}",
            "P10": f"{r['p10']:.1f}",
            "Median": f"{r['median']:.1f}",
            "P90": f"{r['p90']:.1f}",
            "P(blank)": f"{r['p_blank']*100:.0f}%",
            "P(return)": f"{r['p_return']*100:.0f}%",
            "P(haul)": f"{r['p_haul']*100:.0f}%",
        })

    df = pd.DataFrame(rows).sort_values("xPts", ascending=False)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", help="Player name"),
            "xPts": st.column_config.TextColumn("xPts", help="Expected points — average across all simulations. The primary metric for player value this GW"),
            "Std": st.column_config.TextColumn("Std", help="Standard deviation — how much the outcome varies. High Std = unpredictable (could haul or blank)"),
            "P10": st.column_config.TextColumn("P10", help="10th percentile — in a bad GW, the player still scores at least this. The 'floor'"),
            "Median": st.column_config.TextColumn("Median", help="50th percentile — the most likely single outcome. Half the time they score above, half below"),
            "P90": st.column_config.TextColumn("P90", help="90th percentile — in a great GW, the player can reach this. The 'ceiling'"),
            "P(blank)": st.column_config.TextColumn("P(blank)", help="Probability of scoring ≤2 points (appearance only or didn't play). Lower is safer"),
            "P(return)": st.column_config.TextColumn("P(return)", help="Probability of scoring ≥5 points (likely got a goal, assist, or CS). Higher is better"),
            "P(haul)": st.column_config.TextColumn("P(haul)", help="Probability of scoring ≥10 points (multiple returns — goal+assist, brace, etc). Key for captaincy"),
        },
    )


def _render_captain_comparison():
    """Show captain comparison."""
    results = st.session_state["sim_results"]
    names = st.session_state.get("player_names", {})

    rows = []
    for eid, r in results.items():
        doubled = r["points"] * 2
        rows.append({
            "Player": names.get(eid, f"#{eid}"),
            "E[2×pts]": f"{doubled.mean():.1f}",
            "Std": f"{doubled.std():.1f}",
            "P(haul≥20)": f"{(doubled >= 20).mean()*100:.0f}%",
            "P(blank≤4)": f"{(doubled <= 4).mean()*100:.0f}%",
            "P90": f"{np.percentile(doubled, 90):.0f}",
            "_sort": doubled.mean(),
        })

    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns=["_sort"])
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", help="Player name — candidates are from your starting XI only"),
            "E[2×pts]": st.column_config.TextColumn("E[2×pts]", help="Expected DOUBLED points if captained. This is the primary metric — pick the highest"),
            "Std": st.column_config.TextColumn("Std", help="Volatility of doubled points. High Std = boom-or-bust captain. Good for chasing, risky for protecting rank"),
            "P(haul≥20)": st.column_config.TextColumn("P(haul≥20)", help="Probability of captain scoring 20+ doubled points (10+ actual). A massive haul. Best differential metric"),
            "P(blank≤4)": st.column_config.TextColumn("P(blank≤4)", help="Probability of captain scoring ≤4 doubled points (≤2 actual). The risk of a wasted armband. Lower is safer"),
            "P90": st.column_config.TextColumn("P90", help="90th percentile of doubled points — the ceiling if things go well. Higher = more explosive upside"),
        },
    )

    # Highlight recommendation
    best = max(results.items(), key=lambda x: x[1]["mean"])
    best_name = names.get(best[0], f"#{best[0]}")
    st.success(f"🎯 **Recommended Captain: {best_name}** (highest expected doubled points)")
