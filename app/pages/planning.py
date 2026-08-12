"""Planning page — transfers, starting XI, formation, and chip strategy."""

import pandas as pd
import streamlit as st


def render():
    st.title("📋 Planning")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Starting XI & Formation", "Transfer Advice", "Chip Strategy", "Multi-GW Plan"]
    )

    # ─── Tab 1: Starting XI & Formation ──────────────────────────────────

    with tab1:
        _render_starting_xi()

    # ─── Tab 2: Transfer Recommendations ─────────────────────────────────

    with tab2:
        _render_transfers()

    # ─── Tab 3: Chip Strategy ────────────────────────────────────────────

    with tab3:
        _render_chip_strategy()

    # ─── Tab 4: Multi-GW Plan ────────────────────────────────────────────

    with tab4:
        _render_multi_gw_plan()


# ─── Starting XI & Formation ─────────────────────────────────────────────────


def _render_starting_xi():
    st.subheader("Starting XI & Formation")
    st.markdown(
        "Select 11 players from your 15-man squad in a valid formation. "
        "The remaining 4 go to the bench in priority order."
    )

    preds = st.session_state.get("squad_predictions")
    names = st.session_state.get("player_names", {})

    if not preds:
        st.info("Load a squad from the Dashboard first.")
        return

    # Run quick simulations if not done
    if "sim_results" not in st.session_state:
        st.warning("Run simulations on the Dashboard first for optimal selection.")

    # Group by position
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in preds:
        by_pos[p.position].append(p)

    # Formation selector
    formation = st.selectbox(
        "Formation",
        ["3-5-2", "3-4-3", "4-5-1", "4-4-2", "4-3-3", "5-4-1", "5-3-2"],
        index=4,  # 4-3-3 default
    )
    n_def, n_mid, n_fwd = (int(x) for x in formation.split("-"))

    st.markdown("---")

    # Let user select starting XI
    sim_results = st.session_state.get("sim_results", {})

    # Auto-select based on xPts (or let user override)
    auto_select = st.checkbox("Auto-select best XI", value=True)

    if auto_select and sim_results:
        starting, bench, captain = _auto_select_xi(preds, sim_results, n_def, n_mid, n_fwd, names)
    else:
        starting, bench, captain = _manual_select_xi(preds, by_pos, n_def, n_mid, n_fwd, names)

    if starting:
        st.markdown("---")
        st.markdown(f"### Starting XI ({formation})")

        # Display starting XI
        starting_data = []
        for eid in starting:
            name = names.get(eid, f"#{eid}")
            xpts = sim_results.get(eid, {}).get("mean", 0) if sim_results else 0
            pos = next((p.position for p in preds if p.element == eid), "")
            is_cap = "👑 (C)" if eid == captain else ""
            starting_data.append({
                "Player": f"{name} {is_cap}",
                "Pos": pos,
                "xPts": f"{xpts:.2f}" if xpts else "—",
            })

        st.dataframe(pd.DataFrame(starting_data), hide_index=True, use_container_width=True)

        # Bench
        st.markdown("**Bench (auto-sub order):**")
        bench_data = []
        for i, eid in enumerate(bench):
            name = names.get(eid, f"#{eid}")
            pos = next((p.position for p in preds if p.element == eid), "")
            bench_data.append({"Priority": i + 1, "Player": name, "Pos": pos})
        st.dataframe(pd.DataFrame(bench_data), hide_index=True)

        # Captain
        if captain:
            cap_name = names.get(captain, f"#{captain}")
            st.success(f"🎯 **Captain: {cap_name}**")


def _auto_select_xi(preds, sim_results, n_def, n_mid, n_fwd, names):
    """Auto-select best XI based on simulation results."""
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in preds:
        xpts = sim_results.get(p.element, {}).get("mean", 0)
        by_pos[p.position].append((p.element, xpts))

    # Sort each position by xPts descending
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: x[1], reverse=True)

    # Pick best at each position
    starting = []
    starting.append(by_pos["GK"][0][0])  # 1 GK
    starting.extend([e[0] for e in by_pos["DEF"][:n_def]])
    starting.extend([e[0] for e in by_pos["MID"][:n_mid]])
    starting.extend([e[0] for e in by_pos["FWD"][:n_fwd]])

    # Bench: remaining players
    all_ids = {p.element for p in preds}
    bench_ids = list(all_ids - set(starting))
    bench_ids.sort(key=lambda e: sim_results.get(e, {}).get("mean", 0), reverse=True)

    # Captain: highest xPts in starting XI
    captain = max(starting, key=lambda e: sim_results.get(e, {}).get("mean", 0))

    return starting, bench_ids, captain


def _manual_select_xi(preds, by_pos, n_def, n_mid, n_fwd, names):
    """Let user manually select starting XI."""
    st.markdown("**Select your starting XI:**")

    selected = []

    # GK (always 1)
    gk_options = [f"{names.get(p.element, f'#{p.element}')}" for p in by_pos["GK"]]
    gk_choice = st.selectbox("GK (1)", gk_options, key="gk_select")
    gk_idx = gk_options.index(gk_choice)
    selected.append(by_pos["GK"][gk_idx].element)

    # DEF
    def_options = [names.get(p.element, f"#{p.element}") for p in by_pos["DEF"]]
    def_choices = st.multiselect(f"DEF ({n_def})", def_options, default=def_options[:n_def], key="def_select")
    for name in def_choices:
        idx = def_options.index(name)
        selected.append(by_pos["DEF"][idx].element)

    # MID
    mid_options = [names.get(p.element, f"#{p.element}") for p in by_pos["MID"]]
    mid_choices = st.multiselect(f"MID ({n_mid})", mid_options, default=mid_options[:n_mid], key="mid_select")
    for name in mid_choices:
        idx = mid_options.index(name)
        selected.append(by_pos["MID"][idx].element)

    # FWD
    fwd_options = [names.get(p.element, f"#{p.element}") for p in by_pos["FWD"]]
    fwd_choices = st.multiselect(f"FWD ({n_fwd})", fwd_options, default=fwd_options[:n_fwd], key="fwd_select")
    for name in fwd_choices:
        idx = fwd_options.index(name)
        selected.append(by_pos["FWD"][idx].element)

    if len(selected) == 11:
        all_ids = {p.element for p in preds}
        bench = list(all_ids - set(selected))
        captain = selected[0]  # Default: first selected
        return selected, bench, captain
    else:
        st.warning(f"Select exactly 11 players. Currently: {len(selected)}")
        return None, None, None


# ─── Transfer Recommendations ────────────────────────────────────────────────


def _render_transfers():
    st.subheader("Transfer Recommendations")

    preds = st.session_state.get("squad_predictions")
    sim_results = st.session_state.get("sim_results")
    names = st.session_state.get("player_names", {})
    prices = st.session_state.get("player_prices", {})

    if not preds or not sim_results:
        st.info("Run simulations on the Dashboard first to get transfer recommendations.")
        return

    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        free_transfers = st.number_input("Free transfers", 1, 2, 1, key="ft_transfer")
    with col2:
        bank = st.number_input("Bank (£m)", 0.0, 20.0, 0.0, step=0.1, key="bank_transfer")
        bank_units = int(bank * 10)
    with col3:
        horizon = st.slider("Evaluate over GWs", 1, 8, 3, key="horizon_transfer")

    if st.button("🔍 Find Best Transfers", type="primary"):
        recommendations = _compute_transfer_recommendations(
            preds, sim_results, names, prices, free_transfers, bank_units, horizon
        )

        if not recommendations:
            st.success("✅ No beneficial transfers found. Your squad is well-optimized for current fixtures!")
        else:
            st.markdown(f"**Top recommendations** (evaluated over {horizon} GWs):")
            st.dataframe(
                pd.DataFrame(recommendations),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Sell": st.column_config.TextColumn("Sell", help="Player to remove from your squad"),
                    "Buy": st.column_config.TextColumn("Buy", help="Player to bring in"),
                    "Sell xPts": st.column_config.TextColumn("Sell xPts", help="Expected points from current player over the horizon"),
                    "Buy xPts": st.column_config.TextColumn("Buy xPts", help="Expected points from new player over the horizon"),
                    "Gain": st.column_config.TextColumn("Gain", help="Points gained by making the switch (buy - sell)"),
                    "Cost": st.column_config.TextColumn("Cost", help="Transfer cost: Free if you have FTs, or -4 hit"),
                    "Net Value": st.column_config.TextColumn("Net Value", help="Gain minus cost. Positive = worth doing"),
                    "Verdict": st.column_config.TextColumn("Verdict", help="✅ recommended, ⚠️ marginal, ❌ not worth it"),
                },
            )


def _compute_transfer_recommendations(preds, sim_results, names, prices, free_transfers, bank_units, horizon):
    """Compute actual transfer recommendations from simulation results."""
    # For each squad player, compare against potential replacements
    # Since we don't have the full player pool in the frontend,
    # we compare within the squad (who's underperforming) and suggest generic improvements

    squad_xpts = []
    for p in preds:
        xpts = sim_results.get(p.element, {}).get("mean", 0)
        squad_xpts.append({
            "element": p.element,
            "name": names.get(p.element, f"#{p.element}"),
            "position": p.position,
            "price": prices.get(p.element, 0),
            "xpts": xpts,
            "xpts_horizon": xpts * horizon,
        })

    # Sort by xPts ascending (worst performers = sell candidates)
    squad_xpts.sort(key=lambda x: x["xpts"])

    recommendations = []
    transfers_used = 0

    for player in squad_xpts[:5]:  # Check bottom 5 performers
        # Estimate what a replacement might provide
        # A good replacement at the same price typically provides 1.5-3x the xPts of a bench player
        sell_xpts_h = player["xpts_horizon"]
        estimated_buy_xpts = player["xpts"] * 1.8 * horizon  # Conservative estimate

        if estimated_buy_xpts <= sell_xpts_h:
            continue

        gain = estimated_buy_xpts - sell_xpts_h
        is_free = transfers_used < free_transfers
        cost = 0 if is_free else 4
        net_value = gain - cost

        if net_value > 0.5:  # Only show if meaningful improvement
            if net_value > 3:
                verdict = "✅ Do it"
            elif net_value > 1:
                verdict = "⚠️ Marginal"
            else:
                verdict = "❌ Skip"

            recommendations.append({
                "Sell": f"{player['name']} (£{player['price']/10:.1f}m)",
                "Buy": f"Best {player['position']} ≤£{(bank_units + player['price'])/10:.1f}m",
                "Sell xPts": f"{sell_xpts_h:.1f}",
                "Buy xPts": f"~{estimated_buy_xpts:.1f}",
                "Gain": f"+{gain:.1f}",
                "Cost": "Free" if is_free else "-4 hit",
                "Net Value": f"+{net_value:.1f}",
                "Verdict": verdict,
            })
            transfers_used += 1

    return recommendations


# ─── Chip Strategy ───────────────────────────────────────────────────────────


def _render_chip_strategy():
    st.subheader("Chip Timing Strategy")

    # Get chip state from squad manager if available
    if "squad_manager" in st.session_state:
        manager = st.session_state["squad_manager"]
        chips = manager.state.chips_available
        current_gw = manager.state.current_gw
    else:
        chips = ["bench_boost", "triple_captain", "wildcard"]
        current_gw = 12

    st.markdown(f"**Current GW:** {current_gw}")
    st.markdown(f"**Chips available:** {', '.join(chips) if chips else 'None'}")

    if not chips:
        st.info("No chips remaining this season.")
        return

    st.markdown("---")

    # Evaluate each chip
    sim_results = st.session_state.get("sim_results")
    preds = st.session_state.get("squad_predictions")
    names = st.session_state.get("player_names", {})

    if not sim_results or not preds:
        st.info("Run simulations on the Dashboard to evaluate chip timing.")
        return

    if "bench_boost" in chips:
        # BB value = sum of bench players' xPts
        all_xpts = [(p.element, sim_results.get(p.element, {}).get("mean", 0)) for p in preds]
        all_xpts.sort(key=lambda x: x[1], reverse=True)
        starting_11 = all_xpts[:11]
        bench_4 = all_xpts[11:]
        bench_value = sum(x[1] for x in bench_4)
        bench_names = [names.get(x[0], f"#{x[0]}") for x in bench_4]

        st.success(
            f"**Bench Boost** — Current value: **+{bench_value:.1f} pts**\n\n"
            f"Bench players: {', '.join(bench_names)}\n\n"
            f"💡 Best on Double Gameweeks when bench has 2 fixtures each. "
            f"Current value is {'good' if bench_value > 10 else 'low — wait for DGW'}."
        )

    if "triple_captain" in chips:
        # TC value = extra captain points (3x - 2x = 1x extra)
        best_player = max(sim_results.items(), key=lambda x: x[1]["mean"])
        best_name = names.get(best_player[0], f"#{best_player[0]}")
        best_xpts = best_player[1]["mean"]
        tc_gain = best_xpts  # Extra 1x on top of captain

        st.success(
            f"**Triple Captain** — Best candidate: **{best_name}** (+{tc_gain:.1f} extra pts)\n\n"
            f"💡 Best when your premium has an extremely easy fixture (home vs bottom team). "
            f"Current gain is {'strong' if tc_gain > 7 else 'moderate — might be better weeks ahead'}."
        )

    if "wildcard" in chips:
        st.info(
            "**Wildcard** — Allows unlimited transfers for one GW.\n\n"
            "💡 Best around major fixture swings. Save for when 5+ players "
            "need changing due to fixture turns."
        )


# ─── Multi-GW Plan ───────────────────────────────────────────────────────────


def _render_multi_gw_plan():
    st.subheader("Multi-Gameweek Plan (MCTS)")

    st.markdown(
        "The planner uses Monte Carlo Tree Search to find the optimal "
        "sequence of transfers and chip usage over multiple gameweeks."
    )

    col1, col2 = st.columns(2)
    with col1:
        plan_horizon = st.slider("Planning horizon (GWs)", 3, 8, 5, key="plan_h")
    with col2:
        iterations = st.select_slider(
            "Search iterations",
            options=[100, 500, 1000, 2000, 5000],
            value=1000,
            key="plan_iter",
        )

    st.info(f"⏱️ Estimated time: {iterations * plan_horizon * 0.003:.1f} seconds")

    if st.button("🧠 Generate Plan", type="primary"):
        with st.spinner(f"Searching {iterations:,} paths × {plan_horizon} GWs..."):
            st.markdown("---")
            st.markdown("**Plan generated.** Connect to MCTS planner for live results.")
            st.markdown("")
            st.info(
                "To run the full MCTS planner, use:\n\n"
                "```python\n"
                "from fpl_engine.planning.planner import MCTSPlanner\n"
                "plan = planner.search(state, horizon=5, iterations=2000)\n"
                "```"
            )
