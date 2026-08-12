"""Squad Management page — view, modify, and persist your squad."""

import streamlit as st
from fpl_engine.squad.manager import SquadManager


def render():
    st.title("👥 Squad Management")

    # Initialize manager
    if "squad_manager" not in st.session_state:
        manager = SquadManager()
        if manager.exists():
            manager.load()
        st.session_state["squad_manager"] = manager

    manager: SquadManager = st.session_state["squad_manager"]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["My Squad", "Make Transfer", "Advance Gameweek", "Transfer History"]
    )

    # ─── Tab 1: View Squad ───────────────────────────────────────────────

    with tab1:
        if not manager.state.squad:
            st.warning("No squad saved. Set up your squad below or load from the Dashboard.")

            if st.button("📥 Initialize from Dashboard Squad"):
                _init_from_dashboard(manager)
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Gameweek", f"GW{manager.state.current_gw}")
            with col2:
                st.metric("Bank", f"£{manager.state.bank/10:.1f}m")
            with col3:
                st.metric("Free Transfers", manager.state.free_transfers)
            with col4:
                chips = manager.state.chips_available
                st.metric("Chips Left", len(chips))

            st.markdown("---")

            # Squad by position
            by_pos = manager.get_squad_by_position()
            for pos in ["GK", "DEF", "MID", "FWD"]:
                players = by_pos.get(pos, [])
                if players:
                    st.markdown(f"**{pos}** ({len(players)})")
                    cols = st.columns(len(players))
                    for i, p in enumerate(players):
                        with cols[i]:
                            st.markdown(
                                f"**{p['name']}**  \n"
                                f"£{p['price']/10:.1f}m  \n"
                                f"_{p['team']}_"
                            )

            st.markdown("---")
            st.markdown(f"**Chips available:** {', '.join(manager.state.chips_available) or 'None'}")

    # ─── Tab 2: Make Transfer ────────────────────────────────────────────

    with tab2:
        if not manager.state.squad:
            st.info("Initialize your squad first.")
            return

        st.subheader("Execute Transfer")
        st.markdown(
            f"**Free transfers available: {manager.state.free_transfers}** | "
            f"Bank: £{manager.state.bank/10:.1f}m"
        )

        if manager.state.free_transfers == 0:
            st.warning("⚠️ No free transfers! Making a transfer will cost -4 points.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Sell:**")
            # Build sell options
            sell_options = {}
            for eid in manager.state.squad:
                eid_str = str(eid)
                name = manager.state.player_names.get(eid_str, f"#{eid}")
                price = manager.state.player_prices.get(eid_str, 0)
                pos = manager.state.player_positions.get(eid_str, "")
                sell_options[f"{name} ({pos}, £{price/10:.1f}m)"] = eid

            sell_choice = st.selectbox("Select player to sell", list(sell_options.keys()))
            sell_id = sell_options[sell_choice] if sell_choice else None

        with col2:
            st.markdown("**Buy:**")
            buy_name = st.text_input("Player name", placeholder="e.g., Son")
            buy_price = st.number_input("Price (£m)", 4.0, 15.0, 7.0, step=0.1)
            buy_price_units = int(buy_price * 10)

            # Position (must match sell)
            if sell_id:
                sell_pos = manager.state.player_positions.get(str(sell_id), "")
                st.info(f"Position: **{sell_pos}** (must match)")
                buy_position = sell_pos
            else:
                buy_position = st.selectbox("Position", ["GK", "DEF", "MID", "FWD"])

            buy_team = st.text_input("Team", placeholder="e.g., Spurs")
            buy_id = st.number_input("FPL Player ID", min_value=1, value=100)

        # Budget preview
        if sell_id:
            sell_price = manager.state.player_prices.get(str(sell_id), 0)
            available = manager.state.bank + sell_price
            st.markdown(
                f"**Budget available:** £{available/10:.1f}m "
                f"(bank £{manager.state.bank/10:.1f}m + sell £{sell_price/10:.1f}m)"
            )
            if buy_price_units > available:
                st.error(f"❌ Cannot afford! Need £{buy_price_units/10:.1f}m, have £{available/10:.1f}m")

        st.markdown("---")

        if st.button("✅ Confirm Transfer", type="primary"):
            if not sell_id or not buy_name:
                st.error("Please select a player to sell and enter buy details.")
            else:
                result = manager.execute_transfer(
                    sell_id=sell_id,
                    buy_id=buy_id,
                    buy_price=buy_price_units,
                    buy_name=buy_name,
                    buy_position=buy_position,
                    buy_team=buy_team,
                )
                if result.success:
                    manager.save()
                    st.success(result.message)
                    st.rerun()
                else:
                    st.error(result.message)

    # ─── Tab 3: Advance Gameweek ─────────────────────────────────────────

    with tab3:
        st.subheader("Advance to Next Gameweek")
        st.markdown(
            f"Current: **GW{manager.state.current_gw}** | "
            f"Free transfers: **{manager.state.free_transfers}**"
        )

        st.markdown(
            "Click below after the GW deadline passes. This will:\n"
            "- Move to the next gameweek\n"
            "- Roll your free transfer (if unused, max 2)\n"
        )

        if st.button("⏭️ Advance to Next GW", type="primary"):
            manager.advance_gameweek()
            manager.save()
            st.success(
                f"Advanced to GW{manager.state.current_gw}. "
                f"Free transfers: {manager.state.free_transfers}"
            )
            st.rerun()

        st.markdown("---")
        st.subheader("Use Chip")
        if manager.state.chips_available:
            chip_choice = st.selectbox(
                "Select chip to use",
                manager.state.chips_available,
            )
            if st.button(f"🃏 Use {chip_choice}"):
                if manager.use_chip(chip_choice):
                    manager.save()
                    st.success(f"✅ {chip_choice} activated for GW{manager.state.current_gw}")
                    st.rerun()
        else:
            st.info("No chips remaining.")

    # ─── Tab 4: Transfer History ─────────────────────────────────────────

    with tab4:
        st.subheader("Transfer History")

        history = manager.get_transfer_history()
        if not history:
            st.info("No transfers made yet.")
        else:
            for record in reversed(history):
                was_free = "Free" if record.get("was_free") else "-4 hit"
                st.markdown(
                    f"**GW{record['gameweek']}:** "
                    f"{record['sell_name']} (£{record['sell_price']/10:.1f}m) → "
                    f"{record['buy_name']} (£{record['buy_price']/10:.1f}m) "
                    f"[{was_free}]"
                )

        # Chip history
        chip_history = manager.state.chip_history
        if chip_history:
            st.markdown("---")
            st.subheader("Chip History")
            for record in chip_history:
                st.markdown(f"**GW{record['gameweek']}:** {record['chip']}")


def _init_from_dashboard(manager: SquadManager):
    """Initialize squad from dashboard session data."""
    preds = st.session_state.get("squad_predictions")
    names = st.session_state.get("player_names", {})
    prices = st.session_state.get("player_prices", {})
    budget = st.session_state.get("squad_budget", {})

    if not preds or not prices:
        st.error("No squad data in Dashboard. Load a squad there first.")
        return

    squad = [p.element for p in preds]
    positions = {p.element: p.position for p in preds}
    teams = {p.element: p.team for p in preds}
    bank = budget.get("bank", 0) if budget else 0

    try:
        manager.initialize_squad(
            squad=squad,
            names=names,
            prices=prices,
            positions=positions,
            teams=teams,
            bank=bank,
        )
        manager.save()
        st.success(f"✅ Squad saved! {len(squad)} players, bank £{bank/10:.1f}m")
        st.rerun()
    except AssertionError as e:
        st.error(f"Invalid squad: {e}")
