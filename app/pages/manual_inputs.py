"""Manual Inputs page — PlayerContext overrides."""

import streamlit as st
from fpl_engine.models.player_context import AvailabilityStatus, PlayerContext


def render():
    st.title("📝 Manual Inputs")
    st.markdown(
        "Override player context with information the system can't get from APIs: "
        "midweek matches, press conference hints, fitness assessments."
    )

    tab1, tab2 = st.tabs(["Individual Player", "View Overrides"])

    # ─── Tab 1: Set individual player context ────────────────────────────

    with tab1:
        st.subheader("Set Player Context")

        # Initialize overrides in session state
        if "player_contexts" not in st.session_state:
            st.session_state["player_contexts"] = {}

        names = st.session_state.get("player_names", {})
        if not names:
            st.warning("Load a squad from the Dashboard first.")
            return

        # Player selector
        player_options = {f"{name} (#{eid})": eid for eid, name in names.items()}
        selected = st.selectbox("Select Player", options=list(player_options.keys()))
        selected_id = player_options[selected]

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Availability / Injury**")

            status = st.selectbox(
                "Status",
                ["available", "doubtful", "injured", "suspended", "unknown"],
                index=0,
            )
            status_map = {
                "available": AvailabilityStatus.AVAILABLE,
                "doubtful": AvailabilityStatus.DOUBTFUL,
                "injured": AvailabilityStatus.INJURED,
                "suspended": AvailabilityStatus.SUSPENDED,
                "unknown": AvailabilityStatus.UNKNOWN,
            }

            chance_of_playing = st.slider(
                "Chance of playing (%)", 0, 100, 100,
                help="0 = definitely out, 100 = definitely playing"
            )

            returning_from_injury = st.checkbox("Returning from injury")

            injury_weeks = 0.0
            fitness = None
            if returning_from_injury:
                injury_weeks = st.number_input(
                    "Weeks out", min_value=0.0, max_value=30.0, value=2.0, step=0.5
                )
                fitness = st.slider(
                    "Fitness level", 0.0, 1.0, 0.7,
                    help="0.0 = unfit, 1.0 = fully match fit"
                )

        with col2:
            st.markdown("**External Match Context**")

            days_since = st.number_input(
                "Days since last match (any competition)",
                min_value=0.0, max_value=14.0, value=7.0, step=0.5,
                help="Include midweek CL/cup/international matches"
            )

            minutes_last = st.number_input(
                "Minutes played in last match",
                min_value=0, max_value=120, value=0, step=15,
                help="0 = didn't play, 90 = full match"
            )

            important_match_days = st.number_input(
                "Important match in how many days?",
                min_value=0.0, max_value=14.0, value=0.0, step=1.0,
                help="0 = no upcoming important match"
            )

            important_match_type = ""
            if important_match_days > 0:
                important_match_type = st.text_input(
                    "Match type", placeholder="e.g., Champions League Semi-Final"
                )

        st.markdown("---")

        if st.button("💾 Save Override", type="primary"):
            ctx = PlayerContext(
                player_id=selected_id,
                chance_of_playing=chance_of_playing if chance_of_playing < 100 else None,
                status=status_map[status],
                returning_from_injury=returning_from_injury,
                injury_duration_weeks=injury_weeks,
                fitness_level=fitness,
                days_since_last_match=days_since if minutes_last > 0 else None,
                played_minutes_last_match=minutes_last if minutes_last > 0 else None,
                important_match_in_days=important_match_days if important_match_days > 0 else None,
                important_match_type=important_match_type,
                source="manual",
            )
            st.session_state["player_contexts"][selected_id] = ctx
            st.success(f"✅ Context saved for {names[selected_id]}")

    # ─── Tab 2: View all overrides ───────────────────────────────────────

    with tab2:
        st.subheader("Active Overrides")

        contexts = st.session_state.get("player_contexts", {})
        if not contexts:
            st.info("No overrides set yet. Use the form above to add player context.")
        else:
            for eid, ctx in contexts.items():
                name = names.get(eid, f"#{eid}")
                with st.expander(f"**{name}** — {ctx.status.value}"):
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Chance of playing", f"{ctx.chance_of_playing or 100}%")
                        st.metric("Status", ctx.status.value)
                    with cols[1]:
                        st.metric("Days since match", ctx.days_since_last_match or "N/A")
                        st.metric("Mins last match", ctx.played_minutes_last_match or "N/A")
                    with cols[2]:
                        st.metric("Returning?", "Yes" if ctx.returning_from_injury else "No")
                        if ctx.important_match_in_days:
                            st.metric("Important match", f"In {ctx.important_match_in_days}d")

                    if st.button("🗑️ Remove", key=f"remove_{eid}"):
                        del st.session_state["player_contexts"][eid]
                        st.rerun()
