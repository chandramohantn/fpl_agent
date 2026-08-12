"""FPL Decision System — Streamlit Frontend.

Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

import streamlit as st

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(
    page_title="FPL Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar Navigation ─────────────────────────────────────────────────────

st.sidebar.title("⚽ FPL Engine")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "👥 Squad", "📝 Manual Inputs", "📋 Planning", "🔧 Model Management"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Quick Actions:**
    - `python scripts/refresh.py` to update data
    - `python scripts/run_pipeline.py` for full ingestion
    """
)

# ─── Page Router ─────────────────────────────────────────────────────────────

if page == "🏠 Dashboard":
    from pages import dashboard
    dashboard.render()
elif page == "👥 Squad":
    from pages import squad_management
    squad_management.render()
elif page == "📝 Manual Inputs":
    from pages import manual_inputs
    manual_inputs.render()
elif page == "📋 Planning":
    from pages import planning
    planning.render()
elif page == "🔧 Model Management":
    from pages import model_management
    model_management.render()
