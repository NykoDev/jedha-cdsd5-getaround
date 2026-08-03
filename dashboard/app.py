import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px

from common.delay_analysis import load_delay_data, build_chained, load_price_proxy, run_simulation_grid, simulate

st.set_page_config(page_title="GetAround - Délai minimum entre locations", layout="wide")


@st.cache_data
def get_data():
    dfda = load_delay_data()
    chained = build_chained(dfda)
    median_price_per_day = load_price_proxy()
    grid = run_simulation_grid(dfda, chained, median_price_per_day)
    return dfda, chained, median_price_per_day, grid


dfda, chained, median_price_per_day, grid = get_data()

st.title("GetAround — Délai minimum entre deux locations")
st.markdown(
    "Aide à la décision pour le choix d'un **délai minimum** entre deux locations d'un même véhicule, "
    "et du **périmètre** de véhicules concerné (tout le parc ou Connect uniquement)."
)

st.sidebar.header("Paramètres")
threshold = st.sidebar.slider(
    "Seuil minimum entre 2 locations (minutes)",
    min_value=0, max_value=720, step=30, value=120,
)
scope_label = st.sidebar.selectbox("Périmètre des véhicules", ["Toutes les voitures", "Connect uniquement"])
scope = "all" if scope_label == "Toutes les voitures" else "connect_only"

result = simulate(dfda, chained, median_price_per_day, threshold, scope)

col1, col2, col3 = st.columns(3)
col1.metric(
    "Locations affectées",
    f"{result['pct_affected']:.1f} %",
    f"{result['n_affected']} locations",
)
col2.metric(
    "Revenu estimé à risque",
    f"{result['pct_revenue_at_risk']:.1f} %",
    f"{result['revenue_at_risk']:,.0f} € (proxy)",
)
col3.metric(
    "Cas problématiques résolus",
    f"{result['pct_problematic_resolved']:.1f} %",
    f"{result['n_resolved']} / {result['n_problematic']}",
)

st.subheader("% de locations affectées selon le seuil")
fig_affected = px.line(grid, x="threshold", y="pct_affected", color="scope", markers=True)
fig_affected.add_vline(x=threshold, line_dash="dash", line_color="red")
st.plotly_chart(fig_affected, use_container_width=True)

st.subheader("Cas problématiques résolus selon le seuil")
fig_resolved = px.line(grid, x="threshold", y="n_resolved", color="scope", markers=True)
fig_resolved.add_vline(x=threshold, line_dash="dash", line_color="red")
st.plotly_chart(fig_resolved, use_container_width=True)
