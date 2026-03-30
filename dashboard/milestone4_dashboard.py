"""
Milestone 4 — Forecast Integration & Capacity Planning Dashboard
Azure Resource Demand Forecasting with XGBoost
Run: streamlit run milestone4_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Azure Capacity Intelligence",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;s
}

html, body, .stApp {
    background-color: #111827 !important;
    color: #e2e8f0;
}

/* Main container override */
[data-testid="stAppViewContainer"] {
    background-color: #111827 !important;
}

/* Block container (main content area) */
[data-testid="stAppViewContainer"] > .main {
    background-color: #111827 !important;
}

/* Extra safety */
section.main {
    background-color: #111827 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

/* KPI Cards (modern glass look) */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: #3b82f6;
}

/* Accent bar */
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--accent);
    border-radius: 14px 0 0 14px;
}

/* KPI Text */
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1;
}
.kpi-delta {
    font-size: 12px;
    margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.delta-up { color: #f87171; }
.delta-down { color: #34d399; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #60a5fa;
    border-bottom: 1px solid #334155;
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8 !important;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #60a5fa !important;
}

/* Alert boxes */
.alert-critical {
    background: #1f0a0a;
    border-left: 4px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13px;
}
.alert-warn {
    background: #1f1400;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13px;
}

/* Model box */
.model-box {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #cbd5f5;
}
.model-title {
    font-size: 14px;
    font-weight: 600;
    color: #60a5fa;
    margin-bottom: 12px;
}

/* Divider */
hr {
    border-color: #334155;
}

/* Hide branding */
#MainMenu, footer, header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path="processed_data.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = pd.to_datetime(df["timestamp"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def load_forecast(path="../data/forecast_output.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["date"] = pd.to_datetime(df["timestamp"])
    return df


try:
    forecast_df = load_forecast("../data/forecast_output.csv")
    FORECAST_AVAILABLE = True
except:
    FORECAST_AVAILABLE = False

# Try loading; fall back to synthetic demo if file missing
try:
    df = load_data("processed_data.csv")
    DATA_LOADED = True
except FileNotFoundError:
    st.warning(
        "⚠️ processed_data.csv not found in the working directory. Place the CSV alongside this script.")
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🔷 AZURE DASHBOARD")
st.sidebar.markdown("---")
all_regions = sorted(df["region"].unique())
all_services = sorted(df["service_type"].unique())
all_years = sorted(df["year"].unique())

selected_regions = st.sidebar.multiselect(
    "Regions", all_regions, default=all_regions[:6],
    help="Filter by Azure region"
)
selected_services = st.sidebar.multiselect(
    "Service Type", all_services, default=all_services
)
selected_years = st.sidebar.multiselect(
    "Year", all_years, default=all_years
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Capacity Risk Threshold**")
risk_threshold = st.sidebar.slider(
    "Utilization % alert level", 0.5, 1.0, 0.85, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span style='font-family:IBM Plex Mono;font-size:11px;color:#4a6080'>"
    "MILESTONE 4 · FORECAST INTEGRATION<br>& CAPACITY PLANNING</span>",
    unsafe_allow_html=True
)

# Apply filters
mask = (
    df["region"].isin(selected_regions) &
    df["service_type"].isin(selected_services) &
    df["year"].isin(selected_years)
)
dff = df[mask].copy()

if dff.empty:
    st.error("No data for selected filters. Adjust the sidebar.")
    st.stop()


# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#111827",
    font=dict(family="IBM Plex Mono", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#334155", linecolor="#334155"),
    yaxis=dict(gridcolor="#334155", linecolor="#334155"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="#111827", bordercolor="#334155", borderwidth=1)
)
COLORS = ["#3b82f6", "#34d399", "#f59e0b", "#f87171", "#a78bfa",
          "#22d3ee", "#fb923c", "#e879f9", "#4ade80", "#facc15"]


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:IBM Plex Mono;font-size:24px;color:#60a5fa;"
    "letter-spacing:2px;margin-bottom:4px'>AZURE CAPACITY INTELLIGENCE</h1>"
    "<p style='font-family:IBM Plex Sans;font-size:14px;color:#4a6888;"
    "margin-bottom:24px'>Milestone 4 · Forecast Integration & Capacity Planning Dashboard</p>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Overview",
    "📈 Demand Trends",
    "🌍 Regional Analysis",
    "🤖 Model & Forecast",
    "⚠️ Risk Alerts"
])


# ══════════════════════════════════════════════
# TAB 1 — KPI OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>EXECUTIVE KPIs</div>",
                unsafe_allow_html=True)

    total_cost = dff["cost_usd"].sum()
    wasted_cost = dff["wasted_capacity_cost"].sum()
    waste_pct = (wasted_cost / total_cost * 100) if total_cost > 0 else 0
    avg_util = dff["utilization_pct"].mean() * 100
    risk_count = dff["capacity_risk_flag"].sum()
    underutil_count = dff["underutilized_flag"].sum()
    total_incidents = dff["incident_count"].sum()
    avg_mttr = dff[dff["mttr_minutes"] > 0]["mttr_minutes"].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#3b82f6'>
            <div class='kpi-label'>Total Cost (USD)</div>
            <div class='kpi-value'>${total_cost/1e6:.2f}M</div>
            <div class='kpi-delta delta-down'>Filtered period</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#ef4444'>
            <div class='kpi-label'>Wasted Capacity Cost</div>
            <div class='kpi-value'>${wasted_cost/1e6:.2f}M</div>
            <div class='kpi-delta delta-up'>▲ {waste_pct:.1f}% of total spend</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        color = "#34d399" if avg_util < 70 else "#f59e0b" if avg_util < 85 else "#ef4444"
        st.markdown(f"""
        <div class='kpi-card' style='--accent:{color}'>
            <div class='kpi-label'>Avg Utilization</div>
            <div class='kpi-value'>{avg_util:.1f}%</div>
            <div class='kpi-delta' style='color:{color}'>Across all services</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#f59e0b'>
            <div class='kpi-label'>Total Incidents</div>
            <div class='kpi-value'>{int(total_incidents):,}</div>
            <div class='kpi-delta delta-up'>Avg MTTR: {avg_mttr:.0f} min</div>
        </div>""", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#ef4444'>
            <div class='kpi-label'>Capacity Risk Events</div>
            <div class='kpi-value'>{int(risk_count):,}</div>
            <div class='kpi-delta delta-up'>{risk_count/len(dff)*100:.1f}% of records</div>
        </div>""", unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#a78bfa'>
            <div class='kpi-label'>Underutilized Flags</div>
            <div class='kpi-value'>{int(underutil_count):,}</div>
            <div class='kpi-delta delta-down'>{underutil_count/len(dff)*100:.1f}% of records</div>
        </div>""", unsafe_allow_html=True)

    with c7:
        avg_headroom = dff["headroom_units"].mean()
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#22d3ee'>
            <div class='kpi-label'>Avg Headroom (Units)</div>
            <div class='kpi-value'>{avg_headroom:,.0f}</div>
            <div class='kpi-delta delta-down'>Available buffer</div>
        </div>""", unsafe_allow_html=True)

    with c8:
        avg_growth = dff["daily_growth_rate"].mean() * 100
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#34d399'>
            <div class='kpi-label'>Avg Daily Growth Rate</div>
            <div class='kpi-value'>{avg_growth:.3f}%</div>
            <div class='kpi-delta delta-down'>Per day, all regions</div>
        </div>""", unsafe_allow_html=True)

    # Cost vs Waste donut
    st.markdown("<div class='section-header'>COST COMPOSITION</div>",
                unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])

    with col_a:
        actual_cost = total_cost - wasted_cost
        fig_pie = go.Figure(go.Pie(
            labels=["Utilized Spend", "Wasted Capacity"],
            values=[actual_cost, wasted_cost],
            hole=0.65,
            marker_colors=["#3b82f6", "#ef4444"],
            textfont=dict(family="IBM Plex Mono", size=11),
            hovertemplate="%{label}<br>$%{value:,.0f}<extra></extra>"
        ))
        fig_pie.add_annotation(
            text=f"${total_cost/1e6:.1f}M<br>Total",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="IBM Plex Mono", size=14, color="#e0eaff")
        )
        fig_pie.update_layout(**PLOTLY_LAYOUT, title="Cost Efficiency Breakdown",
                              showlegend=True, height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # Monthly cost trend
        monthly = dff.groupby("year_month").agg(
            cost=("cost_usd", "sum"),
            wasted=("wasted_capacity_cost", "sum")
        ).reset_index()
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(x=monthly["year_month"], y=monthly["cost"],
                                  name="Total Cost", marker_color="#3b82f6", opacity=0.8))
        fig_cost.add_trace(go.Bar(x=monthly["year_month"], y=monthly["wasted"],
                                  name="Wasted Cost", marker_color="#ef4444", opacity=0.8))
        fig_cost.update_layout(**PLOTLY_LAYOUT, title="Monthly Cost vs Wasted Capacity",
                               barmode="overlay", height=320,
                               xaxis_title="Month", yaxis_title="USD")
        st.plotly_chart(fig_cost, use_container_width=True)

    # Utilization distribution
    st.markdown("<div class='section-header'>UTILIZATION DISTRIBUTION BY SERVICE</div>",
                unsafe_allow_html=True)
    fig_hist = go.Figure()
    for i, svc in enumerate(dff["service_type"].unique()):
        sub = dff[dff["service_type"] == svc]["utilization_pct"] * 100
        fig_hist.add_trace(go.Histogram(x=sub, name=svc, opacity=0.75,
                                        marker_color=COLORS[i], nbinsx=40))
    fig_hist.add_vline(x=risk_threshold * 100, line_dash="dash",
                       line_color="#ef4444", annotation_text=f"Risk threshold ({risk_threshold*100:.0f}%)")
    fig_hist.update_layout(**PLOTLY_LAYOUT, title="Utilization % Distribution",
                           barmode="overlay", xaxis_title="Utilization (%)", height=300)
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — DEMAND TRENDS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>USAGE & DEMAND OVER TIME</div>",
                unsafe_allow_html=True)

    metric_choice = st.selectbox(
        "Primary Metric", ["usage_units", "utilization_pct", "cost_usd",
                           "headroom_units", "wasted_capacity_cost"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    group_by = st.radio(
        "Group by", ["service_type", "region"], horizontal=True)

    agg_df = dff.groupby(["year_month", group_by])[
        metric_choice].mean().reset_index()

    fig_line = px.line(
        agg_df, x="year_month", y=metric_choice, color=group_by,
        color_discrete_sequence=COLORS,
        labels={metric_choice: metric_choice.replace(
            "_", " ").title(), "year_month": "Month"}
    )
    fig_line.update_traces(line_width=2)
    fig_line.update_layout(**PLOTLY_LAYOUT, title=f"Monthly Avg {metric_choice.replace('_', ' ').title()} by {group_by.replace('_', ' ').title()}",
                           height=380)
    st.plotly_chart(fig_line, use_container_width=True)

    # Growth rate + seasonality
    col1, col2 = st.columns(2)
    with col1:
        growth_df = dff.groupby("year_month")[
            "daily_growth_rate"].mean().reset_index()
        fig_growth = go.Figure(go.Scatter(
            x=growth_df["year_month"], y=growth_df["daily_growth_rate"] * 100,
            mode="lines+markers", line=dict(color="#34d399", width=2),
            marker=dict(size=5), fill="tozeroy", fillcolor="rgba(52,211,153,0.08)"
        ))
        fig_growth.update_layout(**PLOTLY_LAYOUT, title="Avg Daily Growth Rate (%)",
                                 xaxis_title="Month", yaxis_title="%", height=300)
        st.plotly_chart(fig_growth, use_container_width=True)

    with col2:
        season_df = dff.groupby("weekday")[
            "weekly_seasonality_index"].mean().reset_index()
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        season_df["day_name"] = season_df["weekday"].apply(lambda x: days[x])
        fig_season = go.Figure(go.Bar(
            x=season_df["day_name"], y=season_df["weekly_seasonality_index"],
            marker_color=COLORS[:7], marker_opacity=0.85
        ))
        fig_season.add_hline(y=1.0, line_dash="dot", line_color="#6888b0",
                             annotation_text="Baseline")
        fig_season.update_layout(**PLOTLY_LAYOUT, title="Weekly Seasonality Index",
                                 xaxis_title="Day", yaxis_title="Index", height=300)
        st.plotly_chart(fig_season, use_container_width=True)

    # Rolling mean with confidence band
    st.markdown("<div class='section-header'>ROLLING STATISTICS (30-DAY)</div>",
                unsafe_allow_html=True)
    roll_df = dff.groupby("date")[
        ["rolling_mean_30", "rolling_std_30", "usage_units"]].mean().dropna().reset_index()

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=roll_df["date"], y=roll_df["usage_units"],
        name="Actual Usage", mode="lines",
        line=dict(color="#4a6888", width=1), opacity=0.5
    ))
    fig_roll.add_trace(go.Scatter(
        x=roll_df["date"],
        y=roll_df["rolling_mean_30"] + roll_df["rolling_std_30"],
        name="+1σ", line=dict(width=0), showlegend=False
    ))
    fig_roll.add_trace(go.Scatter(
        x=roll_df["date"],
        y=roll_df["rolling_mean_30"] - roll_df["rolling_std_30"],
        name="Confidence Band", fill="tonexty",
        line=dict(width=0), fillcolor="rgba(59,130,246,0.12)"
    ))
    fig_roll.add_trace(go.Scatter(
        x=roll_df["date"], y=roll_df["rolling_mean_30"],
        name="30-Day Rolling Mean", line=dict(color="#3b82f6", width=2)
    ))
    fig_roll.update_layout(**PLOTLY_LAYOUT, title="Usage Units: Actual vs 30-Day Rolling Mean (±1σ)",
                           height=350)
    st.plotly_chart(fig_roll, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>REGIONAL CAPACITY BREAKDOWN</div>",
                unsafe_allow_html=True)

    reg_agg = dff.groupby("region").agg(
        avg_util=("utilization_pct", "mean"),
        total_cost=("cost_usd", "sum"),
        wasted_cost=("wasted_capacity_cost", "sum"),
        risk_events=("capacity_risk_flag", "sum"),
        underutil=("underutilized_flag", "sum"),
        incidents=("incident_count", "sum"),
        avg_headroom=("headroom_units", "mean"),
        records=("usage_units", "count")
    ).reset_index()
    reg_agg["waste_pct"] = reg_agg["wasted_cost"] / reg_agg["total_cost"] * 100
    reg_agg["avg_util_pct"] = reg_agg["avg_util"] * 100

    # Bubble chart
    fig_bubble = px.scatter(
        reg_agg, x="avg_util_pct", y="waste_pct",
        size="total_cost", color="risk_events",
        hover_name="region", text="region",
        color_continuous_scale="RdYlGn_r",
        labels={"avg_util_pct": "Avg Utilization (%)", "waste_pct": "Waste % of Total Cost",
                "risk_events": "Risk Events", "total_cost": "Total Cost"},
        size_max=50
    )
    fig_bubble.add_vline(x=risk_threshold * 100, line_dash="dash",
                         line_color="#ef4444", annotation_text="Risk threshold")
    fig_bubble.update_traces(textposition="top center",
                             textfont=dict(size=9, color="#a0b4d0"))
    fig_bubble.update_layout(**PLOTLY_LAYOUT,
                             title="Regions: Utilization vs Waste % (bubble = cost, color = risk events)",
                             height=420, coloraxis_colorbar=dict(title="Risk Events"))
    st.plotly_chart(fig_bubble, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_waste = reg_agg.nlargest(10, "wasted_cost")
        fig_waste = go.Figure(go.Bar(
            x=top_waste["wasted_cost"] / 1e6,
            y=top_waste["region"],
            orientation="h",
            marker_color="#ef4444", marker_opacity=0.8
        ))
        fig_waste.update_layout(**PLOTLY_LAYOUT, title="Top 10 Regions by Wasted Capacity ($M)",
                                xaxis_title="$M", height=380)
        st.plotly_chart(fig_waste, use_container_width=True)

    with col2:
        fig_risk = go.Figure(go.Bar(
            x=reg_agg.nlargest(10, "risk_events")["risk_events"],
            y=reg_agg.nlargest(10, "risk_events")["region"],
            orientation="h",
            marker_color="#f59e0b", marker_opacity=0.8
        ))
        fig_risk.update_layout(**PLOTLY_LAYOUT, title="Top 10 Regions by Capacity Risk Events",
                               xaxis_title="Events", height=380)
        st.plotly_chart(fig_risk, use_container_width=True)

    # Heatmap: region vs month utilization
    st.markdown("<div class='section-header'>UTILIZATION HEATMAP</div>",
                unsafe_allow_html=True)
    heat_df = dff.groupby(["region", "year_month"])[
        "utilization_pct"].mean().unstack("year_month") * 100
    heat_df = heat_df.fillna(0)
    # Limit columns for readability
    heat_df = heat_df[sorted(heat_df.columns)[-18:]]

    fig_heat = go.Figure(go.Heatmap(
        z=heat_df.values,
        x=heat_df.columns.tolist(),
        y=heat_df.index.tolist(),
        colorscale="RdYlGn_r",
        zmin=30, zmax=100,
        colorbar=dict(title="Util %", tickfont=dict(size=10)),
        hovertemplate="%{y}<br>%{x}<br>Util: %{z:.1f}%<extra></extra>"
    ))
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        title="Regional Utilization % Heatmap (last 18 months)",
        height=500)
    fig_heat.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Data table
    st.markdown("<div class='section-header'>REGIONAL SUMMARY TABLE</div>",
                unsafe_allow_html=True)
    display_cols = {
        "region": "Region",
        "avg_util_pct": "Avg Util %",
        "total_cost": "Total Cost ($)",
        "wasted_cost": "Wasted ($)",
        "waste_pct": "Waste %",
        "risk_events": "Risk Events",
        "incidents": "Incidents",
        "avg_headroom": "Avg Headroom"
    }
    tbl = reg_agg[display_cols.keys()].copy()
    tbl.columns = display_cols.values()
    tbl["Total Cost ($)"] = tbl["Total Cost ($)"].apply(lambda x: f"${x:,.0f}")
    tbl["Wasted ($)"] = tbl["Wasted ($)"].apply(lambda x: f"${x:,.0f}")
    tbl["Avg Util %"] = tbl["Avg Util %"].apply(lambda x: f"{x:.1f}%")
    tbl["Waste %"] = tbl["Waste %"].apply(lambda x: f"{x:.1f}%")
    tbl["Avg Headroom"] = tbl["Avg Headroom"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(tbl.sort_values("Risk Events", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=300)


# ══════════════════════════════════════════════
# TAB 4 — MODEL & FORECAST
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>XGBoost MODEL STATUS & FORECAST</div>",
                unsafe_allow_html=True)

    import joblib
    import os
    import json

    # Load model
    model = None
    model_loaded = False
    try:
        if os.path.exists("../model/xgb_model.pkl"):
            model = joblib.load("../model/xgb_model.pkl")
            model_loaded = True
    except:
        pass

    col1, col2 = st.columns([1, 2])

    # =========================
    # MODEL STATUS
    # =========================
    with col1:
        status_color = "#34d399" if model_loaded else "#f59e0b"
        status_text = "LOADED ✓" if model_loaded else "NOT FOUND"

        st.markdown(f"""
        <div class='model-box'>
            <div class='model-title'>🤖 XGBoost Model Status</div>
            <div style='color:{status_color};margin-bottom:12px'>{status_text}</div>
        </div>""", unsafe_allow_html=True)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    with col2:
        if model_loaded and hasattr(model, "feature_importances_"):
            fi = model.feature_importances_

            try:
                with open("../model/feature_cols.json", "r") as f:
                    feature_cols_model = json.load(f)
            except:
                feature_cols_model = [f"f_{i}" for i in range(len(fi))]

            min_len = min(len(fi), len(feature_cols_model))

            fi_df = pd.DataFrame({
                "Feature": feature_cols_model[:min_len],
                "Importance": fi[:min_len]
            })
        else:
            np.random.seed(42)
            raw = np.random.rand(20)
            raw = raw / raw.sum()
            fi_df = pd.DataFrame({
                "Feature": [f"Feature {i}" for i in range(20)],
                "Importance": raw
            })

        fi_df = fi_df.nlargest(5, "Importance").sort_values("Importance")

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h"
        ))

        fig_fi.update_layout(**PLOTLY_LAYOUT, title="Feature Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

    # =========================
    # FORECAST SECTION
    # =========================
    st.markdown("<div class='section-header'>DEMAND FORECAST (NEXT 30 DAYS)</div>",
                unsafe_allow_html=True)

    sel_region = st.selectbox(
        "Select Region", sorted(dff["region"].unique()))
    sel_service = st.selectbox(
        "Select Service", sorted(dff["service_type"].unique()))

    if not FORECAST_AVAILABLE:
        st.warning("⚠️ Run batch / Colab to generate forecast_output.csv")

    else:
        sub = forecast_df[
            (forecast_df["region"] == sel_region) &
            (forecast_df["service_type"] == sel_service)
        ].sort_values("date")

        if len(sub) > 0:
            fig_fc = go.Figure()

            # Actual (only available for past)
            fig_fc.add_trace(go.Scatter(
                x=sub["date"],
                y=sub["usage_units"],
                name="Actual",
                line=dict(color="#3b82f6", width=2)
            ))

            # Forecast (includes BOTH past + future)
            fig_fc.add_trace(go.Scatter(
                x=sub["date"],
                y=sub["forecast"],
                name="Forecast (Model)",
                line=dict(color="#f59e0b", width=2, dash="dash")
            ))

            fig_fc.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Actual vs Forecast — {sel_region} / {sel_service}"
            )

            st.plotly_chart(fig_fc, use_container_width=True)

        else:
            st.warning("No forecast data available")
        # If model is loaded, do a real prediction instead:
        # Build feature rows for next 30 days, then call model.predict(X_future)


# ══════════════════════════════════════════════
# TAB 5 — RISK ALERTS
# ══════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>CAPACITY RISK & ANOMALY ALERTS</div>",
                unsafe_allow_html=True)

    # High-risk records
    high_risk = dff[dff["utilization_pct"] >= risk_threshold].copy()
    high_risk["date_str"] = high_risk["date"].dt.strftime("%Y-%m-%d")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#ef4444'>
            <div class='kpi-label'>High-Risk Records</div>
            <div class='kpi-value'>{len(high_risk):,}</div>
            <div class='kpi-delta delta-up'>≥ {risk_threshold*100:.0f}% utilization</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        spike_count = dff["usage_spike"].sum()
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#f59e0b'>
            <div class='kpi-label'>Usage Spikes Detected</div>
            <div class='kpi-value'>{int(spike_count):,}</div>
            <div class='kpi-delta delta-up'>Z-score anomalies</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        zscore_max = dff["z_score"].abs().max()
        st.markdown(f"""
        <div class='kpi-card' style='--accent:#a78bfa'>
            <div class='kpi-label'>Max Z-Score</div>
            <div class='kpi-value'>{zscore_max:.2f}</div>
            <div class='kpi-delta delta-up'>Peak anomaly magnitude</div>
        </div>""", unsafe_allow_html=True)

    # Risk events over time
    risk_ts = dff.groupby("year_month").agg(
        risk_events=("capacity_risk_flag", "sum"),
        spikes=("usage_spike", "sum"),
        incidents=("incident_count", "sum")
    ).reset_index()

    fig_risk_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_risk_ts.add_trace(go.Bar(x=risk_ts["year_month"], y=risk_ts["risk_events"],
                                 name="Capacity Risk Events", marker_color="#ef4444", opacity=0.7))
    fig_risk_ts.add_trace(go.Bar(x=risk_ts["year_month"], y=risk_ts["spikes"],
                                 name="Usage Spikes", marker_color="#f59e0b", opacity=0.7))
    fig_risk_ts.add_trace(go.Scatter(x=risk_ts["year_month"], y=risk_ts["incidents"],
                                     name="Incidents", line=dict(color="#a78bfa", width=2),
                                     mode="lines+markers"), secondary_y=True)
    fig_risk_ts.update_layout(**PLOTLY_LAYOUT, title="Monthly Risk Events, Spikes & Incidents",
                              barmode="group", height=360)
    fig_risk_ts.update_yaxes(title_text="Count", secondary_y=False,
                             gridcolor="#1e3060", tickfont=dict(color="#a0b4d0"))
    fig_risk_ts.update_yaxes(title_text="Incidents", secondary_y=True,
                             tickfont=dict(color="#a78bfa"))
    st.plotly_chart(fig_risk_ts, use_container_width=True)

    # Z-Score anomaly scatter
    st.markdown("<div class='section-header'>ANOMALY DETECTION — Z-SCORE</div>",
                unsafe_allow_html=True)
    plot_df = dff.dropna(subset=["z_score"]).copy()
    spike_df = plot_df[plot_df["usage_spike"] == 1]
    normal_df = plot_df[plot_df["usage_spike"] == 0].sample(
        min(5000, len(plot_df)), random_state=42)

    fig_zscore = go.Figure()
    fig_zscore.add_trace(go.Scatter(
        x=normal_df["date"], y=normal_df["z_score"],
        mode="markers", name="Normal",
        marker=dict(color="#3b82f6", size=3, opacity=0.3)
    ))
    fig_zscore.add_trace(go.Scatter(
        x=spike_df["date"], y=spike_df["z_score"],
        mode="markers", name="Spike / Anomaly",
        marker=dict(color="#ef4444", size=6, opacity=0.8, symbol="diamond")
    ))
    fig_zscore.add_hline(y=2, line_dash="dash",
                         line_color="#f59e0b", annotation_text="+2σ")
    fig_zscore.add_hline(y=-2, line_dash="dash",
                         line_color="#f59e0b", annotation_text="-2σ")
    fig_zscore.update_layout(
        **PLOTLY_LAYOUT, title="Z-Score Anomaly Plot (sampled)", height=360)
    st.plotly_chart(fig_zscore, use_container_width=True)

    # Alert list
    st.markdown("<div class='section-header'>RECENT HIGH-RISK ALERTS</div>",
                unsafe_allow_html=True)
    recent_alerts = (
        high_risk.sort_values("date", ascending=False)
        .head(20)[["date_str", "region", "service_type", "utilization_pct",
                   "usage_units", "headroom_units", "incident_count"]]
    )
    for _, row in recent_alerts.iterrows():
        util_pct = row["utilization_pct"] * 100
        severity = "🔴 CRITICAL" if util_pct >= 90 else "🟡 WARNING"
        cls = "alert-critical" if util_pct >= 90 else "alert-warn"
        st.markdown(
            f"<div class='{cls}'>{severity} &nbsp;|&nbsp; "
            f"<b>{row['date_str']}</b> &nbsp;|&nbsp; "
            f"{row['region']} / {row['service_type']} &nbsp;|&nbsp; "
            f"Util: <b>{util_pct:.1f}%</b> &nbsp;|&nbsp; "
            f"Headroom: {row['headroom_units']:,.0f} units &nbsp;|&nbsp; "
            f"Incidents: {int(row['incident_count'])}</div>",
            unsafe_allow_html=True
        )

    # Macro pressure correlation
    st.markdown("<div class='section-header'>MACRO ECONOMIC PRESSURE CORRELATION</div>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_macro = px.scatter(
            dff.sample(min(3000, len(dff)), random_state=1),
            x="macro_pressure", y="usage_units",
            color="service_type",
            color_discrete_sequence=COLORS,
            trendline="ols",
            trendline_scope="overall",
            labels={"macro_pressure": "Macro Pressure Index",
                    "usage_units": "Usage Units"}
        )
        fig_macro.update_layout(
            **PLOTLY_LAYOUT, title="Macro Pressure vs Usage", height=320)
        st.plotly_chart(fig_macro, use_container_width=True)

    with col2:
        corr_cols = ["usage_units", "utilization_pct", "macro_pressure",
                     "it_spending_index", "enterprise_demand_index",
                     "internet_traffic_growth", "gdp_growth_rate", "daily_growth_rate"]
        corr_matrix = dff[corr_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu", zmin=-1, zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="r")
        ))
        fig_corr.update_layout(
            **PLOTLY_LAYOUT, title="Feature Correlation Matrix", height=320)
        st.plotly_chart(fig_corr, use_container_width=True)
