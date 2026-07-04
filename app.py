"""Streamlit dashboard: the PM/growth view of the GenAI product.

KPI header + five tabbed sections (Adoption, Retention, Features, AI Quality, Unit Economics)
over the synthetic dataset, with plan/feature/date filters. The v2-launch date is annotated on
the time series so the growth-vs-cost-vs-quality tradeoff is visible at a glance.

Run:  uv run streamlit run app.py
"""

import datetime as dt

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ai_product_analytics import metrics as m
from ai_product_analytics.data import MODEL_CHANGE, PRODUCT, QUALITY_RECOVERY, generate

st.set_page_config(page_title="AI Product Analytics", layout="wide", page_icon="📈")

ACCENT = "#6C5CE7"
V2 = dt.datetime.combine(MODEL_CHANGE, dt.time())
FIX = dt.datetime.combine(QUALITY_RECOVERY, dt.time())


@st.cache_data
def load():
    ds = generate()
    return ds.users, ds.sessions


def _mark_v2(fig: go.Figure, show_fix: bool = False) -> None:
    fig.add_vline(x=V2, line_dash="dash", line_color="#e17055", annotation_text="v2 launch")
    if show_fix:
        fig.add_vline(x=FIX, line_dash="dot", line_color="#00b894", annotation_text="quality fix")


def _layout(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    return fig


users, sessions = load()

# ---- Filters -------------------------------------------------------------------------------
st.sidebar.header("Filters")
plans = sorted(sessions["plan"].unique().to_list())
features = sorted(sessions["feature"].unique().to_list())
sel_plans = st.sidebar.multiselect("Plan", plans, default=plans)
sel_features = st.sidebar.multiselect("Feature", features, default=features)
dmin, dmax = sessions["date"].min(), sessions["date"].max()
picked = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
lo, hi = picked if isinstance(picked, tuple) and len(picked) == 2 else (dmin, dmax)
sel_plans = sel_plans or plans
sel_features = sel_features or features

f = sessions.filter(
    pl.col("plan").is_in(sel_plans)
    & pl.col("feature").is_in(sel_features)
    & (pl.col("date") >= lo)
    & (pl.col("date") <= hi)
)
# Cohorts/activation use plan-filtered full history (date/feature slices would distort them).
fp = sessions.filter(pl.col("plan").is_in(sel_plans))
fu = users.filter(pl.col("plan").is_in(sel_plans))

st.title(f"📈 AI Product Analytics — {PRODUCT}")
st.caption(
    "Growth, quality & unit economics for a (fictional) GenAI writing assistant. "
    "**Model v2 shipped 2026-05-30** — watch cost step up while quality dips, then recovers."
)

if f.height == 0:
    st.warning("No sessions match the current filters.")
    st.stop()

# ---- KPI header ----------------------------------------------------------------------------
q = m.quality_summary(f)
e = m.economics_summary(f)
dau = m.active_users_series(f, 1)["active_users"].tail(7).mean()
mau = m.active_users_series(f, 28)["active_users"].tail(1).item()
stickiness = dau / mau if mau else 0.0
pre_cost = f.filter(pl.col("date") < MODEL_CHANGE)["cost_usd"].mean()
post_cost = f.filter(pl.col("date") >= MODEL_CHANGE)["cost_usd"].mean()
cost_delta = f"{(post_cost / pre_cost - 1) * 100:+.0f}% vs v1" if pre_cost and post_cost else None

k = st.columns(6)
k[0].metric("MAU", f"{mau:,}")
k[1].metric("Stickiness (DAU/MAU)", f"{stickiness:.0%}")
k[2].metric("CSAT", f"{q['csat']:.0%}")
k[3].metric("Containment", f"{q['containment_rate']:.0%}")
k[4].metric("Total cost", f"${e['total_cost']:,.0f}")
k[5].metric(
    "Cost / session", f"${e['cost_per_session']:.4f}", delta=cost_delta, delta_color="inverse"
)

tab_adopt, tab_ret, tab_feat, tab_qual, tab_econ = st.tabs(
    ["📊 Adoption", "🔁 Retention", "🧩 Features", "✅ AI Quality", "💵 Unit Economics"]
)

# ---- Adoption ------------------------------------------------------------------------------
with tab_adopt:
    fig = go.Figure()
    for label, win, color in [("DAU", 1, ACCENT), ("WAU", 7, "#0984e3"), ("MAU", 28, "#00b894")]:
        s = m.active_users_series(f, win)
        fig.add_scatter(
            x=s["date"].to_list(), y=s["active_users"].to_list(), name=label, line=dict(color=color)
        )
    _mark_v2(fig)
    st.subheader("Active users over time")
    st.plotly_chart(_layout(fig), width="stretch")

    stick = m.stickiness_series(f)
    figs = go.Figure(
        go.Scatter(
            x=stick["date"].to_list(),
            y=stick["stickiness"].to_list(),
            fill="tozeroy",
            line=dict(color=ACCENT),
        )
    )
    figs.update_layout(yaxis_tickformat=".0%")
    st.subheader("Stickiness (DAU / MAU)")
    st.plotly_chart(_layout(figs, 260), width="stretch")

# ---- Retention -----------------------------------------------------------------------------
with tab_ret:
    r = m.cohort_retention(fp, fu)
    cohorts = sorted(r["cohort"].unique().to_list())
    weeks = sorted(r["week_index"].unique().to_list())
    lookup = {(row["cohort"], row["week_index"]): row["retention"] for row in r.to_dicts()}
    z = [[lookup.get((c, w)) for w in weeks] for c in cohorts]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"W{w}" for w in weeks],
            y=[str(c) for c in cohorts],
            colorscale="Purples",
            zmin=0,
            zmax=1,
            hovertemplate="%{y} · %{x}: %{z:.0%}<extra></extra>",
        )
    )
    st.subheader("Weekly signup-cohort retention")
    st.caption("Each row is a signup cohort; each column is weeks since signup. Darker = stickier.")
    st.plotly_chart(_layout(fig, 420), width="stretch")

# ---- Features ------------------------------------------------------------------------------
with tab_feat:
    col1, col2 = st.columns(2)
    fa = m.feature_adoption(f).sort("reach")
    figf = go.Figure(
        go.Bar(
            x=fa["reach"].to_list(),
            y=fa["feature"].to_list(),
            orientation="h",
            marker_color=ACCENT,
            text=[f"{v:.0%}" for v in fa["reach"].to_list()],
        )
    )
    figf.update_layout(xaxis_tickformat=".0%")
    col1.subheader("Feature reach (share of users)")
    col1.plotly_chart(_layout(figf), width="stretch")

    fn = m.activation_funnel(fp, fu)
    fign = go.Figure(
        go.Funnel(
            y=fn["stage"].to_list(),
            x=fn["users"].to_list(),
            marker_color=ACCENT,
            textinfo="value+percent initial",
        )
    )
    col2.subheader("Activation funnel")
    col2.plotly_chart(_layout(fign), width="stretch")

# ---- AI Quality ----------------------------------------------------------------------------
with tab_qual:
    qs = m.quality_series(f).sort("date")
    fig = go.Figure()
    fig.add_scatter(
        x=qs["date"].to_list(),
        y=qs["quality"].to_list(),
        name="Answer quality",
        line=dict(color=ACCENT),
    )
    fig.add_scatter(
        x=qs["date"].to_list(),
        y=qs["csat"].to_list(),
        name="CSAT (👍 share)",
        line=dict(color="#0984e3"),
    )
    _mark_v2(fig, show_fix=True)
    fig.update_layout(yaxis_range=[0, 1], yaxis_tickformat=".0%")
    st.subheader("Answer quality & CSAT")
    st.caption(
        "v2 briefly dips quality until the fix lands — the tradeoff of shipping capability fast."
    )
    st.plotly_chart(_layout(fig), width="stretch")

    fig2 = go.Figure()
    fig2.add_scatter(
        x=qs["date"].to_list(),
        y=qs["refusal_rate"].to_list(),
        name="Refusal rate",
        line=dict(color="#e17055"),
    )
    fig2.add_scatter(
        x=qs["date"].to_list(),
        y=qs["containment_rate"].to_list(),
        name="Containment rate",
        line=dict(color="#00b894"),
    )
    _mark_v2(fig2, show_fix=True)
    fig2.update_layout(yaxis_tickformat=".0%")
    st.subheader("Refusal & containment")
    st.plotly_chart(_layout(fig2, 280), width="stretch")

# ---- Unit Economics ------------------------------------------------------------------------
with tab_econ:
    ue = m.unit_economics_series(f).sort("date")
    fig = go.Figure()
    fig.add_scatter(
        x=ue["date"].to_list(),
        y=ue["cost_per_active_user"].to_list(),
        name="Cost / active user",
        line=dict(color=ACCENT),
    )
    fig.add_scatter(
        x=ue["date"].to_list(),
        y=ue["cost_per_session"].to_list(),
        name="Cost / session",
        line=dict(color="#e17055"),
        yaxis="y2",
    )
    fig.update_layout(
        yaxis=dict(title="$/active user"),
        yaxis2=dict(title="$/session", overlaying="y", side="right", showgrid=False),
    )
    _mark_v2(fig)
    st.subheader("Unit economics over time")
    st.caption(
        "v2 permanently steps up cost per user — capability isn't free; margin has to keep up."
    )
    st.plotly_chart(_layout(fig), width="stretch")

    dim = st.selectbox("Break spend down by", ["model", "plan", "feature"])
    sb = m.spend_by(f, dim).sort("cost")
    figb = go.Figure(
        go.Bar(
            x=sb["cost"].to_list(),
            y=sb[dim].to_list(),
            orientation="h",
            marker_color=ACCENT,
            text=[f"${v:,.0f}" for v in sb["cost"].to_list()],
        )
    )
    st.subheader(f"Spend by {dim}")
    st.plotly_chart(_layout(figb, 280), width="stretch")
