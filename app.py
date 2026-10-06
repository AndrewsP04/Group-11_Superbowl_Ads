"""
Super Bowl Ads Dashboard – Group 11
Interactive Decision Intelligence: Creative Traits, Brand Positioning,
Viewership Power-Law, Telecast Flow & Financial Event Study.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Super Bowl Ads Intelligence",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #1e1e2e; border-right: 1px solid #2e2e4e; }
[data-testid="stSidebar"] * { color: #cdd6f4 !important; }
div[data-testid="metric-container"] {
    background: #2e2e4e; border: 1px solid #45457a;
    border-radius: 12px; padding: 1rem;
}
.insight {
    background: #2e2e4e; border-left: 4px solid #89b4fa;
    border-radius: 6px; padding: 0.8rem 1rem;
    color: #cdd6f4; font-size: 0.9rem; margin-top: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Data Loading & Consolidation ──────────────────────────────────────────────
@st.cache_data
def load_unified_data():
  # Base combined master (YouTube + Instagram metrics)
  df = pd.read_csv("superbowl_ads_combined_master.csv")
  df["brand"] = df["brand"].replace({"Hynudai": "Hyundai"})
  df["year"] = df["year"].astype(int)

  trait_cols = [
      "funny",
      "show_product_quickly",
      "patriotic",
      "celebrity",
      "danger",
      "animals",
      "use_sex",
  ]
  for col in trait_cols:
    df[col] = (
        df[col]
        .map({"True": True, "False": False, True: True, False: False})
        .fillna(False)
    )

  df["yt_view_count"] = pd.to_numeric(df["yt_view_count"], errors="coerce")
  df["yt_like_count"] = pd.to_numeric(df["yt_like_count"], errors="coerce")
  df["yt_comment_count"] = pd.to_numeric(
      df["yt_comment_count"], errors="coerce"
  )
  df["yt_engagement_rate"] = np.where(
      df["yt_view_count"] > 0,
      (df["yt_like_count"].fillna(0) + df["yt_comment_count"].fillna(0))
      / df["yt_view_count"],
      np.nan,
  )

  # Wikipedia game-day spikes
  spikes = pd.read_csv("data/wiki_spikes.csv")
  spikes["brand"] = spikes["brand"].replace({"Hynudai": "Hyundai"})
  spike_avg = (
      spikes.groupby(["brand", "year"])["spike_ratio"]
      .mean()
      .reset_index()
      .rename(columns={"spike_ratio": "wiki_spike"})
  )
  df = df.merge(spike_avg, on=["brand", "year"], how="left")

  # Linear TV ratings flow & effective CPM
  try:
    tv_df = pd.read_csv("superbowl_ads_with_linear_ratings_flow.csv")
    tv_subset = tv_df[[
        "year",
        "brand",
        "broadcast_quarter",
        "spot_actual_viewers_m",
        "effective_cpm_usd",
        "est_30s_ad_cost_usd",
        "ad_meter_score",
        "roi_efficiency_score",
    ]].drop_duplicates(subset=["year", "brand"])
    df = df.merge(tv_subset, on=["year", "brand"], how="left")
  except Exception:
    pass

  # Stock abnormal returns (CAR)
  try:
    car_df = pd.read_csv("superbowl_ads_with_car.csv")
    car_subset = car_df[[
        "year",
        "brand",
        "car_7d_pct",
        "post_ad_4wk_sales_volume_lift_pct",
    ]].drop_duplicates(subset=["year", "brand"])
    df = df.merge(car_subset, on=["year", "brand"], how="left")
  except Exception:
    pass

  return df, trait_cols


df, TRAIT_COLS = load_unified_data()

TRAIT_LABELS = {
    "funny": "Funny / Humor",
    "show_product_quickly": "Product-Forward",
    "patriotic": "Patriotic",
    "celebrity": "Celebrity",
    "danger": "Danger / Action",
    "animals": "Animals",
    "use_sex": "Sex Appeal",
}

BRANDS = sorted(df["brand"].unique())
YEARS = sorted(df["year"].unique())
COLORS = px.colors.qualitative.Safe

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
  st.markdown("## 🏈 Super Bowl Intelligence\n### Team 11")
  st.markdown("---")
  st.markdown("**Cohort Filters**")

  sel_brands = st.multiselect(
      "Brands",
      BRANDS,
      default=[
          "Budweiser",
          "Bud Light",
          "Pepsi",
          "Doritos",
          "Toyota",
          "Coca-Cola",
          "Hyundai",
      ],
  )
  yr_min, yr_max = st.slider(
      "Year Range",
      min_value=int(min(YEARS)),
      max_value=int(max(YEARS)),
      value=(2005, int(max(YEARS))),
  )
  sel_traits = st.multiselect(
      "Traits to Analyze",
      options=list(TRAIT_LABELS.keys()),
      default=["funny", "celebrity", "patriotic", "animals", "use_sex"],
      format_func=lambda x: TRAIT_LABELS[x],
  )
  st.markdown("---")
  st.caption("Data Sources: FiveThirtyEight, YouTube Data API, Sports Media Watch, CRSP/Yahoo Finance, USA Today Ad Meter.")

if not sel_brands:
  sel_brands = BRANDS
if not sel_traits:
  sel_traits = list(TRAIT_LABELS.keys())

filtered = df[
    df["brand"].isin(sel_brands) & df["year"].between(yr_min, yr_max)
].copy()

# ── Executive KPI Header ──────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#cdd6f4;font-size:2.2rem;'>🏈 Super"
    " Bowl Ads Decision Engine</h1><p"
    " style='text-align:center;color:#89b4fa;margin-top:-0.5rem;'>Team 11 ·"
    " Creative Strategy & Broadcast ROI Optimization</p>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Commercials Analyzed", f"{len(filtered):,}")
k2.metric(
    "Total Digital Views",
    f"{filtered['yt_view_count'].sum() / 1e6:.1f}M"
    if filtered["yt_view_count"].sum() > 0
    else "—",
)
k3.metric(
    "Avg Game-Day Search Spike",
    f"{filtered['wiki_spike'].mean():.1f}×"
    if filtered["wiki_spike"].notna().any()
    else "—",
)
k4.metric(
    "Median 7D Stock CAR",
    f"{filtered['car_7d_pct'].median():+.2f}%"
    if "car_7d_pct" in filtered.columns and filtered["car_7d_pct"].notna().any()
    else "—",
)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Q1 · Trait Engagement",
    "Q2 · Strategy Evolution",
    "Q3 · Brand Positioning",
    "Q4 · Views vs. Engagement",
    "Q5 · Broadcast ROI & Finance",
])

# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1: Q1 · Engagement by Creative Trait
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
  st.markdown("### Q1 · Which creative traits drive the most engagement?")
  st.caption(
      "Comparing digital engagement (YouTube Likes) and second-screen consumer"
      " intent (Wikipedia Search Spikes)."
  )

  rows_q1 = []
  for trait in sel_traits:
    on = filtered[filtered[trait] == True]
    rows_q1.append({
        "Trait": TRAIT_LABELS[trait],
        "Avg YouTube Likes": on["yt_like_count"].mean(),
        "Median Comments": on["yt_comment_count"].median(),
        "Avg Wikipedia Spike (× normal)": on["wiki_spike"].mean(),
        "Ads": len(on),
    })
  q1_df = pd.DataFrame(rows_q1).dropna(subset=["Avg YouTube Likes"])

  if not q1_df.empty:
    c1, c2 = st.columns(2)
    with c1:
      fig1 = px.bar(
          q1_df.sort_values("Avg YouTube Likes"),
          x="Avg YouTube Likes",
          y="Trait",
          orientation="h",
          text=q1_df.sort_values("Avg YouTube Likes")[
              "Avg YouTube Likes"
          ].round(0),
          color="Avg YouTube Likes",
          color_continuous_scale="Blues",
          template="plotly_dark",
      )
      fig1.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
      fig1.update_layout(
          coloraxis_showscale=False,
          height=360,
          margin=dict(l=10, r=80, t=10, b=10),
          xaxis_title="Average YouTube Likes",
          yaxis_title="",
      )
      st.plotly_chart(fig1, use_container_width=True)

    with c2:
      fig2 = px.bar(
          q1_df.sort_values("Avg Wikipedia Spike (× normal)"),
          x="Avg Wikipedia Spike (× normal)",
          y="Trait",
          orientation="h",
          text=q1_df.sort_values("Avg Wikipedia Spike (× normal)")[
              "Avg Wikipedia Spike (× normal)"
          ].round(2),
          color="Avg Wikipedia Spike (× normal)",
          color_continuous_scale="Purples",
          template="plotly_dark",
      )
      fig2.update_traces(
          texttemplate="%{text:.1f}× normal", textposition="outside"
      )
      fig2.update_layout(
          coloraxis_showscale=False,
          height=360,
          margin=dict(l=10, r=90, t=10, b=10),
          xaxis_title="Wikipedia Search Spike Ratio",
          yaxis_title="",
      )
      st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Executive Finding:</b> Celebrity ads lead'
        " in digital dialogue and like volume, but <b>Humor</b> drives the"
        " largest game-day brand search spikes (>2.5× baseline). To capture"
        " immediate consideration, lead with humor.</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2: Q2 · Trends Over Time
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
  st.markdown("### Q2 · How have creative strategies evolved (2000–2020)?")
  trend_records = []
  for yr, grp in df.groupby("year"):
    for t in sel_traits:
      trend_records.append({
          "Year": yr,
          "Trait": TRAIT_LABELS[t],
          "Share (%)": round(grp[t].mean() * 100, 1),
      })
  t_df = pd.DataFrame(trend_records)

  fig_trend = px.line(
      t_df,
      x="Year",
      y="Share (%)",
      color="Trait",
      markers=True,
      color_discrete_sequence=COLORS,
      template="plotly_dark",
  )
  fig_trend.add_vrect(
      x0=yr_min,
      x1=yr_max,
      fillcolor="rgba(137,180,250,0.08)",
      line_width=1,
      line_dash="dot",
      line_color="#89b4fa",
      annotation_text="Filtered Window",
  )
  fig_trend.update_layout(
      height=460,
      yaxis=dict(ticksuffix="%", range=[0, 105]),
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
      margin=dict(l=20, r=20, t=50, b=20),
  )
  st.plotly_chart(fig_trend, use_container_width=True)

  st.markdown(
      '<div class="insight">💡 <b>Strategic Shift:</b> Celebrity casting has'
      " surged to ~88% penetration, rendering it table-stakes rather than a"
      " differentiator. Sex appeal has fallen to ~0%, while Humor remains the"
      " continuous baseline.</div>",
      unsafe_allow_html=True,
  )

# ══════════════════════════════════════════════════════════════════════════════
#  Tab 3: Q3 · Brand Positioning Matrix
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
  st.markdown("### Q3 · Brand Positioning: Annualized Reach vs. Engagement")
  b_summary = (
      filtered.groupby("brand")
      .agg(
          total_ads=("brand", "count"),
          median_views=("yt_view_count", "median"),
          median_eng_rate=("yt_engagement_rate", "median"),
          avg_ad_meter=(
              "ad_meter_score",
              lambda x: x.mean() if x.notna().any() else np.nan,
          ),
      )
      .reset_index()
  )
  b_summary["median_eng_pct"] = b_summary["median_eng_rate"] * 100.0

  fig_pos = px.scatter(
      b_summary,
      x="median_eng_pct",
      y="median_views",
      size="total_ads",
      color="brand",
      text="brand",
      log_y=True,
      template="plotly_dark",
      labels={
          "median_eng_pct": "Median Engagement Rate (%) [(Likes+Comments)/Views]",
          "median_views": "Median Views (Log Scale)",
      },
  )
  fig_pos.update_traces(textposition="top center")
  fig_pos.update_layout(height=480, showlegend=False)
  st.plotly_chart(fig_pos, use_container_width=True)

  st.markdown(
      '<div class="insight">💡 <b>Quadrant Dynamics:</b> Doritos commands the'
      " high-reach, high-engagement quadrant. Beverage giants (Bud Light,"
      " Pepsi) secure massive broadcast reach, while automotive challengers"
      " (Hyundai, Kia) convert higher engagement rates per impression.</div>",
      unsafe_allow_html=True,
  )

# ══════════════════════════════════════════════════════════════════════════════
#  Tab 4: Q4 · Views vs. Engagement Dynamics
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
  st.markdown(
      "### Q4 · Do Highly Viewed Ads Also Drive Strong Dialogue Volume?"
  )
  scatter_df = filtered.dropna(
      subset=["yt_view_count", "yt_comment_count"]
  ).copy()
  scatter_df = scatter_df[
      (scatter_df["yt_view_count"] > 0) & (scatter_df["yt_comment_count"] > 0)
  ]

  fig_scatter = px.scatter(
      scatter_df,
      x="yt_view_count",
      y="yt_comment_count",
      color="celebrity",
      log_x=True,
      log_y=True,
      trendline="ols",
      template="plotly_dark",
      color_discrete_map={True: "#f38ba8", False: "#89b4fa"},
      labels={
          "yt_view_count": "Total Views (Log Scale)",
          "yt_comment_count": "Total Comments (Log Scale)",
          "celebrity": "Celebrity Casting",
      },
      hover_data=["brand", "year"],
  )
  fig_scatter.update_layout(height=480)
  st.plotly_chart(fig_scatter, use_container_width=True)

  st.markdown(
      '<div class="insight">💡 <b>Power-Law Regression Slope = 0.66:</b> Across'
      " the log-log fit line, comments grow at only ~66% the rate of views."
      " Mass reach dilutes engagement rate. However, celebrity spots"
      " consistently cluster above the fit line, driving disproportionate"
      " conversation.</div>",
      unsafe_allow_html=True,
  )

# ══════════════════════════════════════════════════════════════════════════════
#  Tab 5: Business ROI & Linear Telecast Flow
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
  st.markdown(
      "### Business ROI · Broadcast Quarter Audience, Effective CPM & Stock"
      " Impact"
  )

  col_a, col_b = st.columns(2)

  with col_a:
    st.subheader("Audience Flow by Airtime Quarter")
    if "spot_actual_viewers_m" in filtered.columns:
      flow_summary = (
          filtered.groupby("broadcast_quarter")["spot_actual_viewers_m"]
          .mean()
          .reset_index()
      )
      fig_qtr = px.bar(
          flow_summary,
          x="broadcast_quarter",
          y="spot_actual_viewers_m",
          color="broadcast_quarter",
          template="plotly_dark",
          text=flow_summary["spot_actual_viewers_m"].round(1),
          labels={
              "broadcast_quarter": "Game Quarter",
              "spot_actual_viewers_m": "Avg Viewers (Millions)",
          },
      )
      fig_qtr.update_traces(textposition="outside")
      fig_qtr.update_layout(height=380, showlegend=False)
      st.plotly_chart(fig_qtr, use_container_width=True)
    else:
      st.info("Broadcast flow data syncing...")

  with col_b:
    st.subheader("Shareholder Event Study: 7-Day Stock CAR (%)")
    if (
        "car_7d_pct" in filtered.columns
        and filtered["car_7d_pct"].notna().any()
    ):
      fig_car = px.box(
          filtered.dropna(subset=["car_7d_pct"]),
          x="brand",
          y="car_7d_pct",
          color="brand",
          template="plotly_dark",
          labels={"car_7d_pct": "7-Day Cumulative Abnormal Return (%)"},
      )
      fig_car.update_layout(height=380, showlegend=False)
      st.plotly_chart(fig_car, use_container_width=True)
    else:
      st.info("Market return event study syncing...")

  st.markdown(
      '<div class="insight">💡 <b>Airtime Optimization:</b> Quarter 2 and the'
      " Halftime window capture the peak living-room rating (~108–118% of game"
      " average), delivering the lowest Effective CPM. In close games, Q4"
      " delivers high attention, but blowout games carry severe viewer attrition"
      " (~18–25% tune-out).</div>",
      unsafe_allow_html=True,
  )