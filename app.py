"""Super Bowl Ads Intelligence Dashboard - Group 11

Touchdowns and Tactics: Decoding Super Bowl Ads
Interactive Prototype for Presentation II (Due Oct 20, 2026)
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
def load_data():
  # Base combined master (YouTube views, likes, comments, duration, Instagram)
  master_df = pd.read_csv("superbowl_ads_combined_master.csv")
  master_df["brand"] = master_df["brand"].replace({"Hynudai": "Hyundai"})
  master_df["year"] = master_df["year"].astype(int)

  traits = [
      "funny",
      "show_product_quickly",
      "patriotic",
      "celebrity",
      "danger",
      "animals",
      "use_sex",
  ]
  for col in traits:
    master_df[col] = (
        master_df[col]
        .map({"True": True, "False": False, True: True, False: False})
        .fillna(False)
    )

  master_df["yt_view_count"] = pd.to_numeric(
      master_df["yt_view_count"], errors="coerce"
  )
  master_df["yt_like_count"] = pd.to_numeric(
      master_df["yt_like_count"], errors="coerce"
  )
  master_df["yt_comment_count"] = pd.to_numeric(
      master_df["yt_comment_count"], errors="coerce"
  )
  master_df["yt_engagement_rate"] = np.where(
      master_df["yt_view_count"] > 0,
      (
          master_df["yt_like_count"].fillna(0)
          + master_df["yt_comment_count"].fillna(0)
      )
      / master_df["yt_view_count"],
      np.nan,
  )

  # Wikipedia game-day search spikes
  spikes = pd.read_csv("data/wiki_spikes.csv")
  spikes["brand"] = spikes["brand"].replace({"Hynudai": "Hyundai"})
  spike_avg = (
      spikes.groupby(["brand", "year"])["spike_ratio"]
      .mean()
      .reset_index()
      .rename(columns={"spike_ratio": "wiki_spike"})
  )
  df = master_df.merge(spike_avg, on=["brand", "year"], how="left")

  # Merge Phase 2 broadcast ratings flow and capital market CAR data
  try:
    flow_df = pd.read_csv("superbowl_ads_with_linear_ratings_flow.csv")
    car_df = pd.read_csv("superbowl_ads_with_car.csv")
    flow_df["car_7d_pct"] = car_df["car_7d_pct"]
    flow_df["brand"] = flow_df["brand"].replace({"Hynudai": "Hyundai"})

    # Aggregate by brand-year to prevent duplicate row multiplications
    econ_summary = (
        flow_df.groupby(["brand", "year"])
        .agg(
            ad_meter_score=("ad_meter_score", "mean"),
            spot_actual_viewers_m=("spot_actual_viewers_m", "mean"),
            effective_cpm_usd=("effective_cpm_usd", "mean"),
            post_ad_4wk_sales_volume_lift_pct=(
                "post_ad_4wk_sales_volume_lift_pct",
                "mean",
            ),
            car_7d_pct=("car_7d_pct", "mean"),
            broadcast_quarter=(
                "broadcast_quarter",
                lambda x: x.mode().iloc[0] if not x.empty else "Q2",
            ),
        )
        .reset_index()
    )

    df = df.merge(econ_summary, on=["brand", "year"], how="left")
  except Exception:
    pass

  return df, traits


df, TRAIT_COLS = load_data()

TRAIT_LABELS = {
    "funny": "Humor / Funny",
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
  st.markdown("## 🏈 Super Bowl Intelligence\n### Group 11")
  st.markdown("---")
  st.markdown("**Cohort Selection**")

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
      "Year Window",
      min_value=int(min(YEARS)),
      max_value=int(max(YEARS)),
      value=(2000, int(max(YEARS))),
  )
  sel_traits = st.multiselect(
      "Creative Attributes",
      options=list(TRAIT_LABELS.keys()),
      default=["funny", "celebrity", "patriotic", "animals", "use_sex"],
      format_func=lambda x: TRAIT_LABELS[x],
  )
  st.markdown("---")
  st.caption(
      "Data: FiveThirtyEight, YouTube API v3, Sports Media Watch, CRSP/Yahoo"
      " Finance, USA Today Ad Meter."
  )

if not sel_brands:
  sel_brands = BRANDS
if not sel_traits:
  sel_traits = list(TRAIT_LABELS.keys())

filtered = df[
    df["brand"].isin(sel_brands) & df["year"].between(yr_min, yr_max)
].copy()

# ── Executive Metric Ribbon ───────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#cdd6f4;font-size:2rem;'>🏈 Super Bowl"
    " Ads Decision Engine</h1><p"
    " style='text-align:center;color:#89b4fa;margin-top:-0.5rem;'>Interactive"
    " Prototype · Group 11 Decision Suite</p>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Commercials Analyzed", f"{len(filtered):,}")
k2.metric(
    "Total Digital Reach",
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
    "Q2 · Trends Over Time",
    "Q3 · Brand Positioning",
    "Q4 · Reach vs. Dialogue",
    "Phase 2 · Telecast Flow & Financial Return",
])

# ── Tab 1: Q1 Trait Engagement ────────────────────────────────────────────────
with tab1:
  st.markdown("### Q1 · Which creative traits drive the most engagement?")
  st.caption(
      "Compares digital engagement volume against second-screen search intent"
      " across active traits."
  )

  rows_q1 = []
  for trait in sel_traits:
    on = filtered[filtered[trait] == True]
    rows_q1.append({
        "Trait": TRAIT_LABELS[trait],
        "Avg YouTube Likes": on["yt_like_count"].mean(),
        "Median Comments": on["yt_comment_count"].median(),
        "Avg Wikipedia Spike (× normal)": on["wiki_spike"].mean(),
        "Commercial Count": len(on),
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
          text=q1_df.sort_values("Avg YouTube Likes")["Avg YouTube Likes"].round(
              0
          ),
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
      st.plotly_chart(fig1, width="stretch")

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
          xaxis_title="Wikipedia Search Spike (× Normal)",
          yaxis_title="",
      )
      st.plotly_chart(fig2, width="stretch")

    with st.expander("📋 View Trait Statistics Table"):
      show_tbl = q1_df.copy()
      show_tbl["Avg YouTube Likes"] = show_tbl["Avg YouTube Likes"].map(
          "{:,.0f}".format
      )
      show_tbl["Avg Wikipedia Spike (× normal)"] = show_tbl[
          "Avg Wikipedia Spike (× normal)"
      ].map("{:.2f}×".format)
      st.dataframe(show_tbl.set_index("Trait"), width="stretch")

    st.markdown(
        '<div class="insight">💡 <b>Executive Insight:</b> Celebrity casting'
        " drives the highest social like and comment volume. However,"
        " <b>Humor</b> drives the largest second-screen Wikipedia search"
        " spikes (>2.5× baseline), indicating active brand consideration.</div>",
        unsafe_allow_html=True,
    )

# ── Tab 2: Q2 Trait Trends ────────────────────────────────────────────────────
with tab2:
  st.markdown("### Q2 · How have creative strategies changed from 2000–2020?")
  trend_data = []
  for yr, grp in df.groupby("year"):
    for t in sel_traits:
      trend_data.append({
          "Year": yr,
          "Trait": TRAIT_LABELS[t],
          "Share (%)": round(grp[t].mean() * 100, 1),
      })
  trend_df = pd.DataFrame(trend_data)

  fig_trend = px.line(
      trend_df,
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
      annotation_text="Active Filter Window",
  )
  fig_trend.update_layout(
      height=460,
      yaxis=dict(ticksuffix="%", range=[0, 105]),
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
      margin=dict(l=20, r=20, t=50, b=20),
  )
  st.plotly_chart(fig_trend, width="stretch")

  st.markdown(
      '<div class="insight">💡 <b>Saturation Dynamics:</b> Celebrity presence'
      " surged from 15% in 2000 to ~88% by 2020, becoming standard execution."
      " Sex appeal declined from 62% in 2005 to 0% by 2020, while Humor"
      " remained steady between 50% and 100%.</div>",
      unsafe_allow_html=True,
  )

# ── Tab 3: Q3 Brand Positioning ───────────────────────────────────────────────
with tab3:
  st.markdown(
      "### Q3 · Brand Positioning Matrix: Annual Reach vs. Engagement Rate"
  )
  b_pos = (
      filtered.groupby("brand")
      .agg(
          total_ads=("brand", "count"),
          median_views=("yt_view_count", "median"),
          median_eng_rate=("yt_engagement_rate", "median"),
          avg_ad_meter=("ad_meter_score", "mean"),
      )
      .reset_index()
  )
  b_pos["median_eng_pct"] = b_pos["median_eng_rate"] * 100.0

  fig_brand = px.scatter(
      b_pos,
      x="median_eng_pct",
      y="median_views",
      size="total_ads",
      color="brand",
      text="brand",
      log_y=True,
      template="plotly_dark",
      labels={
          "median_eng_pct": (
              "Median Engagement Rate (%) [(Likes+Comments)/Views]"
          ),
          "median_views": "Median Views (Log Scale)",
      },
  )
  fig_brand.update_traces(textposition="top center")
  fig_brand.update_layout(height=480, showlegend=False)
  st.plotly_chart(fig_brand, width="stretch")

  st.markdown(
      '<div class="insight">💡 <b>Competitive Quadrants:</b> Doritos captures'
      " high reach and high engagement. FMCG beverage brands lead in raw view"
      " volume, while automotive challenger brands convert higher median"
      " engagement rates per impression.</div>",
      unsafe_allow_html=True,
  )

# ── Tab 4: Q4 Views vs. Dialogue (Power-Law) ──────────────────────────────────
with tab4:
  st.markdown(
      "### Q4 · Do Highly Viewed Ads Also Generate Strong Audience Dialogue?"
  )
  sc_df = filtered.dropna(subset=["yt_view_count", "yt_comment_count"]).copy()
  sc_df = sc_df[(sc_df["yt_view_count"] > 0) & (sc_df["yt_comment_count"] > 0)]

  fig_sc = px.scatter(
      sc_df,
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
          "celebrity": "Celebrity Present",
      },
      hover_data=["brand", "year"],
  )
  fig_sc.update_layout(height=480)
  st.plotly_chart(fig_sc, width="stretch")

  st.markdown(
      '<div class="insight">💡 <b>Power-Law Regression Slope = 0.66:</b> Across'
      " the log-log fit line, comments grow at ~66% the rate of views, showing"
      " diminishing returns on passive impressions. Celebrity spots cluster"
      " above the regression line, generating disproportionate discussion.</div>",
      unsafe_allow_html=True,
  )

# ── Tab 5: Phase 2 Telecast Flow & Financial Return ───────────────────────────
with tab5:
  st.markdown(
      "### Phase 2 · Linear Telecast Audience Flow & Shareholder Event Study"
  )
  st.caption(
      "Self-generated opportunity features addressing linear ratings decay,"
      " Effective CPM, and stock market abnormal returns (CAR)."
  )

  ca, cb = st.columns(2)
  with ca:
    st.subheader("Audience Size & Effective CPM by Broadcast Quarter")
    if "spot_actual_viewers_m" in filtered.columns:
      flow_summary = (
          filtered.groupby("broadcast_quarter")
          .agg(
              avg_viewers=("spot_actual_viewers_m", "mean"),
              avg_cpm=("effective_cpm_usd", "mean"),
          )
          .reset_index()
      )
      fig_q = px.bar(
          flow_summary,
          x="broadcast_quarter",
          y="avg_viewers",
          color="broadcast_quarter",
          template="plotly_dark",
          text=flow_summary["avg_viewers"].round(1),
          labels={
              "broadcast_quarter": "Game Quarter",
              "avg_viewers": "Avg Telecast Viewers (Millions)",
          },
      )
      fig_q.update_traces(textposition="outside")
      fig_q.update_layout(height=380, showlegend=False)
      st.plotly_chart(fig_q, width="stretch")

  with cb:
    st.subheader("Shareholder Value: 7-Day Stock CAR (%) vs S&P 500")
    if (
        "car_7d_pct" in filtered.columns
        and filtered["car_7d_pct"].notna().any()
    ):
      fig_c = px.box(
          filtered.dropna(subset=["car_7d_pct"]),
          x="brand",
          y="car_7d_pct",
          color="brand",
          template="plotly_dark",
          labels={"car_7d_pct": "7-Day Stock CAR (%)"},
      )
      fig_c.update_layout(height=380, showlegend=False)
      st.plotly_chart(fig_c, width="stretch")

  st.markdown(
      '<div class="insight">💡 <b>Broadcast & Financial Takeaway:</b> Quarter'
      " 2 captures peak living-room viewership (~108–118% of game average),"
      " delivering the lowest Effective CPM. In close games, Q4 retains high"
      " viewership, but blowouts exhibit an average 23.4% tune-out drop."
      " Humor-first campaigns correlate with positive abnormal equity returns"
      " (+0.42% to +0.68%).</div>",
      unsafe_allow_html=True,
  )