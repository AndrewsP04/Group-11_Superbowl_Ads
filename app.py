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

k1, k2, k3 = st.columns(3)
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
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "Q1 · Trait Engagement",
    "Q2 · Trends Over Time",
    "Q3 · Brand Positioning",
])

# ── Tab 1: Q1 Trait Engagement ────────────────────────────────────────────────
with tab1:
  st.markdown("### Q1 · Which creative traits are associated with engagement?")
  st.caption(
      "Compares average YouTube engagement rate and average Wikipedia search "
      "spike for ads that use each selected trait."
  )

  rows_q1 = []
  for trait in sel_traits:
    on = filtered[filtered[trait] == True]
    rows_q1.append({
        "Trait": TRAIT_LABELS[trait],
        "Avg Engagement Rate (%)": on["yt_engagement_rate"].mean() * 100,
        "Avg Wikipedia Spike (×)": on["wiki_spike"].mean(),
    })

  q1_df = pd.DataFrame(rows_q1).dropna(
      subset=["Avg Engagement Rate (%)"]
  )

  if not q1_df.empty:
    c1, c2 = st.columns(2)

    with c1:
      engagement_plot = q1_df.sort_values("Avg Engagement Rate (%)")
      fig1 = px.bar(
          engagement_plot,
          x="Avg Engagement Rate (%)",
          y="Trait",
          orientation="h",
          text=[
              f"{rate:.2f}%"
              for rate in engagement_plot["Avg Engagement Rate (%)"]
          ],
          color="Avg Engagement Rate (%)",
          color_continuous_scale="Blues",
          template="plotly_dark",
          title="Average YouTube Engagement Rate by Trait",
      )
      fig1.update_traces(textposition="outside")
      fig1.update_layout(
          coloraxis_showscale=False,
          height=380,
          xaxis=dict(
              title="Average YouTube engagement rate",
              tickformat=".1f",
              ticksuffix="%",
              dtick=0.1,
              range=[0, q1_df["Avg Engagement Rate (%)"].max() * 1.25],
          ),
          yaxis_title="",
          margin=dict(l=10, r=130, t=45, b=10),
      )
      st.plotly_chart(fig1, width="stretch")

    with c2:
      wiki_plot = q1_df.dropna(
          subset=["Avg Wikipedia Spike (×)"]
      ).sort_values("Avg Wikipedia Spike (×)")
      if not wiki_plot.empty:
        fig2 = px.bar(
            wiki_plot,
            x="Avg Wikipedia Spike (×)",
            y="Trait",
            orientation="h",
            text=[
                f"{spike:.1f}×"
                for spike in wiki_plot["Avg Wikipedia Spike (×)"]
            ],
            color="Avg Wikipedia Spike (×)",
            color_continuous_scale="Purples",
            template="plotly_dark",
            title="Average Wikipedia Spike by Trait",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            coloraxis_showscale=False,
            height=380,
            xaxis=dict(
                title="Average Wikipedia search spike",
                tickformat=".0f",
                ticksuffix="×",
                dtick=1,
                range=[0, wiki_plot["Avg Wikipedia Spike (×)"].max() * 1.25],
            ),
            yaxis_title="",
            margin=dict(l=10, r=130, t=45, b=10),
        )
        st.plotly_chart(fig2, width="stretch")

    st.dataframe(
        q1_df.round(
            {"Avg Engagement Rate (%)": 2, "Avg Wikipedia Spike (×)": 2}
        ),
        hide_index=True,
        width="stretch",
    )
  else:
    st.info("No engagement data is available for these filters.")

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
  st.markdown("### Q3 · Brand Comparison: Median Engagement Rate")
  st.caption(
      "Brands are ranked by their median YouTube engagement rate. The label "
      "shows how many ads are included for each brand."
  )
  b_pos = (
      filtered.groupby("brand", as_index=False)
      .agg(
          total_ads=("yt_engagement_rate", "count"),
          median_eng_rate=("yt_engagement_rate", "median"),
      )
  )
  b_pos["median_eng_pct"] = b_pos["median_eng_rate"] * 100.0
  b_pos = b_pos.dropna(subset=["median_eng_pct"]).sort_values(
      "median_eng_pct", ascending=True
  )

  if not b_pos.empty:
    b_pos["bar_label"] = b_pos.apply(
        lambda row: f"{row['median_eng_pct']:.2f}%  (n={int(row['total_ads'])})",
        axis=1,
    )
    fig_brand = px.bar(
      b_pos,
      x="median_eng_pct",
      y="brand",
      orientation="h",
      text="bar_label",
      color="median_eng_pct",
      color_continuous_scale="Blues",
      template="plotly_dark",
      labels={"median_eng_pct": "Median engagement rate (%)", "brand": "Brand"},
      title="Median YouTube engagement rate by brand",
    )
    fig_brand.update_traces(textposition="outside", cliponaxis=False)
    fig_brand.update_layout(
        height=max(360, 48 * len(b_pos)),
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Median engagement rate (%)",
        yaxis_title="",
        margin=dict(l=10, r=100, t=45, b=10),
    )
    st.plotly_chart(fig_brand, width="stretch")
  else:
    st.info("No engagement-rate data is available for these brands and years.")

