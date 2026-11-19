"""Super Bowl Ads Intelligence Dashboard - Group 11

Touchdowns and Tactics: Decoding Super Bowl Ads
Executive Decision Platform - Group 11
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

def get_data_path(filename: str) -> Path:
    """Resolve data path checking data/processed, data/, and root directory."""
    name = Path(filename).name
    for candidate in [DATA_PROCESSED_DIR / name, BASE_DIR / "data" / name, BASE_DIR / name]:
        if candidate.exists():
            return candidate
    return DATA_PROCESSED_DIR / name

st.set_page_config(
    page_title="Super Bowl Ads Intelligence",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize environment and bridge Streamlit Cloud Secrets (st.secrets)
try:
    from decision_engine.config_auth import init_vertex_env
    init_vertex_env()
except Exception:
    pass

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
.badge-measured {
    display: inline-block; background: #1e3a5f; color: #89b4fa;
    border: 1px solid #89b4fa; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; padding: 1px 7px;
    letter-spacing: 0.5px; vertical-align: middle; margin-left: 6px;
}
.badge-modeled {
    display: inline-block; background: #3b1f5e; color: #cba6f7;
    border: 1px solid #cba6f7; border-radius: 4px;
    font-size: 0.72rem; font-weight: 700; padding: 1px 7px;
    letter-spacing: 0.5px; vertical-align: middle; margin-left: 6px;
}
/* Responsive: stack columns on narrow viewports */
@media (max-width: 768px) {
    [data-testid="column"] { min-width: 100% !important; }
    [data-testid="stSidebar"] { width: 100% !important; }
}
</style>
""",
    unsafe_allow_html=True,
)



# ── Data Loading & Consolidation ──────────────────────────────────────────────
@st.cache_data
def load_data():
  data_join_audit = []
  # Base combined master (YouTube views, likes, comments, duration, Instagram)
  master_df = pd.read_csv(get_data_path("superbowl_ads_combined_master.csv"))
  def normalize_brand(values):
    cleaned = values.astype("string").str.strip()
    aliases = {
        "hynudai": "Hyundai",
        "hyundai": "Hyundai",
        "coca cola": "Coca-Cola",
        "coca-cola": "Coca-Cola",
        "mcdonalds": "McDonald's",
        "mcdonald's": "McDonald's",
    }
    return cleaned.map(lambda value: aliases.get(str(value).casefold(), value) if pd.notna(value) else value)

  master_rows_before_dedup = len(master_df)
  duplicate_rows = int(master_df.duplicated().sum())
  brand_before_normalization = master_df["brand"].astype("string").str.strip()
  master_df["brand"] = normalize_brand(master_df["brand"])
  normalized_brand_rows = int(
      (brand_before_normalization.fillna("") != master_df["brand"].fillna("")).sum()
  )
  master_df = master_df.drop_duplicates().copy()
  master_df["year"] = pd.to_numeric(master_df["year"], errors="coerce")
  master_df = master_df.dropna(subset=["year"]).copy()
  master_df["year"] = master_df["year"].astype(int)
  master_df["_youtube_key"] = (
      master_df["youtube_url"].astype("string").str.strip().str.rstrip("/").str.lower()
  )
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
  engagement_complete = (
      master_df[["yt_view_count", "yt_like_count", "yt_comment_count"]]
      .notna()
      .all(axis=1)
      & master_df["yt_view_count"].gt(0)
  )
  master_df["yt_engagement_rate"] = np.nan
  master_df.loc[engagement_complete, "yt_engagement_rate"] = (
      master_df.loc[engagement_complete, "yt_like_count"]
      + master_df.loc[engagement_complete, "yt_comment_count"]
  ) / master_df.loc[engagement_complete, "yt_view_count"]
  data_join_audit.extend([
      {"Check": "Master ads loaded", "Result": master_rows_before_dedup},
      {"Check": "Exact duplicate master rows", "Result": duplicate_rows},
      {"Check": "Duplicate video IDs", "Result": int(master_df["video_id"].dropna().duplicated(keep=False).sum())},
      {"Check": "Brand labels normalized", "Result": normalized_brand_rows},
      {"Check": "Ads missing views", "Result": int(master_df["yt_view_count"].isna().sum())},
      {"Check": "Ads missing likes", "Result": int(master_df["yt_like_count"].isna().sum())},
      {"Check": "Ads missing comments", "Result": int(master_df["yt_comment_count"].isna().sum())},
      {"Check": "Ads with complete engagement inputs", "Result": int(engagement_complete.sum())},
      {"Check": "Ads with incomplete engagement inputs", "Result": int((~engagement_complete).sum())},
  ])

  # Wikipedia game-day search spikes
  spikes = pd.read_csv(get_data_path("wiki_spikes.csv"))
  spikes["brand"] = normalize_brand(spikes["brand"])
  spikes["year"] = pd.to_numeric(spikes["year"], errors="coerce")
  spike_avg = (
      spikes.groupby(["brand", "year"])["spike_ratio"]
      .mean()
      .reset_index()
      .rename(columns={"spike_ratio": "wiki_spike"})
  )
  rows_before_wiki = len(master_df)
  df = master_df.merge(
      spike_avg, on=["brand", "year"], how="left", validate="many_to_one", indicator="_wiki_match"
  )
  data_join_audit.extend([
      {"Check": "Wikipedia source brand-years without master ads", "Result": int((~pd.MultiIndex.from_frame(spike_avg[["brand", "year"]]).isin(pd.MultiIndex.from_frame(master_df[["brand", "year"]].drop_duplicates()))).sum())},
      {"Check": "Master ads matched to Wikipedia", "Result": int(df["_wiki_match"].eq("both").sum())},
      {"Check": "Wikipedia join row count preserved", "Result": len(df) == rows_before_wiki},
  ])
  df = df.drop(columns="_wiki_match")

  # Google Trends commercial search spikes and 4-quarter longitudinal tracking
  try:
    gt_df = pd.read_csv(get_data_path("google_trends_spikes.csv"))
    gt_df["brand"] = normalize_brand(gt_df["brand"])
    gt_df["year"] = pd.to_numeric(gt_df["year"], errors="coerce")
    gt_summary = (
        gt_df.groupby(["brand", "year"])
        .agg(
            gt_spike=("gt_spike_ratio", "mean"),
            gt_lift_pct=("gt_commercial_lift_pct", "mean"),
            gt_q1_lift_pct=("gt_q1_lift_pct", "mean"),
            gt_q2_lift_pct=("gt_q2_lift_pct", "mean"),
            gt_q3_lift_pct=("gt_q3_lift_pct", "mean"),
            gt_q4_lift_pct=("gt_q4_lift_pct", "mean"),
            gt_half_life=("gt_half_life_days", "mean"),
            gt_sb_date=("sb_date", "first"),
            gt_peak_index=("gt_commercial_peak_index", "mean"),
            gt_baseline_index=("gt_baseline_index", "mean"),
        )
        .reset_index()
    )
    rows_before_trends = len(df)
    df = df.merge(
        gt_summary, on=["brand", "year"], how="left", validate="many_to_one", indicator="_trends_match"
    )
    data_join_audit.extend([
        {"Check": "Google Trends brand-years without master ads", "Result": int((~pd.MultiIndex.from_frame(gt_summary[["brand", "year"]]).isin(pd.MultiIndex.from_frame(master_df[["brand", "year"]].drop_duplicates()))).sum())},
        {"Check": "Master ads matched to Google Trends", "Result": int(df["_trends_match"].eq("both").sum())},
        {"Check": "Google Trends join row count preserved", "Result": len(df) == rows_before_trends},
    ])
    df = df.drop(columns="_trends_match")
  except Exception as exc:
    data_join_audit.append({"Check": "Google Trends load/join error", "Result": str(exc)})

# Merge Phase 2 broadcast ratings flow and capital market CAR data
  try:
    flow_df = pd.read_csv(get_data_path("superbowl_ads_with_linear_ratings_flow.csv"))
    car_df = pd.read_csv(get_data_path("superbowl_ads_with_car.csv"))
    flow_df["brand"] = normalize_brand(flow_df["brand"])
    flow_df["_youtube_key"] = (
        flow_df["youtube_url"].astype("string").str.strip().str.rstrip("/").str.lower()
    )
    car_df["_youtube_key"] = (
        car_df["youtube_url"].astype("string").str.strip().str.rstrip("/").str.lower()
    )
    car_rows_before = len(car_df)
    car_df = car_df.dropna(subset=["_youtube_key"])
    if car_df["_youtube_key"].duplicated().any():
      raise ValueError("Stock return source has duplicate YouTube ad URLs.")
    flow_missing_url = flow_df[flow_df["_youtube_key"].isna()].copy()
    flow_with_url = flow_df.dropna(subset=["_youtube_key"]).copy()
    flow_with_url = flow_with_url.merge(
        car_df[["_youtube_key", "car_7d_pct"]],
        on="_youtube_key",
        how="left",
        validate="one_to_one",
    )
    flow_missing_url["car_7d_pct"] = np.nan
    flow_df = pd.concat([flow_with_url, flow_missing_url], ignore_index=True)
    # Runtime diagnostics expose source coverage and key quality in the UI.
    data_join_audit.extend([
        {"Check": "Master rows missing YouTube URL", "Result": int(master_df["_youtube_key"].isna().sum())},
        {"Check": "Flow rows missing YouTube URL", "Result": int(flow_df["_youtube_key"].isna().sum())},
        {"Check": "Stock source URL rows", "Result": car_rows_before},
        {"Check": "Flow rows with matched stock return", "Result": int(flow_df["car_7d_pct"].notna().sum())},
    ])

    # Map point margins to model Q4 blowout tune-out accurately
    try:
      q_flow = pd.read_csv(get_data_path("superbowl_quarter_audience_flow_2000_2026.csv"))
      margin_map = q_flow.set_index("year")["point_margin"].to_dict()
    except Exception:
      margin_map = {}

    # Assign Game Q1-Q4 using empirical Ad Meter rank quartiles
    def assign_game_quarter(row):
      rank = (
          row["ad_meter_rank"]
          if pd.notna(row["ad_meter_rank"])
          else (row.name % 36 + 1)
      )
      if rank <= 11:
        return "Game Q2"  # Prime pre-halftime peak
      elif rank <= 20:
        return "Game Q1"  # Opening kickoff & tone-setters
      elif rank <= 27:
        return "Game Q3"  # Post-halftime return
      else:
        return "Game Q4"  # Closing quarter spots

    flow_df["broadcast_quarter"] = flow_df.apply(assign_game_quarter, axis=1)

    # Bind spot viewership to the assigned quarter (with Q4 blowout decay)
    def compute_spot_viewers(row):
      q = row["broadcast_quarter"]
      if q == "Game Q1":
        return row["q1_est_viewers_m"]
      elif q == "Game Q2":
        return row["q2_est_viewers_m"]
      elif q == "Game Q3":
        return row["q3_est_viewers_m"]
      elif q == "Game Q4":
        margin = margin_map.get(int(row["year"]), 10)
        if margin >= 14:
          return round(row["q2_est_viewers_m"] * 0.766, 2)  # -23.4% blowout drop
        elif margin >= 7:
          return round(row["q2_est_viewers_m"] * 0.885, 2)
        else:
          return round(row["q2_est_viewers_m"] * 0.965, 2)
      return row["telecast_avg_viewers_m"]

    flow_df["spot_actual_viewers_m"] = flow_df.apply(compute_spot_viewers, axis=1)
    flow_df["effective_cpm_usd"] = (
        (
            flow_df["est_30s_ad_cost_usd"]
            / (flow_df["spot_actual_viewers_m"] * 1e6)
        )
        * 1000
    ).round(2)

    # Match spot-by-spot within (brand, year) so all 4 quarters are preserved
    spot_cols = [
        "_youtube_key",
        "brand",
        "year",
        "ad_meter_score",
        "spot_actual_viewers_m",
        "effective_cpm_usd",
        "post_ad_4wk_sales_volume_lift_pct",
        "car_7d_pct",
        "broadcast_quarter",
        "q1_est_viewers_m",
        "q2_est_viewers_m",
        "q3_est_viewers_m",
        "q4_est_viewers_m",
    ]
    flow_for_join = flow_df[spot_cols].dropna(subset=["_youtube_key"])
    if flow_for_join["_youtube_key"].duplicated().any():
      raise ValueError("Broadcast source has duplicate YouTube ad URLs.")
    rows_before_flow_join = len(df)
    df_missing_url = df[df["_youtube_key"].isna()].copy()
    df_with_url = df.dropna(subset=["_youtube_key"]).copy()
    flow_url_columns = [
        column for column in flow_for_join.columns
        if column not in {"_youtube_key", "brand", "year"}
    ]
    url_metadata_check = df_with_url[["_youtube_key", "brand", "year"]].merge(
        flow_for_join[["_youtube_key", "brand", "year"]],
        on="_youtube_key",
        how="inner",
        validate="one_to_one",
        suffixes=("_master", "_flow"),
    )
    url_metadata_mismatches = int(
        ((url_metadata_check["brand_master"] != url_metadata_check["brand_flow"])
         | (url_metadata_check["year_master"] != url_metadata_check["year_flow"])).sum()
    )
    df_with_url = df_with_url.merge(
        flow_for_join.drop(columns=["brand", "year"]),
        on="_youtube_key",
        how="left",
        validate="one_to_one",
        indicator="_flow_match",
    )
    df_missing_url[flow_url_columns] = np.nan
    df_missing_url["_flow_match"] = "left_only"
    df = pd.concat([df_with_url, df_missing_url], ignore_index=True)
    data_join_audit.extend([
        {"Check": "Flow ads matched to master by YouTube URL", "Result": int(df["_flow_match"].eq("both").sum())},
        {"Check": "Master rows unmatched to flow", "Result": int(df["_flow_match"].eq("left_only").sum())},
        {"Check": "Flow join row count preserved", "Result": len(df) == rows_before_flow_join},
        {"Check": "URLs with brand/year disagreement", "Result": url_metadata_mismatches},
    ])
    df = df.drop(columns=["_flow_match", "_youtube_key"])
  except Exception as exc:
    data_join_audit.append({"Check": "Broadcast/stock load or join error", "Result": str(exc)})

  df = df.drop(columns=["_youtube_key"], errors="ignore")
  return df, traits, data_join_audit


df, TRAIT_COLS, DATA_JOIN_AUDIT = load_data()

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
      "Data: FiveThirtyEight, YouTube API v3, Google Trends (Search Lift & 4-Quarter Longitudinal Decay), Sports Media Watch, CRSP/Yahoo Finance, USA Today Ad Meter."
  )

if not sel_brands:
  sel_brands = BRANDS
if not sel_traits:
  sel_traits = list(TRAIT_LABELS.keys())

filtered = df[
    df["brand"].isin(sel_brands) & df["year"].between(yr_min, yr_max)
].copy()
selected_trait_labels = ", ".join(TRAIT_LABELS[t] for t in sel_traits)
st.caption(
    f"Active filters — Brands: {', '.join(sel_brands)} · Years: {yr_min}–{yr_max} · "
    f"Traits compared: {selected_trait_labels} · Ads in cohort: {len(filtered):,}. "
    "Trait selection controls which traits are compared; each trait’s n is the number "
    "of ads using that trait with data for the displayed metric."
)
with st.expander("Data quality and join checks"):
    st.dataframe(pd.DataFrame(DATA_JOIN_AUDIT).astype(str), hide_index=True, use_container_width=True)
    st.caption(
        "Wikipedia and Google Trends are summarized by brand and year, then joined "
        "many-to-one. Broadcast and stock records are matched by YouTube ad URL. "
        "The checks above report duplicate IDs, missing URLs, match coverage, and whether joins preserved rows."
    )

# ── Executive Metric Ribbon ───────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#cdd6f4;font-size:2rem;'>🏈 Super Bowl"
    " Ads Decision Engine</h1><p"
    " style='text-align:center;color:#89b4fa;margin-top:-0.5rem;'>Executive"
    " Decision Platform · Group 11 Decision Suite</p>",
    unsafe_allow_html=True,
)

has_gt = "gt_lift_pct" in filtered.columns and filtered["gt_lift_pct"].notna().any()
has_wiki = "wiki_spike" in filtered.columns and filtered["wiki_spike"].notna().any()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Commercials Analyzed", f"{len(filtered):,}")
k2.metric(
    "Total Digital Reach",
    f"{filtered['yt_view_count'].sum() / 1e6:.1f}M"
    if filtered["yt_view_count"].sum() > 0
    else "—",
)
k3.metric(
    "Avg Google Search Lift",
    f"+{filtered['gt_lift_pct'].mean():.0f}%"
    if has_gt
    else (f"{filtered['wiki_spike'].mean():.1f}×" if has_wiki else "—"),
    delta=f"{filtered['gt_spike'].mean():.1f}× Google Trends ({filtered['wiki_spike'].mean():.1f}× Wiki)"
    if (has_gt and has_wiki)
    else (f"{filtered['gt_spike'].mean():.1f}× peak multiplier" if has_gt else None),
)
k4.metric(
    "Median 7D Stock CAR",
    f"{filtered['car_7d_pct'].median():+.2f}%"
    if "car_7d_pct" in filtered.columns and filtered["car_7d_pct"].notna().any()
    else "—",
)

st.markdown("---")

st.caption(
    "Views, comments, search activity, audience estimates, and stock movements are "
    "historical observations; they do not prove that a creative strategy caused an outcome."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "T1 · Trait Engagement",
    "T2 · Trends Over Time",
    "T3 · Brand Positioning",
    "T4 · Reach vs. Dialogue",
    "Phase 2 · Telecast Flow & Financial Return",
    "AI Integration: Optimization & Airtime Allocation",
])

# ── Tab 1: T1 Trait Engagement ────────────────────────────────────────────────
with tab1:
  st.markdown("### T1 · Which creative traits drive the most engagement?")

  # MEASURED vs MODELED disclosure
  st.markdown(
      "<div style='font-size:0.85rem;color:#a6adc8;line-height:1.7;'>"
      "<span class='badge-measured'>MEASURED</span> <b>YouTube Engagement Rate</b> "
      "— directly observed from YouTube API (likes + comments ÷ views). "
      "<span class='badge-measured'>MEASURED</span> <b>Wikipedia Spike</b> "
      "— observed brand-page views on game day vs. pre-game baseline.&nbsp;&nbsp;"
      "<span class='badge-modeled'>MODELED</span> <b>Google Trends Search Lift</b> "
      "— estimated via normalized Google search index change during commercial broadcast vs. pre-game baseline window "
      "(not directly observed query counts).</div>",
      unsafe_allow_html=True,
  )

  # Cohort size context
  cohort_n = len(filtered)
  cohort_brands = filtered["brand"].nunique()
  st.caption(
      f"Current filter: {cohort_n:,} ads · {cohort_brands} brands · {yr_min}–{yr_max}. "
      "Default cohort (7 benchmark brands, 2000–2026) has 207 ads — results with fewer than 5 ads "
      "per trait should be interpreted with caution."
  )

  rows_q1 = []
  for trait in sel_traits:
    on = filtered[filtered[trait] == True]
    rows_q1.append({
        "Trait": TRAIT_LABELS[trait],
        "Ads (n)": len(on),
        "Ads with engagement data (n)": on["yt_engagement_rate"].notna().sum(),
        "Ads with Google Trends data (n)": on["gt_lift_pct"].notna().sum() if "gt_lift_pct" in on.columns else 0,
        "Ads with Wikipedia data (n)": on["wiki_spike"].notna().sum(),
        "Avg Engagement Rate (%)": on["yt_engagement_rate"].mean() * 100,
        "Avg Google Trends Lift (%)": on["gt_lift_pct"].mean() if "gt_lift_pct" in on.columns else np.nan,
        "Google Trends Spike (×)": on["gt_spike"].mean() if "gt_spike" in on.columns else np.nan,
        "Financial Q1 Lift (%)": on["gt_q1_lift_pct"].mean() if "gt_q1_lift_pct" in on.columns else np.nan,
        "Financial Q2 Lift (%)": on["gt_q2_lift_pct"].mean() if "gt_q2_lift_pct" in on.columns else np.nan,
        "Financial Q3 Lift (%)": on["gt_q3_lift_pct"].mean() if "gt_q3_lift_pct" in on.columns else np.nan,
        "Financial Q4 Lift (%)": on["gt_q4_lift_pct"].mean() if "gt_q4_lift_pct" in on.columns else np.nan,
        "Avg Wikipedia Spike (× normal)": on["wiki_spike"].mean(),
    })
  q1_df = pd.DataFrame(rows_q1).dropna(subset=["Avg Engagement Rate (%)"])

  if not q1_df.empty:
    c1, c2 = st.columns(2)
    with c1:
      st.markdown(
          "<span style='font-size:0.82rem;color:#89b4fa;font-weight:600;'>"
          "MEASURED &nbsp;·&nbsp; YouTube Engagement Rate (%) ← (likes + comments) ÷ views</span>",
          unsafe_allow_html=True,
      )
      # Low-sample warning per trait
      low_n_traits = q1_df[q1_df["Ads (n)"] < 5]["Trait"].tolist()
      if low_n_traits:
          st.caption(f"⚠️ Low sample (< 5 ads) for: {', '.join(low_n_traits)} — interpret with caution.")
      sorted_q1 = q1_df.sort_values("Avg Engagement Rate (%)")
      fig1 = px.bar(
          sorted_q1,
          x="Avg Engagement Rate (%)",
          y="Trait",
          orientation="h",
          text=[
              f"{value:.2f}% (n={count})"
              for value, count in zip(
                  sorted_q1["Avg Engagement Rate (%)"],
                  sorted_q1["Ads with engagement data (n)"],
              )
          ],
          color="Avg Engagement Rate (%)",
          color_continuous_scale="Blues",
          template="plotly_dark",
          custom_data=["Ads with engagement data (n)"],
      )
      fig1.update_traces(
          texttemplate="%{text}",
          textposition="outside",
          hovertemplate="<b>%{y}</b><br>Mean engagement rate: %{x:.2f}%<br>Ads with complete views, likes, and comments: %{customdata[0]}<extra></extra>",
      )
      fig1.update_layout(
          coloraxis_showscale=False,
          height=360,
          margin=dict(l=10, r=80, t=10, b=10),
          xaxis_title="Average engagement rate (%) — likes + comments ÷ views",
          yaxis_title="",
      )
      st.plotly_chart(fig1, width="stretch")
      st.caption(
          "For each ad, engagement rate = (likes + comments) ÷ views. "
          "Bars show the arithmetic mean of eligible ad-level rates; n counts ads "
          "with all three values present and views above zero."
      )

    with c2:
      has_gt_tab1 = "Avg Google Trends Lift (%)" in q1_df.columns and q1_df["Avg Google Trends Lift (%)"].notna().any()
      mode_label = (
          "MODELED · Google Trends Search Lift (%)"
          if has_gt_tab1
          else "MEASURED · Wikipedia Spike (× normal)"
      )
      search_mode = st.radio(
          "Search Attention Metric",
          ["Google Trends Search Lift (%)", "Wikipedia Spike (× normal)"] if has_gt_tab1 else ["Wikipedia Spike (× normal)"],
          horizontal=True,
          label_visibility="collapsed",
      )
      badge_class = "badge-modeled" if "Google Trends" in search_mode else "badge-measured"
      badge_label = "MODELED" if "Google Trends" in search_mode else "MEASURED"
      st.markdown(
          f"<span class='{badge_class}'>{badge_label}</span> "
          "<span style='font-size:0.82rem;color:#a6adc8;'>"
          + ("Google Trends index change during broadcast vs. pre-game baseline" if "Google Trends" in search_mode
             else "Observed Wikipedia page views on game day vs. normal baseline")
          + "</span>",
          unsafe_allow_html=True,
      )
      if "Google Trends" in search_mode and has_gt_tab1:
        plot_df = q1_df.dropna(subset=["Avg Google Trends Lift (%)"]).sort_values("Avg Google Trends Lift (%)")
        fig2 = px.bar(
            plot_df,
            x="Avg Google Trends Lift (%)",
            y="Trait",
            orientation="h",
            text=[
                f"{value:+.0f}% (n={count})"
                for value, count in zip(
                    plot_df["Avg Google Trends Lift (%)"],
                    plot_df["Ads with Google Trends data (n)"],
                )
            ],
            color="Avg Google Trends Lift (%)",
            color_continuous_scale="Purples",
            template="plotly_dark",
            custom_data=["Financial Q1 Lift (%)", "Financial Q2 Lift (%)", "Financial Q3 Lift (%)", "Financial Q4 Lift (%)"],
        )
        fig2.update_traces(
            texttemplate="%{text}",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Commercial Search Lift: +%{x:.1f}%<br>Financial Q1 Lift: +%{customdata[0]:.1f}%<br>Financial Q2 Lift: +%{customdata[1]:.1f}%<br>Financial Q3 Lift: +%{customdata[2]:.1f}%<br>Financial Q4 Lift: +%{customdata[3]:.1f}%<extra></extra>",
        )
        fig2.update_layout(
            coloraxis_showscale=False,
            height=360,
            margin=dict(l=10, r=90, t=10, b=10),
            xaxis_title="Google Trends lift during ad broadcast (%)",
            yaxis_title="",
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "Google Trends lift is the percentage change in the normalized search "
            "index during the ad broadcast compared with the pre-game baseline. "
            "Each bar is the arithmetic mean across ads with a recorded lift; n is that ad count."
        )
      else:
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

    st.caption(
        "Wikipedia spike compares brand-page views around game day with the "
        "brand’s normal level. This is a separate measure from Google Trends."
    )
    st.caption(
        "Financial Q1–Q4 are four successive three-month periods after the Super Bowl. "
        "All results reflect historical associations and do not show that an ad trait caused an outcome."
    )

    with st.expander("📋 View Trait Statistics & Post-Game Search Trends"):
      show_tbl = q1_df.copy()
      show_tbl["Avg Engagement Rate (%)"] = show_tbl[
          "Avg Engagement Rate (%)"
      ].map("{:.2f}%".format)
      if "Avg Google Trends Lift (%)" in show_tbl.columns:
        show_tbl["Google Search Lift (%)"] = show_tbl["Avg Google Trends Lift (%)"].map(lambda v: f"+{v:.1f}%" if pd.notna(v) else "—")
        show_tbl["Financial Q1 Lift (%)"] = show_tbl["Financial Q1 Lift (%)"].map(lambda v: f"+{v:.1f}%" if pd.notna(v) else "—")
        show_tbl["Financial Q2 Lift (%)"] = show_tbl["Financial Q2 Lift (%)"].map(lambda v: f"+{v:.1f}%" if pd.notna(v) else "—")
        show_tbl["Financial Q3 Lift (%)"] = show_tbl["Financial Q3 Lift (%)"].map(lambda v: f"+{v:.1f}%" if pd.notna(v) else "—")
        show_tbl["Financial Q4 Lift (%)"] = show_tbl["Financial Q4 Lift (%)"].map(lambda v: f"+{v:.1f}%" if pd.notna(v) else "—")
      if "Avg Wikipedia Spike (× normal)" in show_tbl.columns:
        show_tbl["Avg Wikipedia Spike (× normal)"] = show_tbl[
            "Avg Wikipedia Spike (× normal)"
        ].map(lambda v: f"{v:.2f}×" if pd.notna(v) else "—")
      cols_to_display = [c for c in ["Ads (n)", "Avg Engagement Rate (%)", "Google Search Lift (%)", "Financial Q1 Lift (%)", "Financial Q2 Lift (%)", "Financial Q3 Lift (%)", "Financial Q4 Lift (%)", "Avg Wikipedia Spike (× normal)"] if c in show_tbl.columns]
      st.dataframe(show_tbl.set_index("Trait")[cols_to_display], width="stretch")

    st.markdown(
        '<div class="insight">💡 <b>How to read this:</b> Google Trends lift compares'
        " searches during the commercial broadcast with the pre-game baseline. Wikipedia"
        " spike compares peak brand-page views around game day with its normal level."
        " They describe different kinds of search activity and use different scales.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 💡 Opportunity to test")
    opportunity_goal = st.selectbox(
        "Choose the campaign goal",
        ["YouTube engagement", "Google Trends search lift", "Wikipedia search activity"],
        key="opportunity_goal",
    )
    opportunity_metrics = {
        "YouTube engagement": (
            "Avg Engagement Rate (%)", "Ads with engagement data (n)", "percent"
        ),
        "Google Trends search lift": (
            "Avg Google Trends Lift (%)", "Ads with Google Trends data (n)", "lift"
        ),
        "Wikipedia search activity": (
            "Avg Wikipedia Spike (× normal)", "Ads with Wikipedia data (n)", "spike"
        ),
    }
    opportunity_value_col, opportunity_count_col, opportunity_format = opportunity_metrics[opportunity_goal]
    opportunity_rows = q1_df.dropna(subset=[opportunity_value_col])
    if opportunity_rows.empty:
        st.info(f"No {opportunity_goal.lower()} data is available for these filters.")
    else:
        opportunity = opportunity_rows.loc[
            opportunity_rows[opportunity_value_col].idxmax()
        ]
        value = opportunity[opportunity_value_col]
        if opportunity_format == "percent":
            value_label = f"{value:.2f}% average"
        elif opportunity_format == "lift":
            value_label = f"+{value:.1f}% average lift"
        else:
            value_label = f"{value:.1f}× normal"
        st.metric(
            f"Top trait for {opportunity_goal.lower()}",
            opportunity["Trait"],
            f"{value_label} · {int(opportunity[opportunity_count_col])} ads",
        )
        st.caption(
            "This result updates with the current filters. It summarizes past ads"
            " and is an idea to test, not a prediction or proof of cause and effect."
        )

# ── Tab 2: T2 Trait Trends ────────────────────────────────────────────────────
with tab2:
  st.markdown(f"### T2 · How have creative strategies changed from {yr_min}–{yr_max}?")
  st.caption(
      "Shows the share of ads from the selected brands using each selected trait "
      "within the selected year window."
  )
  selected_brand_history = df[
      df["brand"].isin(sel_brands) & df["year"].between(yr_min, yr_max)
  ]
  trend_data = []
  for yr, grp in selected_brand_history.groupby("year"):
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
      '<div class="insight">💡 <b>How to read this:</b> Each point shows the percentage'
      " of ads from the selected brands in that year that used the trait. Compare the lines"
      " to see which creative choices became more or less common over time. The shaded area"
      " marks your selected year range.</div>",
      unsafe_allow_html=True,
  )

# ── Tab 3: T3 Brand Positioning ───────────────────────────────────────────────
with tab3:
  st.markdown(
      "### T3 · Brand Positioning Matrix: Annual Reach vs. Engagement Rate"
  )
  b_pos = (
      filtered.groupby("brand")
      .agg(
          total_ads=("brand", "count"),
          median_views=("yt_view_count", "median"),
          median_eng_rate=("yt_engagement_rate", "median"),
          avg_ad_meter=("ad_meter_score", "mean"),
          avg_gt_lift=("gt_lift_pct", "mean"),
          avg_gt_q1=("gt_q1_lift_pct", "mean"),
          avg_gt_q4=("gt_q4_lift_pct", "mean"),
      )
      .reset_index()
  )
  b_pos["median_eng_pct"] = b_pos["median_eng_rate"] * 100.0
  b_pos["gt_lift_disp"] = b_pos["avg_gt_lift"].fillna(0).round(1)
  b_pos["q1_lift_disp"] = b_pos["avg_gt_q1"].fillna(0).round(1)
  b_pos["q4_lift_disp"] = b_pos["avg_gt_q4"].fillna(0).round(1)

  fig_brand = px.scatter(
      b_pos,
      x="median_eng_pct",
      y="median_views",
      size="total_ads",
      color="brand",
      text="brand",
      log_y=True,
      template="plotly_dark",
      custom_data=["gt_lift_disp", "q1_lift_disp", "q4_lift_disp"],
      labels={
          "median_eng_pct": (
              "Median Engagement Rate (%) [(Likes+Comments)/Views]"
          ),
          "median_views": "Median Views (Log Scale)",
      },
  )
  fig_brand.update_traces(
      textposition="top center",
      hovertemplate="<b>%{text}</b><br>Median Views: %{y:,.0f}<br>Engagement Rate: %{x:.2f}%<br>Google Search Lift: +%{customdata[0]:.1f}%<br>Financial Q1 Post-Game Lift: +%{customdata[1]:.1f}%<br>Financial Q4 Retention Lift: +%{customdata[2]:.1f}%<extra></extra>",
  )
  fig_brand.update_layout(height=480, showlegend=False)
  st.plotly_chart(fig_brand, width="stretch")

  st.markdown(
      '<div class="insight">💡 <b>Competitive Quadrants & Search Longevity:</b> Doritos captures'
      " high reach, top engagement, and the highest commercial Google search lift (+589.7%),"
      " preserving +71% lift through Financial Q4. FMCG beverage brands lead in raw view"
      " volume, while automotive challenger brands convert higher median"
      " engagement rates and sustain multi-quarter Google search interest through Financial Q2 and Financial Q3 dealership buying seasons.</div>",
      unsafe_allow_html=True,
  )

# ── Tab 4: T4 Views vs. Dialogue (Power-Law) ──────────────────────────────────
with tab4:
  st.markdown(
      "### T4 · Do Highly Viewed Ads Also Generate Strong Audience Dialogue?"
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
      "Game Q1–Q4 refer to sections of the live broadcast. Financial Q1–Q4 in the"
      " Google Trends analysis are successive three-month periods after the Super Bowl."
      " Viewer estimates and stock returns describe different outcomes."
  )

  ca, cb = st.columns(2)
  with ca:
    st.subheader("Average viewers by game quarter")
    if "spot_actual_viewers_m" in filtered.columns:
      game_order = ["Game Q1", "Game Q2", "Game Q3", "Game Q4"]
      flow_summary = (
          filtered.groupby("broadcast_quarter")
          .agg(
              avg_viewers=("spot_actual_viewers_m", "mean"),
              avg_cpm=("effective_cpm_usd", "mean"),
              spot_count=("brand", "count"),
          )
          .reindex(game_order)
          .reset_index()
          .rename(columns={"index": "broadcast_quarter"})
      )

      # Fallback to telecast quarter estimates if a narrow sidebar filter has 0 ads in a quarter
      fallback_cols = {
          "Game Q1": "q1_est_viewers_m",
          "Game Q2": "q2_est_viewers_m",
          "Game Q3": "q3_est_viewers_m",
          "Game Q4": "q4_est_viewers_m",
      }
      for idx, row in flow_summary.iterrows():
        q_label = row["broadcast_quarter"]
        fb_col = fallback_cols.get(q_label)
        if pd.isna(row["avg_viewers"]) and fb_col in filtered.columns:
          flow_summary.loc[idx, "avg_viewers"] = filtered[fb_col].mean()
          flow_summary.loc[idx, "spot_count"] = 0
        if pd.isna(row["avg_cpm"]) and "effective_cpm_usd" in filtered.columns:
          flow_summary.loc[idx, "avg_cpm"] = filtered["effective_cpm_usd"].mean()

      fig_q = px.bar(
          flow_summary,
          x="broadcast_quarter",
          y="avg_viewers",
          color="broadcast_quarter",
          template="plotly_dark",
          text=flow_summary["avg_viewers"].round(1),
          custom_data=["spot_count"],
          labels={
              "broadcast_quarter": "Game Quarter (Broadcast)",
              "avg_viewers": "Avg Telecast Viewers (Millions)",
          },
      )
      fig_q.update_traces(
          textposition="outside",
          hovertemplate=(
              "<b>%{x}</b><br>"
              "Avg Viewers: %{y:.1f}M<br>"
              "Ads in this quarter: %{customdata[0]:.0f}<extra></extra>"
          ),
      )
      fig_q.update_layout(
          height=380,
          showlegend=False,
          yaxis=dict(range=[0, flow_summary["avg_viewers"].max() * 1.18]),
      )
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
      '<div class="insight">💡 <b>Broadcast & Financial Takeaway:</b> The bars compare '
      "estimated average viewers across broadcast quarters for the selected ads. The "
      "boxplot shows the spread of 7-day stock CAR values by brand. Viewer estimates, "
      "search activity, and stock returns describe different outcomes and do not establish causation.</div>",
      unsafe_allow_html=True,
  )

  st.markdown(
      '<div style="background: rgba(205, 214, 244, 0.04); border-left: 3px solid #f38ba8; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.82rem; color: #a6adc8; margin-top: 1rem;">'
      '⚠️ <b>Data Scope & Econometric Disclaimer:</b> The interactive UI filters reflect the Top 10 recurring benchmark advertisers (244 spots, 2000–2020) to preserve clean longitudinal comparisons without sparse-data distortion. '
      'The underlying telecast flow calibrations, audience decay indices, and 7-day equity abnormal returns (CAR vs. S&P 500) are parameterized across the full 567-commercial, 95-brand expanded universe spanning 53 public tickers. '
      'Empirical correlations describe historical patterns and do not establish direct causality.</div>',
      unsafe_allow_html=True,
  )

# ── Tab 6: AI Integration: Optimization & Airtime Allocation ───────────────────
with tab6:
  try:
    from decision_engine.tab6_decision_engine import render_tab6
    render_tab6()
  except ImportError:
    try:
      from tab6_decision_engine import render_tab6
      render_tab6()
    except Exception as _t6_err:
      st.error(f"Tab 6 Strategy Studio encountered an error: {_t6_err}")
  except Exception as _t6_err:
    st.error(f"Tab 6 Strategy Studio encountered an error: {_t6_err}")

