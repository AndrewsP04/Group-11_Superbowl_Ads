"""
Super Bowl Ads Dashboard – Interactive, Q1, Q2 & Instagram Social Benchmark
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Super Bowl Ads Intelligence", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
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
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("superbowl_ads_combined_master.csv")
    df["brand"] = df["brand"].replace({"Hynudai": "Hyundai"})
    df["year"]  = df["year"].astype(int)

    trait_cols = ["funny", "show_product_quickly", "patriotic",
                  "celebrity", "danger", "animals", "use_sex"]
    for col in trait_cols:
        df[col] = df[col].map({"True": True, "False": False,
                                True: True, False: False}).fillna(False)

    df["yt_view_count"] = pd.to_numeric(df["yt_view_count"], errors="coerce")
    df["yt_like_count"] = pd.to_numeric(df["yt_like_count"], errors="coerce")
    df["yt_engagement_rate"] = np.where(
        df["yt_view_count"] > 0,
        df["yt_like_count"] / df["yt_view_count"], np.nan
    )

    spikes = pd.read_csv("data/wiki_spikes.csv")
    spikes["brand"] = spikes["brand"].replace({"Hynudai": "Hyundai"})
    spike_avg = (
        spikes.groupby(["brand", "year"])["spike_ratio"]
        .mean().reset_index()
        .rename(columns={"spike_ratio": "wiki_spike"})
    )
    df = df.merge(spike_avg, on=["brand", "year"], how="left")

    return df, trait_cols

df, TRAIT_COLS = load_data()

TRAIT_LABELS = {
    "funny":                "Funny",
    "show_product_quickly": "Product-Forward",
    "patriotic":            "Patriotic",
    "celebrity":            "Celebrity",
    "danger":               "Danger / Action",
    "animals":              "Animals",
    "use_sex":              "Sex Appeal",
}

BRANDS = sorted(df["brand"].unique())
YEARS  = sorted(df["year"].unique())
COLORS = px.colors.qualitative.Safe

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏈 Super Bowl Ads\n### Group 11")
    st.markdown("---")
    st.markdown("**Filters**")

    sel_brands = st.multiselect(
        "Brands", BRANDS,
        default=["Budweiser", "Pepsi", "Doritos", "Toyota", "Coca-Cola", "Hyundai"]
    )
    yr_min, yr_max = st.slider(
        "Year Range",
        min_value=int(min(YEARS)), max_value=int(max(YEARS)),
        value=(2010, int(max(YEARS)))
    )
    sel_traits = st.multiselect(
        "Traits to show (Q1 & Q2)",
        options=list(TRAIT_LABELS.keys()),
        default=["funny", "celebrity", "patriotic", "animals"],
        format_func=lambda x: TRAIT_LABELS[x]
    )
    st.markdown("---")
    st.caption("Wikipedia Spike = how many times more people searched the brand on game day vs a normal day (e.g. 3× = 3× normal traffic).")

if not sel_brands: sel_brands = BRANDS
if not sel_traits:  sel_traits = list(TRAIT_LABELS.keys())

filtered = df[
    df["brand"].isin(sel_brands) &
    df["year"].between(yr_min, yr_max)
].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;color:#cdd6f4;font-size:2rem;'>🏈 Super Bowl Ads Intelligence</h1>"
    "<p style='text-align:center;color:#89b4fa;margin-top:-0.5rem;'>Group 11 · Creative Strategy Analysis</p>",
    unsafe_allow_html=True
)

k1, k2, k3 = st.columns(3)
k1.metric("Ads in View", f"{len(filtered):,}")
k2.metric("Total YouTube Views", f"{filtered['yt_view_count'].sum()/1e6:.1f}M" if filtered["yt_view_count"].sum() > 0 else "—")
k3.metric("Avg Wiki Search Spike", f"{filtered['wiki_spike'].mean():.1f}× normal traffic" if filtered["wiki_spike"].notna().any() else "—")

st.markdown("---")

tab1, tab2 = st.tabs([
    "Q1 · Engagement by Trait",
    "Q2 · Trends Over Time"
])

# ══════════════════════════════════════════════════════════════════════════════
#  Q1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Q1 · Which creative traits drive the most engagement?")
    st.caption("Showing only the traits selected in the sidebar. Filtered by selected brands and year range.")

    rows = []
    for trait in sel_traits:
        on = filtered[filtered[trait] == True]
        rows.append({
            "Trait":                           TRAIT_LABELS[trait],
            "Avg YouTube Likes":               on["yt_like_count"].mean(),
            "Avg Wikipedia Spike (× normal)":  on["wiki_spike"].mean(),
            "# Ads":                           len(on),
        })
    q1 = pd.DataFrame(rows).dropna()

    if q1.empty:
        st.warning("No data for the selected filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("YouTube Likes per Trait")
            fig1 = px.bar(
                q1.sort_values("Avg YouTube Likes"),
                x="Avg YouTube Likes", y="Trait", orientation="h",
                text=q1.sort_values("Avg YouTube Likes")["Avg YouTube Likes"].round(0).astype(int),
                color="Avg YouTube Likes", color_continuous_scale="Blues",
                template="plotly_dark",
                hover_data={"# Ads": True},
            )
            fig1.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig1.update_layout(
                coloraxis_showscale=False, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Average YouTube Likes", yaxis_title="",
                margin=dict(l=10, r=80, t=10, b=10), height=370,
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Wikipedia Search Spike per Trait")
            fig2 = px.bar(
                q1.sort_values("Avg Wikipedia Spike (× normal)"),
                x="Avg Wikipedia Spike (× normal)", y="Trait", orientation="h",
                text=q1.sort_values("Avg Wikipedia Spike (× normal)")["Avg Wikipedia Spike (× normal)"].round(2),
                color="Avg Wikipedia Spike (× normal)", color_continuous_scale="Purples",
                template="plotly_dark",
                hover_data={"# Ads": True},
            )
            fig2.update_traces(texttemplate="%{text:.1f}× normal", textposition="outside")
            fig2.update_layout(
                coloraxis_showscale=False, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Wikipedia Spike (× normal daily traffic)", yaxis_title="",
                margin=dict(l=10, r=130, t=10, b=10), height=370,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Summary table
        with st.expander("📋 View numbers as a table"):
            show = q1.copy()
            show["Avg YouTube Likes"] = show["Avg YouTube Likes"].map("{:,.0f}".format)
            show["Avg Wikipedia Spike (× normal)"] = show["Avg Wikipedia Spike (× normal)"].map("{:.2f}×".format)
            st.dataframe(show.set_index("Trait"), use_container_width=True)

        st.markdown(
            '<div class="insight">💡 <b>Insight:</b> '
            'Celebrity and Funny ads earn the most YouTube likes. '
            'Funny ads also drive the biggest Wikipedia search spike — '
            'meaning audiences actively search for the brand after watching. '
            'If the goal is post-game search traffic, <b>lead with humour</b>.</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Q2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Q2 · How have creative strategies changed over time?")
    st.caption("Percentage of Super Bowl ads using each trait per year. Hover over any point to see the exact value.")

    rows2 = []
    for year, grp in df.groupby("year"):  # use full dataset for trend, not filtered
        for trait in sel_traits:
            rows2.append({
                "Year":  year,
                "Trait": TRAIT_LABELS[trait],
                "% of Ads Using Trait": round(grp[trait].mean() * 100, 1),
            })
    trend = pd.DataFrame(rows2)

    # Shade the selected year range
    fig3 = px.line(
        trend, x="Year", y="% of Ads Using Trait",
        color="Trait", markers=True,
        color_discrete_sequence=COLORS,
        template="plotly_dark",
        labels={"% of Ads Using Trait": "% of Ads"},
    )
    fig3.update_traces(line_width=3, marker_size=8)

    # Highlight the selected year range with a shaded region
    fig3.add_vrect(
        x0=yr_min, x1=yr_max,
        fillcolor="rgba(137,180,250,0.08)",
        line_width=1, line_dash="dot", line_color="#89b4fa",
        annotation_text="Selected range", annotation_position="top left",
    )

    fig3.update_layout(
        xaxis_title="Year",
        yaxis_title="% of Super Bowl Ads Using This Trait",
        yaxis=dict(ticksuffix="%", range=[0, 100]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(title="Trait", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=470,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=13),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> '
        'Celebrity endorsements have grown steadily and now appear in 60%+ of ads. '
        'Funny ads remain consistently popular every year. '
        'Patriotic themes peaked around 2017–2019 and are declining. '
        '<b>Celebrity alone is no longer a differentiator — '
        'pair it with Funny for the best results.</b></div>',
        unsafe_allow_html=True
    )

