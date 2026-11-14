"""
tab6_decision_engine.py - Streamlit UI View for the Multi-POV Super Bowl Decision Engine.

Renders Tab 6 in app.py:
1. Auto-training check on first launch (no manual training step required)
2. Natural language storyboard/pitch parser expander (Gemini 3.5 Flash Lite)
3. Interactive multi-parameter simulation sandbox (<20ms response)
4. Three Stakeholder Decision Lenses:
   - Creative Strategy Director (Engagement, Search Surge, EBM Waterfall)
   - Media Buyer / Planner (Telecast flow, Blowout decay, 512-Grid Optimizer)
   - Brand CMO / CFO Risk (Downside 80% CI floor, Non-causal disclosures)
5. Vector RAG Top-3 Nearest Historical Commercials cards (with YouTube links)
6. Google ADK Multi-Agent Boardroom Debate & Executive Memo Generator
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    from .adk_boardroom import run_strategy_board
    from .decision_tools import (
        BUNDLE_PATH,
        QUARTER_VIEWERS_M,
        compute_effective_reach,
        load_bundle,
        optimize_creative_and_airtime,
        parse_concept_pitch_to_controls,
        retrieve_nearest_historical_ads,
        simulate_multi_pov,
    )
except ImportError:
    from adk_boardroom import run_strategy_board
    from decision_tools import (
        BUNDLE_PATH,
        QUARTER_VIEWERS_M,
        compute_effective_reach,
        load_bundle,
        optimize_creative_and_airtime,
        parse_concept_pitch_to_controls,
        retrieve_nearest_historical_ads,
        simulate_multi_pov,
    )


def render_tab6():
    # Auto-train models on first launch if not already cached
    if not BUNDLE_PATH.exists():
        with st.spinner("⚡ Initializing & training Two-Stage RidgeCV + Monotonic EBM models on historical Super Bowl data (first launch only)..."):
            trained_bundle = load_bundle(auto_train=True)
            if trained_bundle:
                st.toast("✅ Models trained and cached to artifacts/ successfully!", icon="🚀")

    # Initialize widget session state defaults if not already present
    if "t6_vert_sel" not in st.session_state:
        st.session_state["t6_vert_sel"] = "Snacks / Food"
    if "t6_runtime_sel" not in st.session_state:
        st.session_state["t6_runtime_sel"] = 30
    if "t6_quarter_sel" not in st.session_state:
        st.session_state["t6_quarter_sel"] = "Game Q2"
    if "t6_spread_sel" not in st.session_state:
        st.session_state["t6_spread_sel"] = 7
    if "t6_arc_sel" not in st.session_state:
        st.session_state["t6_arc_sel"] = 6.0
    if "t6_brand_sel" not in st.session_state:
        st.session_state["t6_brand_sel"] = 7.5
    if "t6_cult_sel" not in st.session_state:
        st.session_state["t6_cult_sel"] = 5.0
    if "t6_traits_sel" not in st.session_state:
        st.session_state["t6_traits_sel"] = ["Humor", "Celebrity"]

    st.markdown("### 🤖 Agentic Campaign Strategy Studio & Multi-POV Decision Engine")
    st.caption(
        "Powered by Two-Stage RidgeCV + Monotonic EBM statistical models, Vector RAG historical comps, "
        "and the Google ADK Multi-Agent Boardroom on Vertex AI."
    )

    # ── 1. Natural Language Pitch Parser ──────────────────────────────────────
    with st.expander("✨ Have a creative brief or script draft? Paste it here to auto-fill controls", expanded=False):
        pitch_input = st.text_area(
            "Campaign Concept Pitch / Storyboard Narrative",
            placeholder="e.g. A high-energy 30-second Doritos commercial featuring a humorous dance-off with a top celebrity, airing during the second half of a tight game...",
            key="tab6_pitch_input",
            height=90,
        )
        if st.button("🪄 Parse Brief into Simulator Controls", key="tab6_parse_btn"):
            if pitch_input.strip():
                with st.spinner("Extracting parameters with Gemini 3.5 Flash Lite..."):
                    parsed = parse_concept_pitch_to_controls(pitch_input)
                    if parsed:
                        # 1. Vertical matching
                        v_in = parsed.get("vertical", "Snacks / Food")
                        v_opts = ["Snacks / Food", "Beverage / FMCG", "Automotive", "Consumer Tech"]
                        matched_v = next((opt for opt in v_opts if opt.lower() in v_in.lower() or v_in.lower() in opt.lower()), "Snacks / Food")
                        st.session_state["t6_vert_sel"] = matched_v

                        # 2. Runtime matching
                        rt_in = int(parsed.get("runtime_sec", 30))
                        rt_opts = [15, 30, 45, 60, 90]
                        st.session_state["t6_runtime_sel"] = min(rt_opts, key=lambda x: abs(x - rt_in))

                        # 3. Quarter matching
                        q_in = str(parsed.get("quarter", "Game Q2"))
                        q_opts = ["Game Q1", "Game Q2", "Game Q3", "Game Q4"]
                        st.session_state["t6_quarter_sel"] = next((opt for opt in q_opts if opt.lower() in q_in.lower()), "Game Q2")

                        # 4. Point spread
                        st.session_state["t6_spread_sel"] = int(np.clip(parsed.get("point_spread", 7), 0, 24))

                        # 5. Traits mapping
                        trait_ui_map = {
                            "funny": "Humor",
                            "humor": "Humor",
                            "celebrity": "Celebrity",
                            "show_product_quickly": "Product-Forward",
                            "product": "Product-Forward",
                            "animals": "Animals",
                            "animal": "Animals",
                            "danger": "Danger/Action",
                            "action": "Danger/Action",
                            "patriotic": "Patriotic",
                            "patriotism": "Patriotic",
                            "use_sex": "Sex Appeal",
                            "sex": "Sex Appeal",
                        }
                        synced_traits = []
                        for t in parsed.get("traits", []):
                            t_clean = str(t).lower().strip()
                            for k, v in trait_ui_map.items():
                                if k in t_clean and v not in synced_traits:
                                    synced_traits.append(v)
                        if synced_traits:
                            st.session_state["t6_traits_sel"] = synced_traits

                        # 6. Continuous semantic scores
                        st.session_state["t6_arc_sel"] = float(np.clip(parsed.get("narrative_arc", 6.0), 1.0, 10.0))
                        st.session_state["t6_brand_sel"] = float(np.clip(parsed.get("brand_centrality", 7.5), 1.0, 10.0))
                        st.session_state["t6_cult_sel"] = float(np.clip(parsed.get("cultural_nostalgia", 5.0), 1.0, 10.0))

                        st.toast("✅ Synchronized simulator controls with creative brief!", icon="🪄")
                        st.rerun()
                    else:
                        st.warning("Could not parse brief. Please ensure GEMINI_API_KEY is configured in .env.")

    # ── 2. Interactive Simulation Controls ────────────────────────────────────
    c1, c2, c3 = st.columns([1.1, 1.1, 1.2])

    with c1:
        sel_vertical = st.selectbox(
            "Brand Vertical",
            ["Snacks / Food", "Beverage / FMCG", "Automotive", "Consumer Tech"],
            key="t6_vert_sel",
        )
        sel_runtime = st.select_slider(
            "Spot Runtime",
            options=[15, 30, 45, 60, 90],
            format_func=lambda x: f":{x}s",
            key="t6_runtime_sel",
        )
        sel_quarter = st.selectbox(
            "Airtime Quarter",
            ["Game Q1", "Game Q2", "Game Q3", "Game Q4"],
            key="t6_quarter_sel",
        )

    with c2:
        sel_spread = st.slider(
            "Game Point Spread (Blowout Risk)",
            min_value=0,
            max_value=24,
            format="%d pts",
            help="Higher spreads trigger monotonic living-room viewer decay in late quarters.",
            key="t6_spread_sel",
        )
        sel_arc = st.slider(
            "Narrative Arc Depth",
            1.0, 10.0, step=0.5,
            help="1.0 = Direct Product Pitch, 10.0 = Character Story / Mini-Movie",
            key="t6_arc_sel",
        )
        sel_brand_cent = st.slider(
            "Brand Centrality Score",
            1.0, 10.0, step=0.5,
            help="1.0 = Brand forgotten, 10.0 = Product integral to story resolution",
            key="t6_brand_sel",
        )

    with c3:
        sel_traits = st.multiselect(
            "Creative Attributes",
            options=["Humor", "Celebrity", "Product-Forward", "Animals", "Danger/Action", "Patriotic", "Sex Appeal"],
            key="t6_traits_sel",
        )
        sel_nostalgia = st.slider(
            "Cultural Nostalgia / Throwback Hook",
            1.0, 10.0, step=0.5,
            help="1.0 = Modern/Neutral, 10.0 = Heavy retro music or pop-culture IP",
            key="t6_cult_sel",
        )

    trait_keys = [
        t.lower().replace("humor", "funny").replace("product-forward", "show_product_quickly").replace("danger/action", "danger").replace("sex appeal", "use_sex")
        for t in sel_traits
    ]

    sim = simulate_multi_pov(
        vertical=sel_vertical,
        runtime_sec=sel_runtime,
        quarter=sel_quarter,
        traits=trait_keys,
        point_spread=sel_spread,
        narrative_arc=sel_arc,
        brand_centrality=sel_brand_cent,
        cultural_nostalgia=sel_nostalgia,
    )

    active_pitch = st.session_state.get("tab6_pitch_input", "")
    nearest_comps = retrieve_nearest_historical_ads(
        traits=trait_keys,
        vertical=sel_vertical,
        runtime_sec=sel_runtime,
        pitch_text=active_pitch,
        k=3,
    )

    st.markdown("---")

    # ── 3. Multi-Stakeholder Decision Lens Switcher ───────────────────────────
    st.markdown("#### 🎯 Select Stakeholder Decision Lens")
    lens = st.radio(
        "Decision Lens",
        ["🎨 Creative Strategy Director POV", "📡 Media Buyer / Planner POV", "💼 Brand CMO / CFO Risk POV"],
        horizontal=True,
        label_visibility="collapsed",
        key="t6_lens_radio",
    )

    if lens == "🎨 Creative Strategy Director POV":
        st.caption("Focus: Maximizing viewer dialogue, second-screen search, and narrative buzz without losing brand recall.")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Predicted YouTube Engagement Rate",
            f"{sim['pred_eng_pct']}%",
            delta=f"80% CI: {sim['eng_ci_low']}% – {sim['eng_ci_high']}%",
        )
        m2.metric(
            "Predicted Wikipedia Search Surge",
            f"{sim['pred_wiki_mult']}x",
            delta=f"80% CI: {sim['wiki_ci_low']}x – {sim['wiki_ci_high']}x",
        )
        m3.metric(
            "Brand Centrality Rating",
            f"{sel_brand_cent:.1f} / 10.0",
            delta="High Recall" if sel_brand_cent >= 7.0 else "Vulnerable to 'Loved Ad, Forgot Brand'",
        )

        if sim.get("waterfall"):
            st.markdown("##### 🔬 Creative Trait Additive Impact (EBM Waterfall)")
            wf_items = sim["waterfall"]
            fig_wf = go.Figure(go.Bar(
                x=[item["impact"] for item in wf_items],
                y=[item["feature"] for item in wf_items],
                orientation="h",
                marker_color=["#f38ba8" if item["impact"] < 0 else "#a6e3a1" for item in wf_items],
                text=[f"{item['impact']:+.3f}" for item in wf_items],
                textposition="outside",
            ))
            fig_wf.add_vline(x=0, line_dash="dash", line_color="#89b4fa")
            fig_wf.update_layout(
                template="plotly_dark",
                height=max(200, len(wf_items) * 36),
                margin=dict(l=20, r=40, t=20, b=20),
                xaxis_title="Marginal Contribution to Log Engagement",
                yaxis_title="",
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    elif lens == "📡 Media Buyer / Planner POV":
        st.caption("Focus: Upfront airtime timing, living-room audience reach efficiency, and fourth-quarter blowout risks.")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            f"Effective Living-Room Reach ({sel_quarter})",
            f"{sim['effective_reach_m']}M viewers",
            delta=f"Audience Retention: {sim['retention_pct']}%",
        )
        m2.metric(
            "Effective Broadcast CPM",
            f"${sim['effective_cpm_usd']}",
            delta="Based on $7M baseline media spend",
        )
        m3.metric(
            "Game Blowout Spread",
            f"{sel_spread} pts",
            delta="⚠️ High Q4 Decay Risk" if sel_spread > 12 and "Q4" in sel_quarter else "Normal Flow",
            delta_color="inverse" if sel_spread > 12 and "Q4" in sel_quarter else "normal",
        )

        qtr_names = ["Game Q1", "Game Q2", "Game Q3", "Game Q4"]
        reaches = [compute_effective_reach(q, sel_spread)[0] for q in qtr_names]
        colors = ["#a6e3a1" if q == sel_quarter else "#45457a" for q in qtr_names]

        fig_reach = go.Figure(go.Bar(
            x=qtr_names,
            y=reaches,
            marker_color=colors,
            text=[f"{r:.1f}M" for r in reaches],
            textposition="outside",
        ))
        fig_reach.update_layout(
            template="plotly_dark",
            height=240,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Audience Size (Millions)",
            title=f"Telecast Audience by Quarter (Blowout Spread: {sel_spread} pts)",
        )
        st.plotly_chart(fig_reach, use_container_width=True)

        st.markdown("##### 🚀 Prescriptive 512-Grid Optimizer: Top 3 Airtime & Creative Blueprints")
        opt_blueprints = optimize_creative_and_airtime(sel_vertical, sel_spread, top_n=3)
        opt_rows = []
        for i, bp in enumerate(opt_blueprints, 1):
            opt_rows.append({
                "Rank": f"#{i}",
                "Airtime Quarter": bp["quarter"],
                "Optimal Creative Mix": bp["trait_label"],
                "Predicted Engagement": f"{bp['pred_eng_pct']}%",
                "Wikipedia Surge": f"{bp['pred_wiki_mult']}x",
                "Telecast Reach": f"{bp['effective_reach_m']}M",
            })
        st.dataframe(opt_rows, hide_index=True, use_container_width=True)

    else:
        st.caption("Focus: Capital allocation governance on $7M+ buy, downside risk floors, and non-causal disclosures.")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Downside Engagement Floor (10th %ile)",
            f"{sim['eng_ci_low']}%",
            delta="Lower 80% Conformal Bound",
            delta_color="off",
        )
        m2.metric(
            "Downside Search Surge Floor (10th %ile)",
            f"{sim['wiki_ci_low']}x",
            delta="Lower 80% Conformal Bound",
            delta_color="off",
        )
        m3.metric(
            "Media Capital at Risk",
            "$7,000,000",
            delta="Upfront 30s broadcast baseline",
        )

        st.markdown(
            '<div style="background: rgba(243, 139, 168, 0.08); border-left: 4px solid #f38ba8; '
            'border-radius: 6px; padding: 0.85rem 1.1rem; font-size: 0.85rem; color: #cdd6f4; margin: 1rem 0;">'
            '⚖️ <b>CFO Governance & Non-Causal Disclaimers:</b> Modeled outputs describe observed historical cohort patterns '
            '(2000–2026) across public advertisers. Commercial lift metrics reflect engagement interest and attention surge, '
            'not guaranteed retail sales revenue or causal stock return lift. Capital commitments must be vetted against '
            'issuer downside cash flow requirements.</div>',
            unsafe_allow_html=True,
        )

    # ── 4. Vector RAG Nearest Historical Commercials ──────────────────────────
    st.markdown("---")
    st.markdown("#### 🎞️ Historical Precedents (Vector RAG Comps)")
    st.caption("Top 3 closest actual Super Bowl commercials retrieved via embedding similarity from the 567-spot dataset:")

    comp_cols = st.columns(len(nearest_comps))
    for col_idx, comp in enumerate(nearest_comps):
        with comp_cols[col_idx]:
            st.markdown(
                f"""
                <div style="background: #181825; border: 1px solid #313244; border-radius: 8px; padding: 1rem; height: 100%;">
                    <div style="font-size: 0.8rem; color: #89b4fa; font-weight: 700;">{comp['year']} · {comp['brand']}</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #cdd6f4; margin: 0.3rem 0;">{comp['title'][:55]}</div>
                    <div style="font-size: 0.82rem; color: #a6adc8;">
                        • YouTube Engagement: <b>{comp['engagement_rate']}%</b><br/>
                        • Wikipedia Surge: <b>{comp['wiki_spike']}x</b><br/>
                        • Match Similarity: <b>{int(comp['similarity'] * 100)}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if comp.get("url"):
                st.link_button("▶️ Watch Commercial", comp["url"], use_container_width=True)

    # ── 5. Google ADK Multi-Agent Boardroom Debate ────────────────────────────
    st.markdown("---")
    st.markdown("#### 🏛️ Google ADK Strategy Board & Pitch Memo Generator")
    st.caption(
        "Convenes an adversarial 3-persona strategy board (Creative Director vs Media Planner vs CFO Risk) "
        "orchestrated by Google ADK ParallelAgent on Vertex AI."
    )

    if st.button("⚡ Convene AI Strategy Board & Generate Executive Brief", key="t6_adk_btn"):
        scenario_text = (
            f"Brand Vertical: {sel_vertical}, Runtime: :{sel_runtime}s, Target Quarter: {sel_quarter}, "
            f"Point Spread: {sel_spread} pts. Creative Traits: {', '.join(sel_traits) if sel_traits else 'None selected'}. "
            f"Narrative Arc: {sel_arc}/10, Brand Centrality: {sel_brand_cent}/10."
        )

        with st.spinner("Executing Google ADK Multi-Agent Boardroom debate on Vertex AI..."):
            memos = run_strategy_board(scenario_text, sim, nearest_comps)

        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown("##### 🎨 Creative Strategy Memo")
            st.info(memos.get("creative", "Creative memo generated."))
        with b2:
            st.markdown("##### 📡 Media Planning Memo")
            st.info(memos.get("media", "Media memo generated."))
        with b3:
            st.markdown("##### 💼 CFO Governance Memo")
            st.info(memos.get("cfo", "CFO memo generated."))

        st.markdown("##### 📋 Executive Consensus Pre-Buy Brief")
        consensus_text = memos.get("consensus", "Strategy board consensus brief.")
        st.success(consensus_text)

        memo_buf = io.BytesIO(consensus_text.encode("utf-8"))
        st.download_button(
            label="⬇️ Download Executive Decision Memo (.md)",
            data=memo_buf,
            file_name="SuperBowl_AI_Executive_Decision_Memo.md",
            mime="text/markdown",
        )
