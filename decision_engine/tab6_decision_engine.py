"""
tab6_decision_engine.py - Streamlit UI View for the Multi-POV Super Bowl Decision Engine.

Renders Tab 6 in app.py:
1. Auto-training check on first launch (no manual training step required)
2. Natural language storyboard/pitch parser expander (Gemini 3.5 Flash Lite)
3. Interactive multi-parameter simulation sandbox (<20ms response)
4. Executive Strategy Brief & Pre-Buy Decision Summary (Top view for decision-makers)
5. Vector RAG Top-3 Nearest Historical Commercials cards (Historical Precedents)
6. Deep-Dive Stakeholder Perspectives & Specialist Audits (Pushed to the end):
   - Creative Strategy Director (AI Memo + Engagement, Search Surge, EBM Waterfall)
   - Media Buyer / Planner (AI Memo + Telecast flow, Blowout decay, 512-Grid Optimizer)
   - Brand CMO / CFO Risk (AI Memo + Downside 80% CI floor, Non-causal disclosures)
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    from .adk_boardroom import generate_baseline_strategy_memos, run_strategy_board
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
    from adk_boardroom import generate_baseline_strategy_memos, run_strategy_board
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

                        # Invalidate cached ADK debate to force fresh synthesis for new pitch
                        if "t6_adk_scenario_key" in st.session_state:
                            del st.session_state["t6_adk_scenario_key"]

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

    scenario_text = (
        f"Brand Vertical: {sel_vertical}, Runtime: :{sel_runtime}s, Target Quarter: {sel_quarter}, "
        f"Point Spread: {sel_spread} pts. Creative Traits: {', '.join(sel_traits) if sel_traits else 'None selected'}. "
        f"Narrative Arc: {sel_arc}/10, Brand Centrality: {sel_brand_cent}/10, Cultural Nostalgia: {sel_nostalgia}/10."
    )
    scenario_sig = f"{sel_vertical}_{sel_runtime}_{sel_quarter}_{sel_spread}_{sel_arc}_{sel_brand_cent}_{sel_nostalgia}_{'-'.join(sorted(trait_keys))}"

    # Maintain or generate strategic memos
    if st.session_state.get("t6_adk_scenario_key") == scenario_sig and "t6_adk_memos" in st.session_state:
        memos = st.session_state["t6_adk_memos"]
    else:
        memos = generate_baseline_strategy_memos(scenario_text, sim, nearest_comps)

    # ── 3. Executive Strategy Brief & Pre-Buy Decision Summary (TOP VIEW) ─────
    st.markdown("---")
    st.markdown("### 📋 Executive Strategy Brief & Pre-Buy Decision Summary")
    st.caption("Synthesized cross-functional decision brief for executive leadership: core KPIs, strategic mandates, and governance clearance.")

    # Headline 4-KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            "Predicted YouTube Engagement",
            f"{sim['pred_eng_pct']}%",
            delta=f"80% CI: {sim['eng_ci_low']}% – {sim['eng_ci_high']}%",
        )
    with kpi2:
        st.metric(
            "Predicted Wikipedia Surge",
            f"{sim['pred_wiki_mult']}x",
            delta=f"80% CI: {sim['wiki_ci_low']}x – {sim['wiki_ci_high']}x",
        )
    with kpi3:
        st.metric(
            f"Effective Telecast Reach ({sel_quarter})",
            f"{sim['effective_reach_m']}M",
            delta=f"Retention: {sim['retention_pct']}% (${sim['effective_cpm_usd']} CPM)",
        )
    with kpi4:
        st.metric(
            "Downside Risk Floor (10th %ile)",
            f"{sim['eng_ci_low']}%",
            delta="Lower 80% Conformal Bound",
            delta_color="off",
        )

    # Executive Consensus Brief Box
    consensus_text = memos.get("consensus", "Executive strategy brief is being calibrated.")
    st.markdown(
        f"""
        <div style="background: #1e1e2e; border: 1px solid #45475a; border-left: 5px solid #89b4fa; border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1rem 0;">
            <div style="font-size: 0.95rem; line-height: 1.6; color: #cdd6f4;">
                {consensus_text.replace(chr(10), '<br/>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action Buttons: Convene Live Board & Download Memo
    btn_col1, btn_col2 = st.columns([1.3, 1.0])
    with btn_col1:
        if st.button("⚡ Convene Live AI Strategy Board (Gemini 3.8 Flash)", key="t6_convene_live_adk_btn"):
            with st.spinner("Executing Google ADK Multi-Agent Boardroom debate on Vertex AI..."):
                live_memos = run_strategy_board(scenario_text, sim, nearest_comps)
                st.session_state["t6_adk_memos"] = live_memos
                st.session_state["t6_adk_scenario_key"] = scenario_sig
                st.toast("✅ AI Strategy Board convened! Executive brief & specialist memos updated.", icon="🏛️")
                st.rerun()

    with btn_col2:
        full_memo_md = (
            f"# Super Bowl Advertising Executive Strategy Brief\n\n"
            f"**Scenario:** {scenario_text}\n\n"
            f"{consensus_text}\n\n"
            f"---\n\n"
            f"## 🎨 Creative Strategy Memo (Creative Director Agent)\n{memos.get('creative', '')}\n\n"
            f"---\n\n"
            f"## 📡 Media Planning Memo (Media Buyer Agent)\n{memos.get('media', '')}\n\n"
            f"---\n\n"
            f"## 💼 CFO Governance Memo (CFO Risk Agent)\n{memos.get('cfo', '')}\n"
        )
        memo_buf = io.BytesIO(full_memo_md.encode("utf-8"))
        st.download_button(
            label="⬇️ Download Executive Decision Memo (.md)",
            data=memo_buf,
            file_name="SuperBowl_AI_Executive_Decision_Memo.md",
            mime="text/markdown",
            key="t6_dl_memo_btn",
        )

    # ── 4. Historical Precedents (Vector RAG Comps) (MIDDLE) ───────────────────
    st.markdown("---")
    st.markdown("### 🎞️ Historical Precedents (Vector RAG Comps)")
    st.caption("Top 3 closest actual Super Bowl commercials retrieved via 3072-dimensional embedding similarity from 25 years of telecasts:")

    comp_cols = st.columns(len(nearest_comps))
    for col_idx, comp in enumerate(nearest_comps):
        with comp_cols[col_idx]:
            st.markdown(
                f"""
                <div style="background: #181825; border: 1px solid #313244; border-radius: 8px; padding: 1.1rem; height: 100%;">
                    <div style="font-size: 0.8rem; color: #89b4fa; font-weight: 700;">{comp['year']} · {comp['brand']}</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #cdd6f4; margin: 0.35rem 0 0.5rem 0; min-height: 2.8rem;">
                        {comp['title'][:65]}
                    </div>
                    <div style="font-size: 0.83rem; color: #a6adc8; line-height: 1.55;">
                        • YouTube Engagement: <b>{comp['engagement_rate']}%</b><br/>
                        • Wikipedia Surge: <b>{comp.get('wiki_spike_display', str(comp.get('wiki_spike', '1.0')) + 'x')}</b><br/>
                        • Match Similarity: <b>{int(comp['similarity'] * 100)}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            target_url = comp.get("url", "")
            if target_url and target_url.startswith("http"):
                if st.button("▶️ Play in App", key=f"t6_play_btn_{col_idx}", use_container_width=True):
                    st.session_state["t6_active_video_url"] = target_url
                    st.session_state["t6_active_video_title"] = f"{comp['year']} · {comp['brand']} — {comp['title']}"
                    st.rerun()

    # In-App Theater Video Player (when a spot is selected to play inside the app)
    active_video_url = st.session_state.get("t6_active_video_url")
    active_video_title = st.session_state.get("t6_active_video_title", "Commercial Spot")
    if active_video_url:
        st.markdown(
            f"""
            <div style="background: #181825; border: 1px solid #89b4fa; border-left: 5px solid #a6e3a1; border-radius: 8px; padding: 0.9rem 1.2rem; margin: 1.2rem 0 0.5rem 0;">
                <div style="font-size: 1rem; font-weight: 700; color: #cdd6f4;">
                    📺 In-App Commercial Player: <span style="color: #a6e3a1;">{active_video_title}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "youtube.com" in active_video_url or "youtu.be" in active_video_url:
            st.video(active_video_url)
        else:
            st.components.v1.iframe(active_video_url, height=450)

        if st.button("✖️ Close Video Player", key="t6_close_video_player"):
            st.session_state["t6_active_video_url"] = None
            st.session_state["t6_active_video_title"] = None
            st.rerun()

    # ── 5. Deep-Dive Stakeholder Perspectives & Specialist Audits (END) ────────
    st.markdown("---")
    st.markdown("### 🔍 Deep-Dive Stakeholder Perspectives & Specialist Audits")
    st.caption("Granular domain analysis integrating Google ADK specialist agent memos directly with interactive statistical telemetry.")

    lens_tab1, lens_tab2, lens_tab3 = st.tabs([
        "🎨 Creative Strategy Director",
        "📡 Media Buyer / Planner",
        "💼 Brand CMO / CFO Risk",
    ])

    # ── Lens Tab 1: Creative Strategy Director ────────────────────────────────
    with lens_tab1:
        st.markdown("##### 🤖 Creative Strategy Memo (Google ADK Agent)")
        st.info(memos.get("creative", "Creative strategy memo is calibrated to concept brief."))

        st.markdown("##### 📊 Creative Performance Telemetry")
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric(
            "Predicted YouTube Engagement Rate",
            f"{sim['pred_eng_pct']}%",
            delta=f"80% CI: {sim['eng_ci_low']}% – {sim['eng_ci_high']}%",
        )
        cm2.metric(
            "Predicted Wikipedia Search Surge",
            f"{sim['pred_wiki_mult']}x",
            delta=f"80% CI: {sim['wiki_ci_low']}x – {sim['wiki_ci_high']}x",
        )
        cm3.metric(
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

    # ── Lens Tab 2: Media Buyer / Planner ──────────────────────────────────────
    with lens_tab2:
        st.markdown("##### 🤖 Media Planning Memo (Google ADK Agent)")
        st.info(memos.get("media", "Media planning memo is calibrated to telecast parameters."))

        st.markdown("##### 📊 Telecast Reach & Audience Decay Analysis")
        mm1, mm2, mm3 = st.columns(3)
        mm1.metric(
            f"Effective Living-Room Reach ({sel_quarter})",
            f"{sim['effective_reach_m']}M viewers",
            delta=f"Audience Retention: {sim['retention_pct']}%",
        )
        mm2.metric(
            "Effective Broadcast CPM",
            f"${sim['effective_cpm_usd']}",
            delta="Based on $7M baseline media spend",
        )
        mm3.metric(
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

    # ── Lens Tab 3: Brand CMO / CFO Risk ──────────────────────────────────────
    with lens_tab3:
        st.markdown("##### 🤖 CFO Governance Memo (Google ADK Agent)")
        st.info(memos.get("cfo", "CFO governance memo is calibrated to risk constraints."))

        st.markdown("##### 📊 Capital Risk Floors & Conformal Bounds")
        rm1, rm2, rm3 = st.columns(3)
        rm1.metric(
            "Downside Engagement Floor (10th %ile)",
            f"{sim['eng_ci_low']}%",
            delta="Lower 80% Conformal Bound",
            delta_color="off",
        )
        rm2.metric(
            "Downside Search Surge Floor (10th %ile)",
            f"{sim['wiki_ci_low']}x",
            delta="Lower 80% Conformal Bound",
            delta_color="off",
        )
        rm3.metric(
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
