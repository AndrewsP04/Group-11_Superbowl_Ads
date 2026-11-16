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
        GAME_FLOW_SCENARIOS,
        HISTORICAL_SUPERBOWL_STATS,
        QUARTER_VIEWERS_M,
        compute_effective_reach,
        compute_executive_roi_and_decision,
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
        GAME_FLOW_SCENARIOS,
        HISTORICAL_SUPERBOWL_STATS,
        QUARTER_VIEWERS_M,
        compute_effective_reach,
        compute_executive_roi_and_decision,
        load_bundle,
        optimize_creative_and_airtime,
        parse_concept_pitch_to_controls,
        retrieve_nearest_historical_ads,
        simulate_multi_pov,
    )


def _on_switch_quarter_click(target_quarter: str):
    st.session_state["t6_quarter_sel"] = target_quarter
    if "t6_adk_scenario_key" in st.session_state:
        del st.session_state["t6_adk_scenario_key"]


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
    if "t6_tier_sel" not in st.session_state:
        st.session_state["t6_tier_sel"] = "Enterprise Megabrand ($15M+ Budget)"
    if "t6_runtime_sel" not in st.session_state:
        st.session_state["t6_runtime_sel"] = 30
    if "t6_quarter_sel" not in st.session_state:
        st.session_state["t6_quarter_sel"] = "Game Q2"
    if "t6_game_flow_sel" not in st.session_state:
        st.session_state["t6_game_flow_sel"] = list(GAME_FLOW_SCENARIOS.keys())[0]
    if "t6_custom_spread_val" not in st.session_state:
        st.session_state["t6_custom_spread_val"] = 7
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
                    if parsed and not parsed.get("is_valid_brief", True):
                        st.error(f"⚠️ {parsed.get('error_message', 'Invalid brief. Please enter a valid commercial storyboard concept.')}")
                    elif parsed:
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

                        # 4. Game flow & Point margin matching
                        gf_in = str(parsed.get("game_flow", "")).lower()
                        if "nail" in gf_in or "close" in gf_in or "one-possession" in gf_in or "overtime" in gf_in:
                            st.session_state["t6_game_flow_sel"] = "⚡ One-Possession Nailbiter (0–7 pts · 56% Historical Odds)"
                        elif "blowout" in gf_in or "runaway" in gf_in:
                            st.session_state["t6_game_flow_sel"] = "🚨 Runaway Blowout Stress-Test (20+ pts · 16% Historical Odds)"
                        elif "comp" in gf_in or "two-possession" in gf_in:
                            st.session_state["t6_game_flow_sel"] = "⚖️ Competitive Two-Possession (8–14 pts · 28% Historical Odds)"
                        else:
                            st.session_state["t6_game_flow_sel"] = list(GAME_FLOW_SCENARIOS.keys())[0]
                        st.session_state["t6_custom_spread_val"] = int(np.clip(parsed.get("point_spread", 7), 0, 35))

                        # 5. Traits mapping (faithfully sync traits; if brief has none, set to empty)
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
                        st.session_state["t6_traits_sel"] = synced_traits

                        # 6. Continuous semantic scores
                        st.session_state["t6_arc_sel"] = float(np.clip(parsed.get("narrative_arc", 6.0), 1.0, 10.0))
                        st.session_state["t6_brand_sel"] = float(np.clip(parsed.get("brand_centrality", 7.5), 1.0, 10.0))
                        st.session_state["t6_cult_sel"] = float(np.clip(parsed.get("cultural_nostalgia", 5.0), 1.0, 10.0))

                        # 7. Brand scale tier matching
                        tier_in = parsed.get("brand_tier")
                        if tier_in:
                            tier_opts = [
                                "Enterprise Megabrand ($15M+ Budget)",
                                "Challenger / Growth Brand ($6M–$15M Budget)",
                                "Emerging / Non-Mainstream (<$6M Budget)",
                            ]
                            matched_tier = next(
                                (opt for opt in tier_opts if opt.split()[0].lower() in str(tier_in).lower() or ("<$6m" in str(tier_in).lower() and "<$6m" in opt.lower())),
                                None
                            )
                            if matched_tier:
                                st.session_state["t6_tier_sel"] = matched_tier

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
        sel_tier = st.selectbox(
            "Brand Capital & Scale Tier",
            [
                "Enterprise Megabrand ($15M+ Budget)",
                "Challenger / Growth Brand ($6M–$15M Budget)",
                "Emerging / Non-Mainstream (<$6M Budget)",
            ],
            format_func=lambda t: {
                "Enterprise Megabrand ($15M+ Budget)": "🏛️ Enterprise Megabrand ($15M+ Budget)",
                "Challenger / Growth Brand ($6M–$15M Budget)": "🚀 Challenger / Growth Brand ($6M–$15M Budget)",
                "Emerging / Non-Mainstream (<$6M Budget)": "🌱 Emerging / Non-Mainstream (<$6M Budget)",
            }.get(t, t),
            help="Scale tier dictates budget headroom: Megabrands can afford any quarter, Challengers should avoid inflated Q2, and Emerging brands (<$6M) should pivot to Social Blitz / Local DMA.",
            key="t6_tier_sel",
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
            format_func=lambda q: {
                "Game Q1": "Game Q1 · Prime Launchpad ($6.8M | 92M Viewers)",
                "Game Q2": "Game Q2 · Premium Scarcity ($7.8M | 104M Viewers)",
                "Game Q3": "Game Q3 · Value Storytelling ($6.4M | 96M Viewers)",
                "Game Q4": "Game Q4 · High-Beta Value ($5.8M | 101M Viewers)",
            }.get(q, q),
            key="t6_quarter_sel",
        )

    with c2:
        sel_game_flow = st.selectbox(
            "Game Flow & Telecast Uncertainty Model",
            options=list(GAME_FLOW_SCENARIOS.keys()),
            help="Models the ambiguity of football using 2000–2024 historical Super Bowl scores. Pre-buy ad slots are bought months in advance before finalists are known.",
            key="t6_game_flow_sel",
        )
        gf_config = GAME_FLOW_SCENARIOS[sel_game_flow]
        is_probabilistic_blend = gf_config.get("is_probabilistic_blend", False)

        if "Custom" in sel_game_flow:
            sel_spread = st.slider(
                "Manual Point Margin Stress-Test",
                min_value=0,
                max_value=35,
                format="%d pts",
                help="Higher spreads trigger monotonic living-room viewer decay in late quarters.",
                key="t6_custom_spread_val",
            )
        else:
            sel_spread = gf_config.get("effective_spread", 6)
            st.caption(f"📊 **Grounding**: {gf_config['description']}")

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

    with st.expander("🏈 Why is Game Flow modeled from 2000–2024 past Super Bowl scores? (Addressing Upfront Pre-Buy Ambiguity)", expanded=False):
        st.markdown(
            """
            **The Pre-Buy Real-World Reality:**  
            Super Bowl commercial airtime is contracted **6 to 9 months in advance** during network Upfronts (spring/early fall) or early season. 
            At contract signing:
            - **No one knows who will make the Super Bowl** (regular season hasn't concluded, playoff brackets don't exist).
            - **Vegas point spreads do not exist** (sportsbooks only release spreads 2 weeks prior to kickoff).
            - Expecting an upfront media buyer or CFO to forecast a specific score margin is unrealistic.

            **Empirical Football Ambiguity Modeling (2000–2024 Modern Era, 25 Games):**  
            Rather than requiring speculative guesswork, this engine models the telecast audience using the actual historical distribution of Super Bowl scores:
            """
        )
        gf_col1, gf_col2, gf_col3 = st.columns(3)
        with gf_col1:
            st.markdown(
                """
                <div style="background: rgba(166, 227, 161, 0.1); border: 1px solid #a6e3a144; border-radius: 6px; padding: 0.6rem;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #a6e3a1;">⚡ One-Possession Nailbiter</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #cdd6f4;">56.0% Odds</div>
                    <div style="font-size: 0.75rem; color: #bac2de;">14 of 25 modern games (≤7 pts · median 3.0 pts). Historic late-game ratings up to 125M+ viewers with 0% viewer decay.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with gf_col2:
            st.markdown(
                """
                <div style="background: rgba(137, 180, 250, 0.1); border: 1px solid #89b4fa44; border-radius: 6px; padding: 0.6rem;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #89b4fa;">⚖️ Competitive Two-Possession</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #cdd6f4;">28.0% Odds</div>
                    <div style="font-size: 0.75rem; color: #bac2de;">7 of 25 modern games (8–19 pts · median 11.0 pts). Steady telecast retention throughout the game.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with gf_col3:
            st.markdown(
                """
                <div style="background: rgba(243, 139, 168, 0.1); border: 1px solid #f38ba844; border-radius: 6px; padding: 0.6rem;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #f38ba8;">🚨 Runaway Blowout</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #cdd6f4;">16.0% Odds</div>
                    <div style="font-size: 0.75rem; color: #bac2de;">Only 4 of 25 games (≥20 pts · median 24.0 pts). Models tail-risk viewer channel surfing (-21.9% decay in Q4).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div style="background: rgba(24, 24, 37, 0.8); border: 1px solid #313244; border-radius: 6px; padding: 0.5rem 0.75rem; margin-top: 0.5rem; font-size: 0.78rem; color: #a6adc8;">
                🎲 <b>Default Engine Mode:</b> By default, the engine applies the <b>Empirical Weighted Blend (56% Close / 28% Competitive / 16% Blowout)</b> across all reach and ROI calculations, giving upfront media buyers a statistically sound expectation without gambling on who plays in February.
            </div>
            """,
            unsafe_allow_html=True,
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
        is_probabilistic_blend=is_probabilistic_blend,
        narrative_arc=sel_arc,
        brand_centrality=sel_brand_cent,
        cultural_nostalgia=sel_nostalgia,
        brand_tier=sel_tier,
    )

    active_pitch = st.session_state.get("tab6_pitch_input", "")
    live_yt_enabled = st.session_state.get("t6_live_yt_toggle", True)
    nearest_comps = retrieve_nearest_historical_ads(
        traits=trait_keys,
        vertical=sel_vertical,
        runtime_sec=sel_runtime,
        pitch_text=active_pitch,
        k=3,
        enable_live_augment=live_yt_enabled,
    )

    scenario_text = (
        f"Brand Vertical: {sel_vertical}, Brand Tier: {sel_tier}, Runtime: :{sel_runtime}s, Target Quarter: {sel_quarter}, "
        f"Game Flow Model: {gf_config['short_name']} (Point Margin: {sel_spread} pts, Probabilistic Blend: {is_probabilistic_blend}). Creative Traits: {', '.join(sel_traits) if sel_traits else 'None selected'}. "
        f"Narrative Arc: {sel_arc}/10, Brand Centrality: {sel_brand_cent}/10, Cultural Nostalgia: {sel_nostalgia}/10."
    )
    scenario_sig = f"{sel_vertical}_{sel_tier}_{sel_runtime}_{sel_quarter}_{sel_game_flow}_{sel_spread}_{sel_arc}_{sel_brand_cent}_{sel_nostalgia}_{'-'.join(sorted(trait_keys))}"

    # Compute the 3 Executive Pillars (Channel Allocation, Financial ROI & Budget Cap, Creative Recipe)
    exec_dec = compute_executive_roi_and_decision(
        sim=sim,
        vertical=sel_vertical,
        runtime_sec=sel_runtime,
        quarter=sel_quarter,
        point_spread=sel_spread,
        is_probabilistic_blend=is_probabilistic_blend,
        game_flow_scenario=gf_config["short_name"],
        traits=trait_keys,
        brand_centrality=sel_brand_cent,
        narrative_arc=sel_arc,
        cultural_nostalgia=sel_nostalgia,
        brand_tier=sel_tier,
    )

    # Maintain or generate strategic memos
    if st.session_state.get("t6_adk_scenario_key") == scenario_sig and "t6_adk_memos" in st.session_state:
        memos = st.session_state["t6_adk_memos"]
    else:
        memos = generate_baseline_strategy_memos(
            scenario_text,
            sim,
            nearest_comps,
            exec_dec=exec_dec,
            brand_tier=sel_tier,
            game_flow_scenario=gf_config["short_name"],
            is_probabilistic_blend=is_probabilistic_blend,
        )

    # ── 3. Executive Strategy Brief & Pre-Buy Decision Summary (TOP VIEW) ─────
    st.markdown("---")
    st.markdown("### 📋 Executive Strategy Brief & Pre-Buy Decision Summary")
    st.caption("Synthesized cross-functional decision brief for executive leadership: channel choice, financial ROI & spending limits, and winning creative playbook.")

    # 3 High-Impact Executive Pillar Cards (Replacing the previous 4 KPI row)
    c_p1, c_p2, c_p3 = st.columns(3)

    cv = exec_dec["channel_verdict"]
    fr = exec_dec["financial_roi"]
    cr = exec_dec["creative_recipe"]
    qa = exec_dec.get("quarter_advisory", {})
    ta = exec_dec.get("tier_advisory", {})

    with c_p1:
        emerging_solvency_box = ""
        if ta.get("is_emerging"):
            emerging_solvency_box = (
                f'<div style="background: rgba(250, 179, 135, 0.14); border-left: 3px solid #fab387; padding: 0.4rem 0.6rem; border-radius: 4px; color: #fab387; font-size: 0.76rem; margin-bottom: 0.5rem; line-height: 1.35;">'
                f'⚠️ <b>Brand Solvency Ceiling (Under $6M Budget):</b> National broadcast (${fr["total_capital_m"]}M) exceeds viable corporate reserves. 100% digital reallocation or Local Cable DMA buy is recommended.'
                f'</div>'
            )

        card1_html = (
            f'<div style="background: #181825; border: 1px solid #313244; border-top: 4px solid {cv["color"]}; border-radius: 8px; padding: 1.15rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
            f'<div>'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.25rem;">'
            f'<span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; color: #a6adc8; font-weight: 700;">1. Channel Choice</span>'
            f'<div style="display: flex; gap: 0.3rem;">'
            f'<span style="font-size: 0.70rem; padding: 2px 6px; border-radius: 4px; background: rgba(255, 255, 255, 0.08); color: {ta.get("tier_color", "#89b4fa")}; font-weight: 700; border: 1px solid {ta.get("tier_color", "#89b4fa")}44;">{ta.get("tier_badge", "Scale")}</span>'
            f'<span style="font-size: 0.72rem; padding: 2px 7px; border-radius: 12px; background: rgba(166, 227, 161, 0.15); color: {cv["color"]}; font-weight: 700; border: 1px solid {cv["color"]}55;">{cv["badge"]}</span>'
            f'</div>'
            f'</div>'
            f'{emerging_solvency_box}'
            f'<div style="font-size: 1.05rem; font-weight: 700; color: #cdd6f4; margin-bottom: 0.4rem; line-height: 1.3;">{cv["summary"]}</div>'
            f'<div style="font-size: 0.83rem; color: #bac2de; line-height: 1.45; margin-bottom: 0.5rem;">{cv["detail"]}</div>'
            f'<div style="background: rgba(30, 30, 46, 0.9); border: 1px solid #313244; border-radius: 6px; padding: 0.5rem 0.65rem; margin-bottom: 0.5rem; font-size: 0.77rem; line-height: 1.4;">'
            f'<div style="font-weight: 700; color: {qa.get("scarcity_color", "#fab387")}; margin-bottom: 0.15rem;">📍 {qa.get("selected_quarter", "Quarter")}: {qa.get("headline", "")}</div>'
            f'<div style="color: #a6adc8; margin-bottom: 0.35rem;">{qa.get("warning", "")}</div>'
            f'<div style="background: rgba(137, 180, 250, 0.12); border-left: 3px solid #89b4fa; padding: 0.3rem 0.5rem; border-radius: 4px; color: #89b4fa; font-weight: 600;">💡 <b>Allotment Pivot:</b> {qa.get("smart_alt_summary", "")}</div>'
            f'</div>'
            f'</div>'
            f'<div style="background: rgba(30, 30, 46, 0.85); border: 1px solid #313244; border-radius: 6px; padding: 0.55rem 0.7rem; font-size: 0.78rem; color: #a6adc8; line-height: 1.4;">💡 <b>Social Media Tradeoff:</b> {cv["social_comparison"]}</div>'
            f'</div>'
        )
        st.markdown(card1_html, unsafe_allow_html=True)

    with c_p2:
        roi_sign = "+" if fr['net_profit_m'] >= 0 else ""
        roi_color = "#a6e3a1" if fr['net_profit_m'] >= 0 else "#f38ba8"

        tier_financial_note = ""
        if ta.get("is_emerging"):
            tier_financial_note = (
                f'<div style="background: rgba(166, 227, 161, 0.12); border: 1px solid #a6e3a155; border-radius: 6px; padding: 0.45rem 0.65rem; font-size: 0.77rem; color: #a6e3a1; margin-top: 0.4rem; line-height: 1.35;">'
                f'🎯 <b>Recommended Digital Allocation:</b> Reallocate <b>${ta.get("digital_budget_k", 500)}K</b> into social creator blitz → Expected <b>${ta.get("digital_expected_value_m", 2.4)}M GMV ({ta.get("digital_roas", "4.8x ROAS")})</b>. Saves <b>${ta.get("capital_saved_m", 8.5)}M cash</b> vs national broadcast!'
                f'</div>'
            )
        elif ta.get("is_challenger") and sel_quarter == "Game Q2":
            q3_savings = round(fr['airtime_cost_m'] - QUARTER_BASE_COST_M['Game Q3'] * (sel_runtime / 30.0), 1)
            tier_financial_note = (
                f'<div style="background: rgba(137, 180, 250, 0.12); border: 1px solid #89b4fa55; border-radius: 6px; padding: 0.45rem 0.65rem; font-size: 0.77rem; color: #89b4fa; margin-top: 0.4rem; line-height: 1.35;">'
                f'💡 <b>Challenger Value Move:</b> Shifting from Q2 to Q3 saves <b>${q3_savings}M</b> to fund your digital retargeting blitz!'
                f'</div>'
            )

        card2_html = (
            f'<div style="background: #181825; border: 1px solid #313244; border-top: 4px solid #89b4fa; border-radius: 8px; padding: 1.15rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
            f'<div>'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">'
            f'<span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; color: #a6adc8; font-weight: 700;">2. Financial ROI & Budget Cap</span>'
            f'<span style="font-size: 0.72rem; padding: 2px 7px; border-radius: 12px; background: rgba(137, 180, 250, 0.15); color: #89b4fa; font-weight: 700; border: 1px solid #89b4fa55;">💰 {fr["roi_multiplier"]}x ROI Return</span>'
            f'</div>'
            f'<div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.2rem;">'
            f'<span style="font-size: 1.65rem; font-weight: 800; color: #cdd6f4;">${fr["expected_gross_value_m"]}M</span>'
            f'<span style="font-size: 0.85rem; color: {roi_color}; font-weight: 700;">({roi_sign}{fr["roi_pct"]}% Net ROI)</span>'
            f'</div>'
            f'<div style="font-size: 0.77rem; color: #a6adc8; margin-bottom: 0.6rem;">Expected Commercial & Earned Media Brand Value</div>'
            f'<div style="font-size: 0.82rem; color: #cdd6f4; line-height: 1.5; border-top: 1px solid #313244; padding-top: 0.5rem;">'
            f'• <b>Max Viable Budget:</b> <span style="color: #fab387; font-weight: 700;">${fr["max_viable_budget_m"]}M Spend Ceiling</span><br/>'
            f'• <b>Capital Outlay:</b> ${fr["total_capital_m"]}M (${fr["airtime_cost_m"]}M Airtime + ${fr["prod_cost_m"]}M Prod)<br/>'
            f'• <b>Downside Safety Floor:</b> ${fr["downside_floor_m"]}M (10th %ile Capital Protection)'
            f'</div>'
            f'</div>'
            f'<div>'
            f'<div style="background: rgba(30, 30, 46, 0.85); border: 1px solid #313244; border-radius: 6px; padding: 0.45rem 0.65rem; font-size: 0.78rem; color: #a6adc8; margin-top: 0.4rem;">'
            f'💵 <b>Breakeven Spend:</b> <b>${fr["cost_per_viewer"]} / viewer reached</b> (${sim["effective_cpm_usd"]} CPM)'
            f'</div>'
            f'{tier_financial_note}'
            f'</div>'
            f'</div>'
        )
        st.markdown(card2_html, unsafe_allow_html=True)

    with c_p3:
        bullets_html = "".join([f"<li style='margin-bottom: 0.25rem;'><b>{title}:</b> <span style='color: #bac2de;'>{desc}</span></li>" for title, desc in cr['recipe_elements']])
        card3_html = (
            f'<div style="background: #181825; border: 1px solid #313244; border-top: 4px solid #fab387; border-radius: 8px; padding: 1.15rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">'
            f'<div>'
            f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">'
            f'<span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.8px; color: #a6adc8; font-weight: 700;">3. Creative Content Recipe</span>'
            f'<span style="font-size: 0.72rem; padding: 2px 7px; border-radius: 12px; background: rgba(250, 179, 135, 0.15); color: #fab387; font-weight: 700; border: 1px solid #fab38755;">⚡ What to Put in Ad</span>'
            f'</div>'
            f'<div style="font-size: 0.88rem; font-weight: 700; color: #fab387; margin-bottom: 0.35rem;">{cr["formula_title"]}</div>'
            f'<ul style="font-size: 0.78rem; color: #cdd6f4; padding-left: 1.1rem; margin-bottom: 0.5rem; line-height: 1.35;">{bullets_html}</ul>'
            f'</div>'
            f'<div>'
            f'<div style="font-size: 0.77rem; color: #f38ba8; margin-bottom: 0.35rem; line-height: 1.3;">⚠️ <b>{cr["avoid_elements"]}</b></div>'
            f'<div style="background: rgba(30, 30, 46, 0.85); border: 1px solid #313244; border-radius: 6px; padding: 0.45rem 0.65rem; font-size: 0.76rem; color: #a6e3a1; font-weight: 600;">🚀 {cr["immediate_action"]}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card3_html, unsafe_allow_html=True)

    # Streamlined Supporting Telemetry Ribbon
    st.markdown(
        f"""
        <div style="background: rgba(24, 24, 37, 0.6); border: 1px solid #313244; border-radius: 6px; padding: 0.45rem 0.9rem; margin: 0.75rem 0 1rem 0; font-size: 0.79rem; color: #a6adc8; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
            <span>📡 <b>Telecast Delivery:</b> <span style="color: #cdd6f4; font-weight: 600;">{sim['effective_reach_m']}M viewers ({sim['retention_pct']}% retention)</span></span>
            <span>💵 <b>Effective CPM:</b> <span style="color: #cdd6f4; font-weight: 600;">${sim['effective_cpm_usd']}</span></span>
            <span>📈 <b>Predicted YouTube Engagement:</b> <span style="color: #cdd6f4; font-weight: 600;">{sim['pred_eng_pct']}% [CI: {sim['eng_ci_low']}%–{sim['eng_ci_high']}%]</span></span>
            <span>🔍 <b>Search Surge:</b> <span style="color: #cdd6f4; font-weight: 600;">{sim['pred_wiki_mult']}x</span></span>
            <span>🛡️ <b>Downside Risk Floor:</b> <span style="color: #cdd6f4; font-weight: 600;">{sim['eng_ci_low']}%</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Intelligent Airtime Allotment & Quarter Feasibility Matrix ───────────
    with st.expander("🧭 Intelligent Airtime Allotment & Quarter Feasibility Matrix (Compare All 4 Quarters)", expanded=True):
        st.caption("Commercial inventory availability, upfront market pricing, and strategic positioning trade-offs across all 4 broadcast quarters:")
        q_cols = st.columns(4)
        for q_idx, q_row in enumerate(qa.get("matrix", [])):
            with q_cols[q_idx]:
                is_sel = q_row["is_selected"]
                card_border = "2px solid #89b4fa" if is_sel else "1px solid #313244"
                card_bg = "rgba(137, 180, 250, 0.08)" if is_sel else "#181825"
                active_badge = '<span style="font-size: 0.68rem; background: #89b4fa; color: #11111b; font-weight: 800; padding: 1px 6px; border-radius: 4px; margin-left: 0.3rem;">ACTIVE</span>' if is_sel else ''

                delta_str = f"+${q_row['cost_delta_vs_q2']}M" if q_row['cost_delta_vs_q2'] > 0 else (f"-${abs(q_row['cost_delta_vs_q2'])}M vs Q2" if q_row['cost_delta_vs_q2'] < 0 else "Peak Baseline")
                delta_color = "#f38ba8" if q_row['cost_delta_vs_q2'] > 0 else ("#a6e3a1" if q_row['cost_delta_vs_q2'] < 0 else "#fab387")

                st.markdown(
                    f"""
                    <div style="background: {card_bg}; border: {card_border}; border-radius: 8px; padding: 0.85rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                                <span style="font-size: 0.95rem; font-weight: 800; color: #cdd6f4;">{q_row['quarter']} {active_badge}</span>
                                <span style="font-size: 0.72rem; font-weight: 700; color: {delta_color};">{delta_str}</span>
                            </div>
                            <div style="font-size: 1.25rem; font-weight: 800; color: #89b4fa; margin-bottom: 0.2rem;">
                                ${q_row['cost_m']}M <span style="font-size: 0.75rem; color: #a6adc8; font-weight: 500;">(:{sel_runtime}s)</span>
                            </div>
                            <div style="font-size: 0.76rem; color: #fab387; font-weight: 700; margin-bottom: 0.4rem;">
                                {q_row['scarcity']}
                            </div>
                            <div style="font-size: 0.75rem; color: #a6adc8; line-height: 1.4; border-top: 1px solid #313244; padding-top: 0.4rem; margin-bottom: 0.5rem;">
                                • <b>Reach:</b> {q_row['reach_m']}M ({q_row['retention_pct']}% ret)<br/>
                                • <b>CPM:</b> ${q_row['cpm']}<br/>
                                • <b>Allotment:</b> {q_row['allotment_status']}
                            </div>
                            <div style="font-size: 0.74rem; color: #bac2de; line-height: 1.35; background: rgba(30, 30, 46, 0.7); padding: 0.4rem; border-radius: 4px; margin-bottom: 0.4rem;">
                                🎯 <b>Best For:</b> {q_row['ideal_for']}
                            </div>
                            <div style="font-size: 0.73rem; color: #cdd6f4; line-height: 1.35; background: rgba(24, 24, 37, 0.9); padding: 0.4rem; border-radius: 4px; border-left: 3px solid #89b4fa;">
                                🏛️ <b>Scale Feasibility:</b> <span style="font-weight: 700;">{q_row.get('tier_feasibility', '')}</span><br/>
                                <span style="color: #a6adc8;">{q_row.get('tier_action', '')}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if not is_sel:
                    st.button(
                        f"⚡ Switch to {q_row['quarter']}",
                        key=f"t6_switch_q_{q_idx}",
                        on_click=_on_switch_quarter_click,
                        args=(q_row["quarter"],),
                        use_container_width=True,
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

    # ── 4. Historical Precedents & Live Comps (Vector RAG) (MIDDLE) ───────────
    st.markdown("---")
    col_t3_title, col_t3_toggle = st.columns([2.2, 1.0])
    with col_t3_title:
        st.markdown("### 🎞️ Historical Precedents & Big Game Comps (Vector RAG)")
        st.caption("Closest Super Bowl commercial spots combining 25-year telecast archives with real-time YouTube Big Game pulls:")
    with col_t3_toggle:
        live_toggle_val = st.toggle(
            "⚡ Live YouTube Augment",
            value=st.session_state.get("t6_live_yt_toggle", True),
            help="When active, dynamically queries YouTube Data API v3 to pull real-world commercials for modern categories (e.g. Consumer Tech, AI, Crypto, Apps) unrepresented in the historical 2000-2020 archive.",
            key="t6_live_yt_toggle",
        )

    TRAIT_BADGE_CONFIG = {
        "Humor": {"icon": "😂", "bg": "rgba(249, 226, 175, 0.15)", "fg": "#f9e2af", "border": "#f9e2af55"},
        "Celebrity": {"icon": "⭐", "bg": "rgba(203, 166, 247, 0.15)", "fg": "#cba6f7", "border": "#cba6f755"},
        "Product-Forward": {"icon": "📦", "bg": "rgba(137, 220, 235, 0.15)", "fg": "#89dceb", "border": "#89dceb55"},
        "Patriotic": {"icon": "🇺🇸", "bg": "rgba(243, 139, 168, 0.15)", "fg": "#f38ba8", "border": "#f38ba855"},
        "Animals": {"icon": "🐾", "bg": "rgba(166, 227, 161, 0.15)", "fg": "#a6e3a1", "border": "#a6e3a155"},
        "Danger/Action": {"icon": "⚡", "bg": "rgba(235, 160, 172, 0.15)", "fg": "#eba0ac", "border": "#eba0ac55"},
        "Sex Appeal": {"icon": "✨", "bg": "rgba(245, 194, 231, 0.15)", "fg": "#f5c2e7", "border": "#f5c2e755"},
    }

    comp_cols = st.columns(len(nearest_comps))
    for col_idx, comp in enumerate(nearest_comps):
        with comp_cols[col_idx]:
            trait_names = comp.get("trait_names", [])
            if trait_names:
                pills = []
                for t_name in trait_names:
                    cfg = TRAIT_BADGE_CONFIG.get(t_name, {"icon": "🏷️", "bg": "rgba(147, 153, 178, 0.15)", "fg": "#9399b2", "border": "#9399b255"})
                    pills.append(
                        f'<span style="font-size: 0.72rem; padding: 2px 7px; border-radius: 12px; background: {cfg["bg"]}; color: {cfg["fg"]}; font-weight: 600; border: 1px solid {cfg["border"]};">{cfg["icon"]} {t_name}</span>'
                    )
                traits_html = " ".join(pills)
            else:
                traits_html = '<span style="font-size: 0.72rem; padding: 2px 7px; border-radius: 12px; background: rgba(180, 190, 254, 0.15); color: #b4befe; font-weight: 600; border: 1px solid #b4befe55;">📖 Pure Narrative Story</span>'

            spot_fmt = comp.get("spot_format", f":{comp.get('runtime_sec', 30)}s spot")
            approach_desc = comp.get("approach", "Cinematic narrative storytelling")
            is_live = comp.get("source") == "live"
            source_badge = (
                '<span style="font-size: 0.68rem; padding: 2px 6px; border-radius: 4px; background: rgba(166, 227, 161, 0.18); color: #a6e3a1; font-weight: 700; border: 1px solid #a6e3a155;">📡 Live Registry</span>'
                if is_live
                else '<span style="font-size: 0.68rem; padding: 2px 6px; border-radius: 4px; background: rgba(137, 180, 250, 0.18); color: #89b4fa; font-weight: 700; border: 1px solid #89b4fa55;">🏛️ 25-Yr Archive</span>'
            )

            metric_2_label = "Live Video Telemetry" if is_live else "Wikipedia Surge"
            metric_2_val = comp.get("views_display") if (is_live and comp.get("views_display")) else comp.get("wiki_spike_display", str(comp.get("wiki_spike", "1.0")) + "x")

            st.markdown(
                f"""
                <div style="background: #181825; border: 1px solid #313244; border-radius: 8px; padding: 1.1rem; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                            <div style="display: flex; align-items: center; gap: 0.4rem;">
                                <span style="font-size: 0.8rem; color: #89b4fa; font-weight: 700;">{comp['year']} · {comp['brand'][:18]}</span>
                                {source_badge}
                            </div>
                            <span style="font-size: 0.73rem; color: #a6adc8; background: #313244; padding: 2px 7px; border-radius: 4px; font-weight: 600;">{spot_fmt}</span>
                        </div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #cdd6f4; margin: 0.35rem 0 0.5rem 0; min-height: 2.8rem; line-height: 1.35;">
                            {comp['title'][:65]}
                        </div>
                        <div style="background: rgba(30, 30, 46, 0.85); border: 1px solid #313244; border-radius: 6px; padding: 0.45rem 0.65rem; margin-bottom: 0.6rem;">
                            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; color: #a6adc8; font-weight: 600;">Creative Approach</div>
                            <div style="font-size: 0.82rem; font-weight: 600; color: #fab387; margin-top: 0.15rem; line-height: 1.3;">
                                🎯 {approach_desc}
                            </div>
                        </div>
                        <div style="margin-bottom: 0.7rem;">
                            <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px; color: #a6adc8; font-weight: 600; margin-bottom: 0.25rem;">Traits Detected</div>
                            <div style="display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;">
                                {traits_html}
                            </div>
                        </div>
                    </div>
                    <div style="font-size: 0.82rem; color: #a6adc8; line-height: 1.55; border-top: 1px solid #313244; padding-top: 0.55rem; margin-top: 0.3rem;">
                        • YouTube Engagement: <b>{comp['engagement_rate']}%</b><br/>
                        • {metric_2_label}: <b>{metric_2_val}</b><br/>
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
            "Game Flow Model",
            gf_config.get("badge", f"{sel_spread} pts"),
            delta=f"Margin: {sel_spread} pts" if "Custom" in sel_game_flow else "Empirical Super Bowl Comp",
            delta_color="inverse" if sel_spread > 12 and "Q4" in sel_quarter else "normal",
        )

        qtr_names = ["Game Q1", "Game Q2", "Game Q3", "Game Q4"]
        reaches = [compute_effective_reach(q, sel_spread, is_probabilistic_blend=is_probabilistic_blend)[0] for q in qtr_names]
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
            title=f"Telecast Audience by Quarter ({gf_config['short_name']} · Margin: {sel_spread} pts)",
        )
        st.plotly_chart(fig_reach, use_container_width=True)

        st.markdown("##### 🚀 Prescriptive 512-Grid Optimizer: Top 3 Airtime & Creative Blueprints")
        opt_blueprints = optimize_creative_and_airtime(sel_vertical, sel_spread, is_probabilistic_blend=is_probabilistic_blend, top_n=3)
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
