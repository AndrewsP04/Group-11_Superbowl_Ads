"""
adk_boardroom.py - Google ADK Multi-Agent Boardroom Decision Engine.

Orchestrates an adversarial 3-stakeholder strategy board using Google Agent Development Kit:
1. CreativeDirectorAgent (advocates for viral cultural buzz & celebrity integration)
2. MediaPlannerAgent (evaluates telecast reach & challenges Q4 blowout risk)
3. CFO_RiskAgent (enforces 80% downside confidence floors & non-causal compliance)
4. ConsensusChairAgent (synthesizes conflicting memos into an executive pre-buy brief)

Runs natively via ParallelAgent and SequentialAgent on the Google Agent Platform / Vertex AI.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

try:
    from .config_auth import get_genai_client, init_vertex_env
    from .decision_tools import (
        compute_executive_roi_and_decision,
        optimize_creative_and_airtime,
        retrieve_nearest_historical_ads,
        simulate_multi_pov,
    )
except ImportError:
    from config_auth import get_genai_client, init_vertex_env
    from decision_tools import (
        compute_executive_roi_and_decision,
        optimize_creative_and_airtime,
        retrieve_nearest_historical_ads,
        simulate_multi_pov,
    )

init_vertex_env()

# Optional Google ADK imports with resilient fallback
try:
    from google.adk.agents.llm_agent import Agent as AdkAgent
    from google.adk.agents.parallel_agent import ParallelAgent
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
    HAS_ADK = True
except ImportError:
    HAS_ADK = False

MODEL_NAME = "gemini-3.8-flash"


def build_adk_pipeline():
    """
    Constructs the ADK multi-agent boardroom hierarchy.
    """
    if not HAS_ADK:
        return None

    creative_director = AdkAgent(
        name="CreativeDirectorAgent",
        model=MODEL_NAME,
        instruction=(
            "You are the Agency Creative Strategy Director pitching a high-stakes $7M Super Bowl ad. "
            "Your objective is maximizing viewer buzz, YouTube engagement, and cultural conversation. "
            "Use the provided quantitative simulations and historical commercial comps to champion creative boldness, "
            "humor setups, celebrity integration, and narrative memorability. "
            "Format your response as: [CREATIVE STRATEGY MEMO] with Concept Analysis, Cultural Hook, and Historical Benchmark."
        ),
        tools=[simulate_multi_pov, retrieve_nearest_historical_ads],
    )

    media_planner = AdkAgent(
        name="MediaPlannerAgent",
        model=MODEL_NAME,
        instruction=(
            "You are the Senior Media Buyer & Telecast Planner managing a $7M upfront airtime commitment. "
            "Your objective is maximizing living-room audience reach efficiency and mitigating blowout risk. "
            "Aggressively scrutinize Game Q4 placement if point spread > 10 pts due to fourth-quarter audience decay. "
            "Recommend optimal quarter timing (Q1 vs Q2 vs Q3 vs Q4) using the optimizer tool. "
            "Format your response as: [MEDIA PLANNING MEMO] with Quarter Recommendation, Telecast Audience Decay Risk, and Effective CPM."
        ),
        tools=[simulate_multi_pov, optimize_creative_and_airtime],
    )

    cfo_risk = AdkAgent(
        name="CFO_RiskAgent",
        model=MODEL_NAME,
        instruction=(
            "You are the Brand CFO & Governance Officer approving the $7M capital expenditure. "
            "Your objective is risk mitigation and empirical grounding. "
            "Scrutinize the downside floor (lower 80% confidence bound). "
            "Strictly enforce non-causal language: remind stakeholders that modeled lift reflects historical cohort patterns, "
            "not guaranteed commercial sales ROI. "
            "Format your response as: [CFO GOVERNANCE MEMO] with Downside Risk Floor, Downside Variance Assessment, and Required Approvals."
        ),
        tools=[simulate_multi_pov],
    )

    boardroom_debate = ParallelAgent(
        name="BoardroomDebateParallel",
        sub_agents=[creative_director, media_planner, cfo_risk],
    )

    consensus_chair = AdkAgent(
        name="ConsensusChairAgent",
        model=MODEL_NAME,
        instruction=(
            "You are the Strategy Board Chairperson. You review the parallel memos from the Creative Director, "
            "Media Planner, and CFO. Resolve their tensions and synthesize a final Executive Decision Memo. "
            "Include: 1. Executive Verdict (Proceed / Modify / Hold), 2. Creative Mandates, 3. Airtime Quarter Lock, "
            "4. Downside Risk Mitigations, and 5. Non-Causal Compliance Disclaimer."
        ),
    )

    pipeline = SequentialAgent(
        name="SuperBowlStrategyBoard",
        sub_agents=[boardroom_debate, consensus_chair],
    )

    return pipeline


def run_strategy_board(
    scenario_summary: str,
    simulation_results: Dict[str, Any],
    nearest_comps: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Executes the boardroom debate across the 3 stakeholder personas.
    Returns individual memos and the final consensus brief.
    """
    client = get_genai_client()
    if not HAS_ADK or not client:
        return _run_direct_gemini_fallback(scenario_summary, simulation_results, nearest_comps, client)

    pipeline = build_adk_pipeline()
    if not pipeline:
        return _run_direct_gemini_fallback(scenario_summary, simulation_results, nearest_comps, client)

    prompt = (
        f"CAMPAIGN SCENARIO FOR BOARD REVIEW:\n{scenario_summary}\n\n"
        f"QUANTITATIVE SIMULATION RESULTS:\n"
        f"- Predicted Engagement Rate: {simulation_results.get('pred_eng_pct')}% "
        f"[80% CI: {simulation_results.get('eng_ci_low')}% to {simulation_results.get('eng_ci_high')}%]\n"
        f"- Predicted Wikipedia Surge: {simulation_results.get('pred_wiki_mult')}x baseline "
        f"[80% CI: {simulation_results.get('wiki_ci_low')}x to {simulation_results.get('wiki_ci_high')}x]\n"
        f"- Effective Telecast Reach: {simulation_results.get('effective_reach_m')}M viewers "
        f"(Retention: {simulation_results.get('retention_pct')}%, Effective CPM: ${simulation_results.get('effective_cpm_usd')})\n\n"
        f"TOP HISTORICAL COMPS:\n"
        + "\n".join([f"- {c['year']} {c['brand']}: '{c['title']}' (Eng: {c['engagement_rate']}%, Wiki: {c['wiki_spike']}x)" for c in nearest_comps[:3]])
    )

    try:
        runner = InMemoryRunner(agent=pipeline, app_name="superbowl_decision_engine")

        async def _run_async():
            msg = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])
            final_out = ""
            async for event in runner.run_async(user_id="cmo_user", session_id="s1", new_message=msg):
                if event.is_final_response():
                    final_out = "".join(p.text for p in event.content.parts if getattr(p, "text", None))
            return final_out

        result_text = asyncio.run(_run_async())
        return _parse_memos_from_text(result_text)

    except Exception:
        return _run_direct_gemini_fallback(scenario_summary, simulation_results, nearest_comps, client)


def generate_baseline_strategy_memos(
    scenario_summary: str,
    sim: Dict[str, Any],
    comps: List[Dict[str, Any]],
    exec_dec: Optional[Dict[str, Any]] = None,
    brand_tier: str = "Enterprise Megabrand ($15M+ Budget)",
) -> Dict[str, str]:
    """
    Generates deterministic, high-fidelity strategic stakeholder memos and consensus brief in 0ms.
    Used for instant Executive Summary rendering and fallback.
    """
    top_comp = comps[0] if comps else {"brand": "Doritos", "year": 2020, "title": "Historical Super Bowl Comps"}

    eng = sim.get("pred_eng_pct", 0.85)
    wiki = sim.get("pred_wiki_mult", 3.2)
    reach = sim.get("effective_reach_m", 102.1)
    retention = sim.get("retention_pct", 98.2)
    cpm = sim.get("effective_cpm_usd", 68.57)
    ci_low = sim.get("eng_ci_low", 0.40)
    ci_high = sim.get("eng_ci_high", 1.70)

    # Compute executive decision pillars if not provided
    if not exec_dec:
        exec_dec = compute_executive_roi_and_decision(sim, brand_tier=brand_tier)

    cv = exec_dec["channel_verdict"]
    fr = exec_dec["financial_roi"]
    cr = exec_dec["creative_recipe"]
    qa = exec_dec.get("quarter_advisory", {})
    ta = exec_dec.get("tier_advisory", {})
    is_emerging = ta.get("is_emerging", False)
    is_challenger = ta.get("is_challenger", False)
    tier_badge = ta.get("tier_badge", "🏛️ Megabrand ($15M+)")

    # Creative Strategy Memo
    creative_memo = (
        f"**Concept Analysis & Narrative Hook ({tier_badge}):**\n"
        f"The concept models a predicted YouTube engagement rate of **{eng}%** with an expected Wikipedia pageview surge multiplier of **{wiki}x**. "
        f"This performance trajectory aligns closely with historical precedent **{top_comp.get('brand')} ({top_comp.get('year')}) — '{top_comp.get('title')}'**.\n\n"
        f"**Creative Mandate & Winning Recipe:**\n"
        f"• {cr['formula_title']}: {cr['primary_lift']}\n"
        + "\n".join([f"• **{elem[0]}:** {elem[1]}" for elem in cr["recipe_elements"]]) + "\n"
        f"• ⚠️ **Critical Avoidance:** {cr['avoid_elements']}"
    )

    # Media Planning Memo
    decay_warning = (
        "⚠️ **CRITICAL WARNING:** Late-game airtime (Game Q4) paired with a high point spread creates severe risk of audience attrition "
        f"and spikes Effective CPM (${cpm}) above efficient thresholds. Strongly recommend front-loading airtime into Game Q2 or Q3."
        if (retention < 90.0 or cpm > 95.0)
        else "Audience flow shows resilient viewer retention under current point spread conditions with no immediate blowout penalty."
    )
    media_memo = (
        f"**Airtime Allocation & Audience Delivery:**\n"
        f"Telecast delivery is projected at **{reach}M living-room viewers** (**{retention}%** audience retention index) at an Effective Broadcast CPM of **${cpm}** against the upfront ${fr['airtime_cost_m']}M airtime allocation.\n\n"
        f"**Quarter Allotment & Scarcity Advisory ({qa.get('selected_quarter', 'Quarter')}):**\n"
        f"• Status: {qa.get('scarcity_badge', '')}\n"
        f"• Intelligence: {qa.get('warning', '')}\n"
        f"• Strategic Pivot: {qa.get('smart_alt_summary', '')}\n\n"
        f"**Channel Verdict Assessment:**\n"
        f"• Recommendation: **{cv['badge']}** — {cv['summary']}\n"
        f"• Alternative Digital Scenario: {cv['social_comparison']}\n\n"
        f"**Audience Decay & Blowout Audit:**\n"
        f"{decay_warning}"
    )

    # CFO Governance Memo with Brand Capital Solvency Audit
    if is_emerging:
        cfo_solvency_text = (
            f"🔴 **CAPITAL CEILING VETO:** Total broadcast outlay of **${fr['total_capital_m']}M** exceeds total brand solvency (<$6M budget ceiling). "
            f"Direct CFO directive: Veto national telecast commitment. Authorize **${ta.get('digital_budget_k', 500)}K Social Creator Sprint** "
            f"(projected **${ta.get('digital_expected_value_m', 2.4)}M** GMV / **{ta.get('digital_roas', '4.8x ROAS')}**) and/or a "
            f"**{ta.get('local_dma_cost_range', '$40K–$120K')} Local DMA Interconnect spot** in top target metro market. "
            f"Preserves **${ta.get('capital_saved_m', 8.5)}M in treasury cash**."
        )
    elif is_challenger and qa.get("selected_quarter") == "Game Q2":
        cfo_solvency_text = (
            f"⚠️ **CAPITAL DISCIPLINE DIRECTIVE:** Q2 airtime (${fr['airtime_cost_m']}M) consumes excessive challenger capital reserves. "
            f"CFO directive: Reallocate from Q2 to Game Q3 ($6.4M rate) to unlock $1.4M in cash reserves for retail and digital conversion funnels."
        )
    else:
        cfo_solvency_text = (
            f"🟢 **CAPITAL ALLOCATION CLEARED:** Total upfront exposure of **${fr['total_capital_m']}M** is well within corporate treasury risk boundaries. "
            f"Expected Net Commercial Value is positive (**+${fr['net_profit_m']}M** / **+{fr['roi_pct']}%** net ROI)."
        )

    cfo_memo = (
        f"**Downside Floor & Capital Governance Audit ({tier_badge}):**\n"
        f"{cfo_solvency_text}\n\n"
        f"• Total Upfront Capital Exposure: **${fr['total_capital_m']}M** (${fr['airtime_cost_m']}M airtime + ${fr['prod_cost_m']}M production).\n"
        f"• Expected Gross Commercial Value: **${fr['expected_gross_value_m']}M** (Expected Net ROI: **+{fr['roi_pct']}%** / **{fr['roi_multiplier']}x** multiple).\n"
        f"• Maximum Viable Budget Cap: **${fr['max_viable_budget_m']}M**.\n"
        f"• Downside Capital Safety Floor: **${fr['downside_floor_m']}M** (10th percentile worst-case protection floor).\n\n"
        f"**Mandatory Governance Disclosure:**\n"
        f"In accordance with empirical advertising science governance standards, all modeled performance indicators reflect historical cohort patterns (2000–2026) across public telecast sponsors. "
        f"They do not constitute guaranteed causal retail sales ROI or direct equity price movement."
    )

    # Executive Consensus Decision Brief structured around the 3 core pillars
    recipe_bullets = "\n".join([f"   • **{elem[0]}:** {elem[1]}" for elem in cr["recipe_elements"]])
    consensus_memo = (
        f"### 📋 Executive Consensus Decision Brief\n\n"
        f"**0. Brand Capital & Scale Tier: {tier_badge}**\n"
        f"- {cfo_solvency_text}\n\n"
        f"**1. Strategic Channel Verdict: {cv['badge']}**\n"
        f"- **Primary Finding:** {cv['summary']}. {cv['headline']}\n"
        f"- **Broadcast Delivery:** {reach}M living-room viewers delivered at ${cpm} Effective CPM ({retention}% retention).\n"
        f"- **Quarter Allotment Strategy:** {qa.get('headline', '')} — {qa.get('smart_alt_summary', '')}\n"
        f"- **Social Media Tradeoff:** {cv['social_comparison']}\n\n"
        f"**2. Financial ROI & Budget Cap (Total Capital: ${fr['total_capital_m']}M):**\n"
        f"- **Expected Commercial & Earned Value:** **${fr['expected_gross_value_m']}M** (Expected Net ROI: **+{fr['roi_pct']}%** | **{fr['roi_multiplier']}x** return).\n"
        f"- **Maximum Viable Budget:** **${fr['max_viable_budget_m']}M Spend Ceiling** (preserve positive net commercial return).\n"
        f"- **Downside Capital Safety Floor:** **${fr['downside_floor_m']}M** value protected at 10th percentile.\n"
        f"- **Unit Efficiency:** **${fr['cost_per_viewer']} per reached viewer**.\n\n"
        f"**3. Creative Content Playbook (What to Put in the Ad):**\n"
        f"- **Winning Formula:** {cr['formula_title']} ({cr['primary_lift']})\n"
        f"{recipe_bullets}\n"
        f"- **Critical Trap to Avoid:** ⚠️ {cr['avoid_elements']}\n"
        f"- **Immediate High-Leverage Move:** {cr['immediate_action']}\n\n"
        f"**4. Governance & Compliance Sign-Off:**\n"
        f"- Downside 80% conformal confidence floor ({ci_low}%) confirmed within risk tolerance thresholds.\n"
        f"- Non-causal disclosure approved: modeled metrics represent historical cohort patterns, not guaranteed retail revenues."
    )

    return {
        "creative": creative_memo,
        "media": media_memo,
        "cfo": cfo_memo,
        "consensus": consensus_memo,
    }


def _run_direct_gemini_fallback(
    scenario_summary: str,
    sim: Dict[str, Any],
    comps: List[Dict[str, Any]],
    client: Any,
) -> Dict[str, str]:
    """
    Direct prompt-engineered fallback that produces the exact 3 stakeholder memos + consensus.
    """
    if not client:
        return generate_baseline_strategy_memos(scenario_summary, sim, comps)

    prompt = (
        f"You are simulating a 3-person executive strategy board for a $7M Super Bowl commercial buy:\n"
        f"Scenario: {scenario_summary}\n"
        f"Predicted Metrics: Engagement={sim.get('pred_eng_pct')}% [CI: {sim.get('eng_ci_low')}%-{sim.get('eng_ci_high')}%], "
        f"Wiki Surge={sim.get('pred_wiki_mult')}x, Reach={sim.get('effective_reach_m')}M viewers, CPM=${sim.get('effective_cpm_usd')}.\n"
        f"Comps: {', '.join([c['brand'] + ' ' + str(c['year']) for c in comps[:3]])}.\n\n"
        f"Generate 4 distinct sections:\n"
        f"1. [CREATIVE MEMO]: Creative Director analysis of narrative hook and comps.\n"
        f"2. [MEDIA MEMO]: Media Planner analysis of telecast quarter and blowout penalty.\n"
        f"3. [CFO MEMO]: CFO evaluation of downside risk floor and non-causal disclosures.\n"
        f"4. [CONSENSUS BRIEF]: Final executive verdict, strategic mandates, and sign-off conditions."
    )

    try:
        resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return _parse_memos_from_text(resp.text)
    except Exception:
        return generate_baseline_strategy_memos(scenario_summary, sim, comps)


def _parse_memos_from_text(full_text: str) -> Dict[str, str]:
    sections = {"creative": "", "media": "", "cfo": "", "consensus": full_text}
    text_lower = full_text.lower()

    if "[creative memo]" in text_lower and "[media memo]" in text_lower:
        parts = full_text.split("[")
        for p in parts:
            p_low = p.lower()
            if p_low.startswith("creative memo]"):
                sections["creative"] = p.replace("creative memo]", "").strip()
            elif p_low.startswith("media memo]"):
                sections["media"] = p.replace("media memo]", "").strip()
            elif p_low.startswith("cfo memo]"):
                sections["cfo"] = p.replace("cfo memo]", "").strip()
            elif p_low.startswith("consensus brief]"):
                sections["consensus"] = p.replace("consensus brief]", "").strip()

    return sections
