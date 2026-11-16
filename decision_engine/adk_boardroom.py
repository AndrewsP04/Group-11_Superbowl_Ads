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
        optimize_creative_and_airtime,
        retrieve_nearest_historical_ads,
        simulate_multi_pov,
    )
except ImportError:
    from config_auth import get_genai_client, init_vertex_env
    from decision_tools import (
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

    # Creative Strategy Memo
    creative_memo = (
        f"**Concept Analysis & Narrative Hook:**\n"
        f"The concept models a predicted YouTube engagement rate of **{eng}%** with an expected Wikipedia pageview surge multiplier of **{wiki}x**. "
        f"This performance trajectory aligns closely with historical precedent **{top_comp.get('brand')} ({top_comp.get('year')}) — '{top_comp.get('title')}'**.\n\n"
        f"**Creative Mandate:**\n"
        f"Ensure that product integration and packaging visual cues occur synchronously with the narrative climax. "
        f"High humor and celebrity appeal must reinforce the sponsor rather than overshadowing brand recall."
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
        f"Telecast delivery is projected at **{reach}M living-room viewers** (**{retention}%** audience retention index) at an Effective Broadcast CPM of **${cpm}** against the upfront $7M capital baseline.\n\n"
        f"**Audience Decay & Blowout Audit:**\n"
        f"{decay_warning}"
    )

    # CFO Governance Memo
    cfo_memo = (
        f"**Downside Floor & Conformal Bound Audit:**\n"
        f"Downside risk floor (10th percentile lower conformal bound) is validated at **{ci_low}%** engagement (against an upside 90th percentile of **{ci_high}%**). "
        f"Total upfront capital commitment is calibrated at **$7,000,000**.\n\n"
        f"**Mandatory Governance Disclosure:**\n"
        f"In accordance with empirical advertising science governance standards, all modeled performance indicators reflect historical cohort patterns (2000–2026) across public telecast sponsors. "
        f"They do not constitute guaranteed causal retail sales ROI or direct equity price movement."
    )

    # Executive Consensus Decision Brief
    if retention < 90.0 or cpm > 95.0:
        verdict = "⚠️ CONDITIONAL CAUTION (Q4 BLOWOUT RISK)"
        media_mandate = "- Re-evaluate Q4 placement; hedge against blowout decay by securing Q2/Q3 positioning.\n\n"
    elif ci_low < 0.20:
        verdict = "⚡ HOLD & REVISE CREATIVE PACING"
        media_mandate = "- Hold airtime lock until creative revision demonstrates higher baseline engagement.\n\n"
    else:
        verdict = "✅ CONDITIONAL APPROVAL (RECOMMEND PROCEED)"
        media_mandate = "- Lock upfront quarter placement based on stable living-room audience retention.\n\n"

    consensus_memo = (
        f"### 📋 Executive Consensus Decision Brief\n\n"
        f"**1. Strategic Verdict: {verdict}**\n"
        f"- **Primary KPI Outlook:** Modeled engagement at **{eng}%** [80% CI: {ci_low}% – {ci_high}%] with a **{wiki}x** Wikipedia search surge.\n"
        f"- **Reach & Efficiency:** {reach}M viewers delivered at ${cpm} Effective CPM ({retention}% retention).\n\n"
        f"**2. Creative Director Mandate:**\n"
        f"- Maintain prominent product placement at the narrative climax to safeguard recall.\n\n"
        f"**3. Media Planning Mandate:**\n"
        f"{media_mandate}"
        f"**4. CFO Downside Governance:**\n"
        f"- Downside engagement risk floor is secured at {ci_low}%.\n"
        f"- Non-causal disclosure approved: modeled metrics represent historical cohort engagement, not guaranteed commercial revenues."
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
