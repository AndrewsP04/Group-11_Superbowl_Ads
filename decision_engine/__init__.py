"""
decision_engine - Super Bowl Ads AI & Multi-Agent Decision Engine Package.

Modular components:
- config_auth: Vertex AI & Gemini authentication setup
- enrich_and_train: Two-Stage RidgeCV + Monotonic EBM offline training pipeline
- decision_tools: Sub-20ms local deterministic inference & Vector RAG comps
- adk_boardroom: Google ADK Multi-Agent Boardroom Engine on Vertex AI
- tab6_decision_engine: Streamlit Tab 6 Strategy Studio view
"""

from .config_auth import get_genai_client, init_vertex_env
from .decision_tools import (
    compute_effective_reach,
    optimize_creative_and_airtime,
    parse_concept_pitch_to_controls,
    retrieve_nearest_historical_ads,
    simulate_multi_pov,
)
from .adk_boardroom import run_strategy_board
from .tab6_decision_engine import render_tab6

__all__ = [
    "init_vertex_env",
    "get_genai_client",
    "simulate_multi_pov",
    "compute_effective_reach",
    "retrieve_nearest_historical_ads",
    "optimize_creative_and_airtime",
    "parse_concept_pitch_to_controls",
    "run_strategy_board",
    "render_tab6",
]
