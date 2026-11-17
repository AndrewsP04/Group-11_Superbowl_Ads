"""
config_auth.py - Centralized Authentication & Client Setup for Vertex AI & Gemini.

Standardizes on GEMINI_API_KEY from .env, routes requests through Vertex AI
(GOOGLE_GENAI_USE_VERTEXAI = "TRUE"), and bridges credentials for google-adk
and google-genai.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    genai = None  # type: ignore
    HAS_GENAI = False

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def init_vertex_env() -> Optional[str]:
    """
    Initializes environment variables for Vertex AI and Google ADK.
    Loads GEMINI_API_KEY and enforces GOOGLE_GENAI_USE_VERTEXAI = 'TRUE'.
    Bridges GEMINI_API_KEY to GOOGLE_API_KEY for ADK internal operations.
    Also bridges Streamlit Cloud Secrets (st.secrets) to os.environ when deployed.
    """
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    else:
        load_dotenv()

    # Bridge Streamlit Cloud Secrets (st.secrets) to os.environ if running on Streamlit Cloud
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "YOUTUBE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"]:
                val = None
                try:
                    val = st.secrets.get(k) or st.secrets.get(k.lower())
                except Exception:
                    if k in st.secrets:
                        val = st.secrets[k]
                    elif k.lower() in st.secrets:
                        val = st.secrets[k.lower()]
                if val and not os.getenv(k):
                    os.environ[k] = str(val).strip()
    except Exception:
        pass

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
    return api_key


def get_genai_client():
    """
    Returns an authenticated genai.Client instance configured with vertexai=True,
    or None if google-genai is not installed or no API key is present.
    """
    api_key = init_vertex_env()
    if HAS_GENAI and api_key:
        try:
            return genai.Client(vertexai=True, api_key=api_key)
        except Exception:
            try:
                return genai.Client(api_key=api_key)
            except Exception:
                return None
    return None
