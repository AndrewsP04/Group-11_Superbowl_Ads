"""
enrich_and_train.py - Offline Training Pipeline for Super Bowl Ads Decision Engine.

1. Loads superbowl_ads_combined_master.csv and wiki_spikes.csv (with brand normalization).
2. Computes empirical log targets:
   - y_eng_log: log1p(YouTube engagement rate %)
   - y_wiki_log: log(Wikipedia pageview surge multiplier)
3. Enriches records with Gemini continuous semantic metrics (0-10) and gemini-embedding-001
   vectors, cached in artifacts/gemini_semantic_cache.json (with deterministic offline fallback).
4. Trains two-stage heads:
   - Stage 1: RidgeCV macro detrending (Year, Runtime)
   - Stage 2: ExplainableBoostingRegressor (EBM) on creative residuals
5. Evaluates 5-fold GroupKFold by year, extracts 80% conformal prediction bounds (10th/90th percentiles).
6. Serializes artifacts/decision_engine_bundle.joblib.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

# Optional Microsoft EBM import with fallback to Ridge
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    HAS_EBM = True
except ImportError:
    HAS_EBM = False

# Import Vertex-routed Gemini auth helper
try:
    from .config_auth import get_genai_client, init_vertex_env
except ImportError:
    from config_auth import get_genai_client, init_vertex_env

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CACHE_FILE = ARTIFACTS_DIR / "gemini_semantic_cache.json"
BUNDLE_FILE = ARTIFACTS_DIR / "decision_engine_bundle.joblib"

TRAITS = [
    "funny",
    "show_product_quickly",
    "patriotic",
    "celebrity",
    "danger",
    "animals",
    "use_sex",
]


def find_data_file(filename: str) -> Path:
    candidates = [
        BASE_DIR / "data/processed" / filename,
        BASE_DIR / "data/raw" / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / filename,
        Path("data/processed") / filename,
        Path(filename),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not locate required dataset: {filename}")


def normalize_brand_name(brand_series: pd.Series) -> pd.Series:
    cleaned = brand_series.astype(str).str.strip()
    mapping = {
        "hynudai": "Hyundai",
        "hyundai": "Hyundai",
        "coca cola": "Coca-Cola",
        "coca-cola": "Coca-Cola",
        "mcdonalds": "McDonald's",
        "mcdonald's": "McDonald's",
        "e*trade": "E-Trade",
        "etrade": "E-Trade",
    }
    return cleaned.apply(lambda b: mapping.get(b.lower(), b))


def classify_vertical(brand: str) -> str:
    b = str(brand).lower()
    if any(k in b for k in ["bud", "pepsi", "coca", "beer", "dew", "drink"]):
        return "Beverage"
    if any(k in b for k in ["toyota", "hyundai", "kia", "audi", "chevy", "nissan", "ford", "bmw"]):
        return "Automotive"
    if any(k in b for k in ["tech", "google", "amazon", "etrade", "e-trade", "squarespace"]):
        return "ConsumerTech"
    return "SnacksFood"


def extract_offline_heuristic_semantics(row: pd.Series) -> Dict[str, float]:
    title = str(row.get("yt_title", "")).lower()
    funny = bool(row.get("funny", False))
    celebrity = bool(row.get("celebrity", False))
    quick_prod = bool(row.get("show_product_quickly", False))
    animals = bool(row.get("animals", False))
    patriotic = bool(row.get("patriotic", False))

    story_arc = 5.0
    if funny and animals:
        story_arc += 2.2
    if quick_prod:
        story_arc -= 0.8
    if "story" in title or "movie" in title or "tale" in title:
        story_arc += 1.5

    brand_cent = 5.5
    if quick_prod:
        brand_cent += 2.4
    if celebrity and not quick_prod:
        brand_cent -= 1.0

    cultural = 4.0
    if celebrity:
        cultural += 2.5
    if patriotic:
        cultural += 2.0
    if any(k in title for k in ["nostalgia", "classic", "80s", "90s", "legend", "anthem"]):
        cultural += 2.0

    return {
        "narrative_arc_score": float(np.clip(story_arc, 1.0, 10.0)),
        "brand_centrality_score": float(np.clip(brand_cent, 1.0, 10.0)),
        "cultural_nostalgia_score": float(np.clip(cultural, 1.0, 10.0)),
    }


def enrich_with_gemini(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, Any] = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    client = get_genai_client()
    narratives, brand_cents, nostalgias = [], [], []
    embeddings_list = []

    print(f"Enriching {len(df)} commercial spots with semantic scores and embeddings...")
    for idx, row in df.iterrows():
        key = f"{row['year']}_{str(row['brand']).strip()}_{str(row.get('video_id', idx))}"
        cached_entry = cache.get(key)

        if (
            cached_entry
            and isinstance(cached_entry.get("scores"), dict)
            and len(cached_entry.get("embedding", [])) == 3072
        ):
            scores = cached_entry["scores"]
            emb = cached_entry.get("embedding")
        else:
            scores = None
            emb = None

            if client:
                prompt_text = (
                    f"Super Bowl Ad: {row['year']} {row['brand']}. "
                    f"Title: {row.get('yt_title', '')}. "
                    f"Traits: funny={row.get('funny')}, celebrity={row.get('celebrity')}, "
                    f"quick_product={row.get('show_product_quickly')}, animals={row.get('animals')}."
                )
                try:
                    score_resp = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=(
                            f"Analyze this Super Bowl commercial and rate three scores from 1.0 to 10.0 in JSON object format: "
                            f"'narrative_arc_score' (1=pure pitch, 10=rich emotional story), "
                            f"'brand_centrality_score' (1=forgotten brand, 10=product central to plot), "
                            f"'cultural_nostalgia_score' (1=no throwback/IP, 10=heavy cultural/nostalgic hook).\n\n"
                            f"{prompt_text}"
                        ),
                        config={"response_mime_type": "application/json"},
                    )
                    parsed_json = json.loads(score_resp.text)
                    if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
                        scores = parsed_json[0]
                    elif isinstance(parsed_json, dict):
                        scores = parsed_json
                    else:
                        scores = None
                except Exception:
                    scores = None

                try:
                    emb_resp = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=prompt_text,
                    )
                    if hasattr(emb_resp, "embeddings") and emb_resp.embeddings:
                        emb = emb_resp.embeddings[0].values
                    elif hasattr(emb_resp, "embedding"):
                        emb = emb_resp.embedding.values
                    else:
                        emb = None
                except Exception:
                    emb = None

            if not isinstance(scores, dict):
                scores = extract_offline_heuristic_semantics(row)

            if emb is None or not isinstance(emb, (list, np.ndarray)) or len(emb) != 3072:
                trait_bits = [float(bool(row.get(t, False))) for t in TRAITS]
                year_norm = (float(row["year"]) - 2000.0) / 26.0
                base_vec = np.array(trait_bits + [year_norm, float(scores.get("narrative_arc_score", 5.0)) / 10.0])
                reps = int(np.ceil(3072 / len(base_vec)))
                emb = np.tile(base_vec, reps)[:3072].tolist()

            cache[key] = {"scores": scores, "embedding": emb}

        narratives.append(float(scores.get("narrative_arc_score", 5.0)))
        brand_cents.append(float(scores.get("brand_centrality_score", 5.0)))
        nostalgias.append(float(scores.get("cultural_nostalgia_score", 5.0)))
        embeddings_list.append(emb)

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    df["narrative_arc_score"] = narratives
    df["brand_centrality_score"] = brand_cents
    df["cultural_nostalgia_score"] = nostalgias

    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_matrix = embeddings_matrix / norms

    return df, embeddings_matrix


def run_pipeline():
    init_vertex_env()
    print("Loading raw Super Bowl datasets...")
    master_path = find_data_file("superbowl_ads_combined_master.csv")
    wiki_path = find_data_file("wiki_spikes.csv")

    df_master = pd.read_csv(master_path)
    df_wiki = pd.read_csv(wiki_path)

    df_master["brand"] = normalize_brand_name(df_master["brand"])
    df_wiki["brand"] = normalize_brand_name(df_wiki["brand"])
    df_master["year"] = pd.to_numeric(df_master["year"], errors="coerce").fillna(2010).astype(int)
    df_wiki["year"] = pd.to_numeric(df_wiki["year"], errors="coerce").fillna(2010).astype(int)

    def clean_slug_to_title(url: Any, brand: str, year: int) -> str:
        if not isinstance(url, str) or not url.strip():
            return f"{brand} Super Bowl Commercial ({year})"
        slug = url.strip().rstrip("/").split("/")[-1]
        slug = re.sub(r"^\d{4}-", "", slug)
        slug = re.sub(rf"^{re.escape(str(brand).lower())}-", "", slug, flags=re.IGNORECASE)
        slug = re.sub(r"super-bowl(-\w+)?-commercial-?", "", slug, flags=re.IGNORECASE)
        slug = re.sub(r"super-bowl(-\w+)?-ad-?", "", slug, flags=re.IGNORECASE)
        slug = re.sub(r"super-bowl-?", "", slug, flags=re.IGNORECASE)
        words = [w.capitalize() for w in slug.split("-") if w]
        clean = " ".join(words).strip()
        return clean if clean else f"{brand} Super Bowl Commercial ({year})"

    clean_titles = []
    for _, r in df_master.iterrows():
        val = r.get("yt_title")
        if pd.isna(val) or str(val).lower() in ["nan", "none", "null", ""]:
            clean_titles.append(clean_slug_to_title(r.get("superbowl_ads_dot_com_url"), r.get("brand"), r.get("year")))
        else:
            s = str(val).strip()
            s = re.sub(r"\.(mp4|mov|avi|wmv|mkv)$", "", s, flags=re.IGNORECASE).strip()
            s = re.sub(r"\[(HD|HQ|720p|1080p|Official)\]", "", s, flags=re.IGNORECASE).strip()
            clean_titles.append(s)
    df_master["yt_title"] = clean_titles

    clean_urls = []
    for _, r in df_master.iterrows():
        u = r.get("youtube_url")
        if pd.isna(u) or str(u).lower() in ["nan", "none", "null", ""] or not str(u).startswith("http"):
            sb_u = r.get("superbowl_ads_dot_com_url")
            if not pd.isna(sb_u) and isinstance(sb_u, str) and sb_u.startswith("http"):
                clean_urls.append(sb_u.strip())
            else:
                q = f"{r['brand']} {r['year']} Super Bowl Commercial {r['yt_title']}".replace(" ", "+")
                clean_urls.append(f"https://www.youtube.com/results?search_query={q}")
        else:
            clean_urls.append(str(u).strip())
    df_master["youtube_url"] = clean_urls

    df = pd.merge(df_master, df_wiki[["brand", "year", "spike_ratio"]], on=["brand", "year"], how="left")
    df["spike_ratio"] = pd.to_numeric(df["spike_ratio"], errors="coerce")

    for t in TRAITS:
        df[t] = df[t].map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}).fillna(0.0)

    df["vertical"] = df["brand"].apply(classify_vertical)
    df["spot_runtime"] = pd.to_numeric(df.get("yt_duration_seconds", 30), errors="coerce").fillna(30.0)
    df["is_60s"] = (df["spot_runtime"] >= 50.0).astype(float)

    views = pd.to_numeric(df.get("yt_view_count", 1000), errors="coerce").fillna(1000.0).clip(lower=100.0)
    likes = pd.to_numeric(df.get("yt_like_count", 10), errors="coerce").fillna(10.0)
    comments = pd.to_numeric(df.get("yt_comment_count", 2), errors="coerce").fillna(2.0)

    df["eng_rate_pct"] = ((likes + comments) / views) * 100.0
    df["y_eng_log"] = np.log1p(df["eng_rate_pct"].clip(lower=0.0, upper=100.0))

    # Wikipedia target strictly on non-null empirical observations (avoids flattening variance with median)
    wiki_mask = df["spike_ratio"].notna() & (df["spike_ratio"] > 0)
    df["y_wiki_log"] = np.nan
    df.loc[wiki_mask, "y_wiki_log"] = np.log(df.loc[wiki_mask, "spike_ratio"].clip(lower=1.0, upper=100.0))

    df, embeddings_matrix = enrich_with_gemini(df)

    macro_feats = ["year", "spot_runtime"]
    creative_feats = TRAITS + [
        "narrative_arc_score",
        "brand_centrality_score",
        "cultural_nostalgia_score",
        "is_60s",
    ]

    for v in ["Beverage", "Automotive", "ConsumerTech", "SnacksFood"]:
        col = f"v_{v}"
        df[col] = (df["vertical"] == v).astype(float)
        creative_feats.append(col)

    X_macro = df[macro_feats].values
    X_creative = df[creative_feats].values
    y_eng = df["y_eng_log"].values
    years = df["year"].values

    print(f"Dataset prepared: {len(df)} spots across {len(np.unique(years))} years.")
    print("Fitting Two-Stage Models (Stage 1: RidgeCV, Stage 2: EBM)...")

    # YouTube Engagement Model
    ridge_eng = RidgeCV(alphas=np.logspace(-2, 3, 30))
    ridge_eng.fit(X_macro, y_eng)
    res_eng = y_eng - ridge_eng.predict(X_macro)

    # Wikipedia Surge Model (strictly trained on empirical non-null observations)
    X_macro_wiki = X_macro[wiki_mask.values]
    X_creative_wiki = X_creative[wiki_mask.values]
    y_wiki_train = df.loc[wiki_mask, "y_wiki_log"].values

    ridge_wiki = RidgeCV(alphas=np.logspace(-2, 3, 30))
    ridge_wiki.fit(X_macro_wiki, y_wiki_train)
    res_wiki = y_wiki_train - ridge_wiki.predict(X_macro_wiki)

    if HAS_EBM:
        print("Using Microsoft ExplainableBoostingRegressor (EBM)...")
        ebm_eng = ExplainableBoostingRegressor(
            interactions=4,
            learning_rate=0.015,
            max_bins=32,
            random_state=42,
            feature_names=creative_feats,
        )
        ebm_eng.fit(X_creative, res_eng)

        ebm_wiki = ExplainableBoostingRegressor(
            interactions=2,
            learning_rate=0.015,
            max_bins=32,
            random_state=42,
            feature_names=creative_feats,
        )
        ebm_wiki.fit(X_creative_wiki, res_wiki)
    else:
        print("InterpretML not installed, falling back to Ridge for Stage 2...")
        ebm_eng = Ridge(alpha=1.0)
        ebm_eng.fit(X_creative, res_eng)

        ebm_wiki = Ridge(alpha=1.0)
        ebm_wiki.fit(X_creative_wiki, res_wiki)

    gkf = GroupKFold(n_splits=min(5, len(np.unique(years))))
    oof_eng_preds = np.zeros_like(y_eng)
    for tr, te in gkf.split(X_creative, y_eng, groups=years):
        m1 = RidgeCV(alphas=np.logspace(-2, 3, 10)).fit(X_macro[tr], y_eng[tr])
        r_tr = y_eng[tr] - m1.predict(X_macro[tr])
        m2 = (
            ExplainableBoostingRegressor(
                interactions=2, learning_rate=0.02, max_bins=32, random_state=42, feature_names=creative_feats
            )
            if HAS_EBM
            else Ridge(alpha=1.0)
        )
        m2.fit(X_creative[tr], r_tr)
        oof_eng_preds[te] = m1.predict(X_macro[te]) + m2.predict(X_creative[te])

    errors_eng = y_eng - oof_eng_preds
    q10_eng = float(np.percentile(errors_eng, 10))
    q90_eng = float(np.percentile(errors_eng, 90))
    rmse_eng = float(np.sqrt(np.mean(errors_eng ** 2)))

    print(f"Validation GroupKFold OOF RMSE: {rmse_eng:.3f}")
    print(f"Conformal 80% Residual Error Quantiles: [10th: {q10_eng:+.3f}, 90th: {q90_eng:+.3f}]")

    bundle = {
        "ridge_eng": ridge_eng,
        "ebm_eng": ebm_eng,
        "ridge_wiki": ridge_wiki,
        "ebm_wiki": ebm_wiki,
        "macro_feats": macro_feats,
        "creative_feats": creative_feats,
        "q10_eng": q10_eng,
        "q90_eng": q90_eng,
        "rmse_eng": rmse_eng,
        "historical_records": df[[
            "year", "brand", "yt_title", "youtube_url", "superbowl_ads_dot_com_url", "eng_rate_pct", "spike_ratio", "spot_runtime"
        ] + TRAITS].to_dict(orient="records"),
        "embeddings_matrix": embeddings_matrix,
        "has_ebm": HAS_EBM,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_FILE)
    print(f"Decision Engine bundle successfully saved to {BUNDLE_FILE} ({BUNDLE_FILE.stat().st_size / 1024:.1f} KB).")


if __name__ == "__main__":
    run_pipeline()
