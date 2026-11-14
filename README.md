# Super Bowl Ads Decision Engine

An enterprise-grade interactive Streamlit decision engine and analytics suite for comparing Super Bowl advertising creative traits, telecast audience flow, YouTube engagement, Wikipedia pageview surges, and equity market impact. 

Powered by a **Two-Stage RidgeCV + Monotonic Explainable Boosting Machine (EBM)** predictive architecture, **3072-dimensional Vector RAG historical retrieval**, and an on-demand **Google ADK Multi-Agent Strategy Boardroom** on Vertex AI.

---

## Key Highlights & Innovations

- **Rejection of Tree Ensembles (LightGBM/XGBoost):** Small sample sizes ($N=244\text{--}567$ spots over 25 years) cause standard tree models to overfit on viral outliers, flatline outside historical ranges, and create jagged step-function cliffs on interactive sliders. Our **Two-Stage RidgeCV + Monotonic EBM** (`interpret`) decouples macro temporal trends (Stage 1) from additive creative residuals and pairwise non-linear synergies (Stage 2).
- **Active 2026 Google Cloud Models:**
  - **`gemini-3.5-flash-lite`:** Sub-second natural language concept brief parsing and offline semantic narrative extraction.
  - **`gemini-embedding-001`:** Full 3072-dimensional vector RAG embeddings matching pitches against 25 years of historical commercials.
  - **`gemini-3.8-flash`:** Multi-step reasoning and adversarial boardroom debates orchestrated via Google Agent Development Kit (ADK).
- **Sub-20ms Deterministic Local Simulation:** Instant slider responsiveness (<20ms forward passes) without lag or redundant remote LLM calls.
- **True 512-Grid Vectorized Prescriptive Optimizer:** Evaluates all $2^7 \times 4 = 512$ creative trait and broadcast quarter permutations in a single NumPy batch pass in under **`2 ms`**.
- **CFO Governance & Conformal Prediction Bounds:** Enforces 80% conformal prediction intervals ($10^{\text{th}}$ percentile downside floors) and strict non-causal disclosures (modeled cohort interest vs. guaranteed retail revenue).

---

## Project Structure

```text
Group-11_Superbowl_Ads/
├── app.py                                          # Main Streamlit dashboard application (Tabs 1–6)
├── requirements.txt                                # Python dependencies (interpret, google-adk, plotly, etc.)
├── README.md                                       # Comprehensive project documentation
├── .env                                            # Local environment configuration (API keys & Vertex routing)
├── .gitignore                                      # Excludes .env, virtual environments, and caches
│
├── decision_engine/                                # Core AI & Multi-Agent Decision Engine Package
│   ├── __init__.py                                 # Package initializer & public exports
│   ├── tab6_decision_engine.py                     # Tab 6: Agentic Strategy Studio & Multi-POV UI
│   ├── decision_tools.py                           # Sub-20ms local deterministic inference & 3072-d Vector RAG
│   ├── adk_boardroom.py                            # Google ADK Multi-Agent Boardroom Engine (gemini-3.8-flash)
│   ├── enrich_and_train.py                         # Two-Stage RidgeCV + EBM trainer & semantic enricher
│   └── config_auth.py                              # Centralized Vertex AI & Gemini SDK authentication
│
├── artifacts/                                      # Serialized models & semantic embedding caches
│   ├── decision_engine_bundle.joblib               # Fitted RidgeCV + EBM models, conformal bounds & RAG matrix
│   └── gemini_semantic_cache.json                  # Cached 3072-d embeddings & continuous semantic scores
│
├── data/
│   ├── raw/                                        # Source & scraped initial inputs
│   │   ├── superbowl-ads.csv                       # Baseline ad creative attributes
│   │   ├── superbowl_telecast_macro_ratings_2000_2026.csv
│   │   └── instagram_posts.csv
│   │
│   ├── processed/                                  # Cleaned & joined datasets powering app.py
│   │   ├── superbowl_ads_combined_master.csv       # Unified master ad traits & social metrics
│   │   ├── superbowl_ads_with_linear_ratings_flow.csv
│   │   ├── superbowl_ads_with_car.csv
│   │   ├── superbowl_quarter_audience_flow_2000_2026.csv
│   │   ├── superbowl_ads_enriched.csv
│   │   ├── superbowl-ads-admeter-sales-impact-expanded.csv
│   │   ├── google_trends_spikes.csv
│   │   ├── google_trends_quarterly_decay.csv
│   │   ├── wiki_spikes.csv
│   │   └── wiki_daily.csv
│   │
│   └── market/                                     # Historical equity and advertiser daily returns
│       ├── superbowl_advertisers_daily_returns_2000_2026.parquet
│       ├── superbowl_advertisers_daily_returns_2000_2026.csv
│       ├── equity_daily_returns_2000_2026.parquet
│       └── equity_daily_returns_2000_2026.csv
│
├── scripts/                                        # Data extraction, modeling & calculation engines
│   ├── event_study.py                              # CAR[-1, +5] cumulative abnormal return model
│   ├── fetch_google_trends.py                      # Google Trends spike & longitudinal decay extractor
│   └── fetch_wikipedia.py                          # Wikimedia pageviews API extractor
│
├── notebooks/                                      # Exploratory analysis & scrapers
│   ├── data.ipynb                                  # YouTube API enrichment & visual analysis
│   ├── datascrape.ipynb                            # Apify Instagram scraping workflow
│   ├── sports_media_watch_scrape.ipynb             # SportsMediaWatch ratings & audience flow
│   └── yfinance.ipynb                              # Yahoo Finance market return puller
│
├── docs/                                           # Research documentation
│   └── GOOGLE_TRENDS_SUPERBOWL_RESEARCH.md         # 4-quarter search decay findings
│
└── assets/                                         # Exported figures & dashboard previews
    ├── superbowl_dashboard_combined.png
    ├── superbowl_performance_dashboard.png
    ├── superbowl_preliminary_visuals.png
    └── superbowl_real_dashboard.png
```

---

## Environment Configuration (`.env`)

Create or update a `.env` file in the project root:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=TRUE
YOUTUBE_API_KEY=your_youtube_api_key_here  # Optional: for raw YouTube scraping workflows
```

* **`GEMINI_API_KEY`**: Authenticates Google GenAI API calls on Vertex AI.
* **`GOOGLE_GENAI_USE_VERTEXAI=TRUE`**: Directs SDK and Google ADK calls through Vertex AI Agent Platform endpoints.
* **Security Notice**: `.gitignore` explicitly excludes `.env` and `.env.*`. Ensure keys are rotated in Google Cloud Console (`itcs-5122-513114`) before public distribution.

---

## Installation and Quickstart

1. **Clone the repository and set up a virtual environment:**
   ```bash
   git clone <repo_url>
   cd Group-11_Superbowl_Ads
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   *(Note: The engine automatically checks for artifacts on launch. If the bundle is missing, it auto-trains and serializes all models on first run with zero manual intervention).*

4. **(Optional) Run offline model training independently:**
   ```bash
   python3 -m decision_engine.enrich_and_train
   ```

---

## Dashboard Navigation & Tab Architecture

### **Tabs 1–5: Descriptive & Diagnostic Analytics**
* **T1 · Trait Engagement:** Compares average YouTube engagement rate, Google Trends commercial search lift (with longitudinal Financial Q1–Q4 decay), and average Wikipedia pageview spikes for ads using each selected creative trait.
* **T2 · Trends Over Time:** Visualizes how the prevalence of selected creative traits evolves by year (2000–2026) across the full dataset.
* **T3 · Brand Positioning:** Ranks selected brands by median YouTube engagement rate and views, with hover metrics detailing search lift and multi-quarter retention.
* **T4 · Reach vs. Dialogue:** Models the power-law relationship between raw impressions and active viewer dialogue (comments).
* **Phase 2 · Telecast Flow & Financial Return:** Analyzes live telecast ratings decay by Game Quarter (Game Q1–Game Q4), Effective CPM, and 7-day cumulative abnormal stock returns (CAR vs. S&P 500).

---

### **Tab 6: Agentic Campaign Strategy Studio & Multi-POV Decision Engine**

The decision engine unifies machine learning inference with multi-stakeholder governance:

#### **1. Dual Input Modes**
* **Natural Language Storyboard / Concept Parser:** Paste a free-form creative pitch (e.g., *"A 30-second Doritos spot with a celebrity dance-off..."*). `gemini-3.5-flash-lite` extracts brand vertical, airtime duration, broadcast quarter, point spread, active traits, and continuous narrative scores, synchronizing the UI controls via `st.rerun()`.
* **Interactive Control Sliders:** Granular manual tuning of narrative depth, brand centrality, nostalgia hooks, runtime (:15s to :90s), and game blowout point spread.

#### **2. Three Stakeholder Decision Lenses**
* 🎨 **Creative Strategy Director POV:**
  * Focuses on YouTube engagement rate % and Wikipedia search surge multipliers.
  * **Plotly EBM Waterfall:** Renders exact additive feature contributions and non-linear pairwise interaction synergies (e.g., `Synergy: Celebrity Presence × Snacks/Food Vertical`, `Story Arc Depth`, `Humor Tone`).
* 📡 **Media Buyer / Planner POV:**
  * Living-room linear telecast delivery (calibrated from 2000–2026 Nielsen/SportsMediaWatch telecast ratings).
  * Monotonic blowout decay model: $11\times$ steeper audience drop-off in Game Q4 if game point spread exceeds 10 points.
  * **512-Grid Prescriptive Combinatorial Optimizer:** Vectorized sweep of all $2^7 \times 4 = 512$ trait-by-quarter permutations in $<2\text{ ms}$, ranking top blueprints by composite reach, search surge, and engagement.
* 💼 **Brand CMO / CFO Risk POV:**
  * Evaluates upfront capital commitment ($7M baseline for :30s).
  * Conformal prediction downside risk floors ($10^{\text{th}}$ percentile 80% CI).
  * Enforces strict non-causal governance disclaimers in accordance with ITCS 5122 standards.

#### **3. Historical Precedents (Vector RAG Comps)**
* Cosine similarity retrieval against 25 years of Super Bowl commercials using **3072-dimensional `gemini-embedding-001` vectors**.
* Directly surfaces the Top-3 nearest historical spots with YouTube engagement rates, Wikipedia search spikes, match percentages, and playable YouTube links.

#### **4. Google ADK Multi-Agent Strategy Boardroom**
* On-demand adversarial debate orchestrated via `gemini-3.8-flash`:
  * `CreativeDirectorAgent`: Champions cultural buzz, storytelling arc, and viral potential.
  * `MediaPlannerAgent`: Scrutinizes living-room reach, blowout tune-out risks, and Effective CPM.
  * `CFO_RiskAgent`: Audits downside risk floors and capital allocation governance.
  * `ConsensusChairAgent`: Synthesizes debates into a formal **Executive Consensus Decision Brief** with conditional verdicts, strategic mandates, and one-click Markdown export (`.md`).

---

## Example Concept Prompts for Tab 6 Testing

Copy and paste any of the following concept pitches into the **"Campaign Concept Pitch / Storyboard Narrative"** text area in **Tab 6** and click **🪄 Parse Brief into Simulator Controls** to test the engine:

### 1. High-Performing Super Bowl Best Practice ("Good Idea")
> A high-energy 30-second Doritos commercial in the Snacks & Food category. Two A-list comedic celebrities engage in a dramatic, humorous dance duel over the last bag of Doritos, with the loud crunch serving as the musical beat drop and punchline. Scheduled for Game Q2 during a tight 3-point game, ensuring high product visibility throughout.

### 2. Abstract Arthouse Pitfall ("Bad Idea")
> A dark, slow-paced 60-second philosophical drama for an unknown crypto startup in Consumer Tech. Actors wander through a foggy, dystopian wasteland whispering existential poetry about the future of society with chaotic explosions, with zero product or company name shown until a tiny 1-second obscure web link at second 59. Scheduled for Game Q4 during a 22-point blowout game.

### 3. Fourth-Quarter Blowout Volatility Test
> We are pitching a high-stakes 30-second Super Bowl spot for Doritos in the Snacks & Food category. The concept features an energetic, humorous dance-off between top pop celebrities and everyday fans, with the product integrated right into the punchline. We are leaning toward booking Game Q4 to capture final-quarter attention, even though the expected game point spread is around 10 points.

### 4. Automotive Cinematic Storytelling (:60s)
> A cinematic 60-second Toyota commercial telling an inspiring, patriotic story about an everyday hero and a rescue dog overcoming adversity on winter roads, scheduled for Game Q1 kickoff during a tight 3-point game.

### 5. Beverage Slapstick & 90s Nostalgia (:30s)
> A humorous 30-second Bud Light-style beer commercial featuring talking farm animals and a classic 90s rock anthem at a backyard barbecue, planned for Game Q3 during a 14-point margin game.

### 6. High-Action Consumer Tech AI Thriller (:45s)
> A high-stakes 45-second consumer tech commercial for a smart AI home device featuring a fast-paced action movie chase with a top action movie star, showing product features instantly, planned for Game Q2 in a 7-point game.

### 7. Direct-to-Consumer FinTech Mascot (:15s)
> A punchy, disruptive 15-second Consumer Tech commercial for a commission-free investing app. A funny talking golden retriever explains compound interest to wall street traders with rapid bold packaging and immediate phone screen UI. Scheduled for Game Q3 in an 11-point game.

### 8. Rapid Direct Product Teaser in Blowout (:15s)
> A fast-paced 15-second potato chip teaser focusing purely on bold flavor, loud crunches, and immediate packaging visibility, slotted for Game Q4 when the point spread is an extreme 21-point blowout.

---

## Statistical Methodology & Governance

1. **Temporal Cross-Validation (`GroupKFold` on Year):** Models are evaluated using 5-fold temporal splitting by year to prevent chronological data leakage across multi-decade telecasts.
2. **Empirical Wikipedia Surge Modeling:** Wikipedia search models train strictly on non-null empirical spike ratio observations (`df.dropna(subset=['spike_ratio'])`), avoiding variance collapse caused by naive global median imputation.
3. **Conformal Prediction Intervals:** Dual-target heads generate 80% conformal bounds from out-of-fold residual distributions, providing defensible risk floors for media buyers and finance executives.
4. **Non-Causal Reporting Disclaimer:** In compliance with empirical marketing science standards, modeled commercial lift metrics describe observed historical cohort patterns (2000–2026) across public advertisers and do not establish direct causal retail sales revenue or stock return guarantees.
