# Super Bowl Ads Decision Engine

An interactive Streamlit app for comparing Super Bowl ad creative traits, YouTube engagement, Wikipedia pageview spikes, and brand performance.

## Requirements

- Python 3.9 or newer
- The project files and directory structure listed below
- Internet access the first time you install Python packages

## Project Structure

The repository is organized into distinct functional layers:

```text
Group-11_Superbowl_Ads/
├── app.py                                          # Main Streamlit dashboard application
├── requirements.txt                                # Python dependencies
├── README.md                                       # Project documentation
├── .gitignore                                      # Git configuration
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

## Install and Run

Open Terminal and move into the project folder:

Install the packages:

```bash
python3 -m pip install -r requirements.txt
# or: python3 -m pip install streamlit pandas numpy plotly statsmodels scikit-learn shap pyarrow scipy
```

Start the app:

```bash
python3 -m streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, open the **Local URL** printed in Terminal, usually `http://localhost:8501`. Keep the Terminal window open while using the app. Stop the app with **Ctrl+C**.

## App Guide

- **T1 · Trait Engagement:** compares average YouTube engagement rate, Google Trends commercial search lift (with longitudinal Financial Q1–Q4 decay), and average Wikipedia spike for ads using each selected trait.
- **T2 · Trends Over Time:** shows how the prevalence of selected creative traits changes by year across the full dataset. The highlighted area marks the selected year window.
- **T3 · Brand Positioning:** ranks selected brands by median YouTube engagement rate and views, with hover metrics detailing commercial search lift and multi-quarter retention.
- **T4 · Reach vs. Dialogue:** models the power-law relationship between raw impressions and active dialogue (comments).
- **Phase 2 · Telecast Flow & Financial Return:** analyzes telecast ratings decay by Game Quarter (Game Q1–Game Q4), Effective CPM, and 7-day cumulative abnormal stock returns (CAR).
- **AI Integration: Optimization & Airtime Allocation:** interactive prescriptive decision sandbox outlining the machine learning architecture for forecasting Ad Meter scores, 4-week sales volume lift, and optimal broadcast timing.

Use the sidebar to select brands, a year window, and creative traits. Clear a brand selection to return to all brands.

## Metric Definitions and Limitations

- **Game Quarters (Game Q1–Game Q4)** = the four 15-minute quarters of the live Super Bowl telecast broadcast (plus Halftime), determining living-room viewer size, blowout tune-outs, and Effective CPM.
- **Financial Quarters (Financial Q1–Financial Q4)** = the four 3-month fiscal periods following the Super Bowl ad (Q1: immediate post-game/spring, Q2: spring/summer, Q3: summer, Q4: fall/holiday), used to track long-term search interest decay.
- **YouTube engagement rate** = (likes + comments) / views. In charts it is displayed as a percentage.
- **Google Trends search lift** = percentage change in Google US search query volume index during the commercial broadcast window relative to the pre-game baseline, with longitudinal decay tracked across the following four Financial Quarters (Financial Q1–Financial Q4).
- **Wikipedia spike** = peak brand-article page views on Super Bowl game day or the following day, divided by the median daily page views during the pre-game baseline period. It is a proxy for attention or search interest, not a direct measure of ad recall, persuasion, or purchase intent.
- Results describe patterns in the available historical data. They do not establish that a creative trait caused engagement, and results based on few ads should be interpreted cautiously.
