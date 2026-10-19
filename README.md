# Super Bowl Ads Decision Engine

An interactive Streamlit app for comparing Super Bowl ad creative traits, YouTube engagement, Wikipedia pageview spikes, and brand performance.

## Requirements

- Python 3.9 or newer
- The project files listed below
- Internet access the first time you install Python packages

## Project files

Keep this structure so the app can find its data files:

```text
Group-11-SuperBowlAds/
├── app.py
├── superbowl_ads_combined_master.csv
└── data/
    └── wiki_spikes.csv
```

The app requires `superbowl_ads_combined_master.csv` and `data/wiki_spikes.csv`. Other research files and notebooks are not required to launch the current app.

## Install and run

Open Terminal and move into the project folder:

Install the packages:

```bash
python3 -m pip install streamlit pandas numpy plotly
```

Start the app:

```bash
python3 -m streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, open the **Local URL** printed in Terminal, usually `http://localhost:8501`. Keep the Terminal window open while using the app. Stop the app with **Ctrl+C**.

## App guide

- **Q1 · Trait Engagement:** compares average YouTube engagement rate and average Wikipedia spike for ads using each selected trait.
- **Q2 · Trends Over Time:** shows how the prevalence of selected creative traits changes by year across the full dataset. The highlighted area marks the selected year window.
- **Q3 · Brand Positioning:** ranks selected brands by median YouTube engagement rate. The number of ads is shown beside each bar.

Use the sidebar to select brands, a year window, and creative traits. Clear a brand selection to return to all brands.

## Metric definitions and limitations

- **YouTube engagement rate** = (likes + comments) / views. In charts it is displayed as a percentage.
- **Wikipedia spike** = peak brand-article page views on Super Bowl game day or the following day, divided by the median daily page views during the pre-game baseline period. It is a proxy for attention or search interest, not a direct measure of ad recall, persuasion, or purchase intent.
- Results describe patterns in the available historical data. They do not establish that a creative trait caused engagement, and results based on few ads should be interpreted cautiously.

The app's AI strategy recommendation is not implemented in this version.
