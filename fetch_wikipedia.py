"""Pull Wikipedia daily pageviews for each brand around each Super Bowl (2016+).
Wikimedia pageviews data only starts July 2015, so earlier games can't be covered.
Run:  python fetch_wikipedia.py
Outputs: data/wiki_daily.csv (daily series) and data/wiki_spikes.csv (one row per brand-year)
"""
import os
import time
import urllib.parse

import pandas as pd
import requests

# Wikimedia asks for a descriptive User-Agent with contact info. Edit this.
HEADERS = {"User-Agent": "Group11-SuperBowlAds-class-project (your_email@example.edu)"}
API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "en.wikipedia/all-access/user/{title}/daily/{start}/{end}")

GAME_DATES = {
    2016: "2016-02-07", 2017: "2017-02-05", 2018: "2018-02-04", 2019: "2019-02-03",
    2020: "2020-02-02", 2021: "2021-02-07", 2022: "2022-02-13", 2023: "2023-02-12",
    2024: "2024-02-11", 2025: "2025-02-09", 2026: "2026-02-08",
}

# Brand name in your CSV -> Wikipedia article title. VERIFY each by opening the page
# and copying the title from the URL; redirects are NOT followed by the pageviews API.
ARTICLES = {
    "Bud Light": "Bud_Light",
    "Budweiser": "Budweiser",
    "Pepsi": "Pepsi",
    "Doritos": "Doritos",
    "Kia": "Kia",
    "Coca-Cola": "Coca-Cola",
    "Hyundai": "Hyundai_Motor_Company",
    "NFL": "National_Football_League",
    "Toyota": "Toyota",
    "E-Trade": "E-Trade",
}

BEFORE, AFTER = 21, 14  # days before / after game day to download


def daily_views(title, start, end):
    url = API.format(title=urllib.parse.quote(title, safe=""),
                     start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:  # no data or wrong title
        print(f"  no data for {title} ({start.date()}); check the title")
        return pd.Series(dtype=float)
    r.raise_for_status()
    s = pd.Series({pd.to_datetime(i["timestamp"][:8]): i["views"] for i in r.json()["items"]})
    # the API omits days with zero views, so fill them in
    return s.reindex(pd.date_range(start, end), fill_value=0)


def main():
    os.makedirs("data", exist_ok=True)
    daily_rows, spike_rows = [], []
    for brand, title in ARTICLES.items():
        for year, game in GAME_DATES.items():
            g = pd.Timestamp(game)
            s = daily_views(title, g - pd.Timedelta(days=BEFORE), g + pd.Timedelta(days=AFTER))
            time.sleep(0.2)
            if s.empty:
                continue
            for d, v in s.items():
                daily_rows.append({"brand": brand, "year": year, "date": d,
                                   "days_from_game": (d - g).days, "views": v})
            base = s[(s.index >= g - pd.Timedelta(days=BEFORE)) & (s.index <= g - pd.Timedelta(days=4))]
            # ads air in the evening US time, so UTC "game day + 1" can hold the spike
            peak = s[(s.index >= g) & (s.index <= g + pd.Timedelta(days=1))].max()
            baseline = base.median()
            spike_rows.append({"brand": brand, "year": year, "baseline_median": baseline,
                               "peak_views": peak,
                               "spike_ratio": peak / baseline if baseline > 0 else None})
    pd.DataFrame(daily_rows).to_csv("data/wiki_daily.csv", index=False)
    pd.DataFrame(spike_rows).to_csv("data/wiki_spikes.csv", index=False)
    print(f"Saved {len(daily_rows)} daily rows and {len(spike_rows)} brand-year spikes.")


if __name__ == "__main__":
    main()
