import pandas as pd
import numpy as np

# 1. Load market returns and expanded ad records
returns_df = pd.read_parquet("superbowl_advertisers_daily_returns_2000_2026.parquet")
ads_df = pd.read_csv("superbowl-ads-admeter-sales-impact-expanded.csv")

# 2. Historical Super Bowl Monday dates (first trading day post-game)
SB_MONDAYS = {
    2000: "2000-01-31", 2001: "2001-01-29", 2002: "2002-02-04", 2003: "2003-01-27",
    2004: "2004-02-02", 2005: "2005-02-07", 2006: "2006-02-06", 2007: "2007-02-05",
    2008: "2008-02-04", 2009: "2009-02-02", 2010: "2010-02-08", 2011: "2011-02-07",
    2012: "2012-02-06", 2013: "2013-02-04", 2014: "2014-02-03", 2015: "2015-02-02",
    2016: "2016-02-08", 2017: "2017-02-06", 2018: "2018-02-05", 2019: "2019-02-04",
    2020: "2020-02-03", 2021: "2021-02-08", 2022: "2022-02-14", 2023: "2023-02-13",
    2024: "2024-02-12", 2025: "2025-02-10", 2026: "2026-02-09"
}

# 3. Compute CAR[-1, +5] per commercial
car_results = []
market_col = "^GSPC"

for idx, row in ads_df.iterrows():
    ticker = row["ticker_symbol"]
    year = int(row["year"])
    
    # Handle symbol remaps
    if ticker == "BMWYY": ticker = "BMW.DE"
    elif ticker == "K": ticker = "GIS"
    elif ticker == "SKX": ticker = "DECK"
    elif ticker == "HYMTF": ticker = "005380.KS"
    
    if ticker not in returns_df.columns or year not in SB_MONDAYS:
        car_results.append(np.nan)
        continue
        
    t0_date = pd.to_datetime(SB_MONDAYS[year])
    if t0_date not in returns_df.index:
        car_results.append(np.nan)
        continue
        
    t0_idx = returns_df.index.get_loc(t0_date)
    
    # Require at least 250 trading days prior for estimation window
    if t0_idx < 250:
        car_results.append(np.nan)
        continue
        
    # Estimation Window: t-250 to t-31
    est_data = returns_df.iloc[t0_idx-250 : t0_idx-30][[ticker, market_col]].dropna()
    if len(est_data) < 100:
        car_results.append(np.nan)
        continue
        
    # OLS Market Model: R_i = alpha + beta * R_m
    cov_matrix = np.cov(est_data[ticker], est_data[market_col])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    alpha = est_data[ticker].mean() - beta * est_data[market_col].mean()
    
    # Event Window: t-1 to t+5
    event_data = returns_df.iloc[t0_idx-1 : t0_idx+6][[ticker, market_col]].dropna()
    expected_ret = alpha + beta * event_data[market_col]
    abnormal_ret = event_data[ticker] - expected_ret
    car_val = abnormal_ret.sum() * 100.0  # Express in %
    
    car_results.append(round(car_val, 2))

ads_df["car_7d_pct"] = car_results
ads_df.to_csv("superbowl_ads_with_car.csv", index=False)
print(f"Calculated CAR for {ads_df['car_7d_pct'].notna().sum()} commercial entries.")