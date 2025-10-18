import pandas as pd
import numpy as np

# files
data = pd.read_csv("data.csv", parse_dates=["Date"])
clusters = pd.read_csv("backtest_features_with_clusters.csv", parse_dates=["Date"])
saved = pd.read_csv("backtest_portfolio_returns.csv", parse_dates=["Date"])

# your weights mapping (ensure this matches your code)
def weights_for_cluster(c):
    try: c = int(c)
    except: return {}
    if c == 2: return {"GLD":0.25,"HYG":0.25,"TLT":0.25,"SPY":0.25}
    if c == 0: return {"SPY":0.25,"QQQ":0.25,"XLE":0.25,"IWM":0.25}
    if c == 1: return {"TLT":0.3,"GLD":0.3,"UUP":0.4}
    return {}

# derive tickers from data columns (Open)
tickers = [c[:-5] for c in data.columns if c.endswith("_Open")]
tickers = sorted(tickers)

# compute open-to-open returns per asset (open_t / open_{t-1} - 1)
open_rets = pd.DataFrame({"Date": data["Date"]})
for t in tickers:
    open_rets[f"{t}_open_open"] = data[f"{t}_Open"].pct_change().fillna(0)

# merge with clusters (use trade_cluster as used by saved strategy)
df = pd.merge(open_rets, clusters[["Date","trade_cluster"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
df["prev_cluster"] = df["trade_cluster"].shift(1)  # if your saved strategy used shift, handle appropriately

# compute portfolio open-to-open returns and per-asset contributions
def row_portfolio_open_open(row):
    cl = row["trade_cluster"]
    w = weights_for_cluster(cl)
    if not w: return 0.0, {}
    s = sum(w.values())
    if abs(s-1.0) > 1e-9:
        w = {k:v/s for k,v in w.items()}
    contribs = {}
    total = 0.0
    for sym, weight in w.items():
        col = f"{sym}_open_open"
        val = row.get(col, 0.0)
        contribs[sym] = weight * val
        total += contribs[sym]
    return total, contribs

all_rows = []
for _, r in df.iterrows():
    tot, contribs = row_portfolio_open_open(r)
    rowd = {"Date": r["Date"], "portfolio_open_open": tot}
    rowd.update(contribs)
    all_rows.append(rowd)

df_port = pd.DataFrame(all_rows)
df = pd.merge(df, df_port, on="Date", how="left")

# Comparison to saved strategy_return
cmp = pd.merge(saved, df[["Date","portfolio_open_open"]], on="Date", how="inner")
cmp["diff"] = cmp["strategy_return"] - cmp["portfolio_open_open"]
print("Comparison: abs diff>1e-9 count:", (cmp["diff"].abs()>1e-9).sum())
print("diff stats:", cmp["diff"].min(), cmp["diff"].mean(), cmp["diff"].max())

# Output top days with per-asset contributions (by saved strategy importance)
top_dates = saved.nlargest(20, "strategy_return")["Date"].dt.strftime("%Y-%m-%d").tolist()
top_df = df[df["Date"].dt.strftime("%Y-%m-%d").isin(top_dates)].copy()
top_df = top_df.sort_values("Date")
top_df.to_csv("top20_per_asset_contribs.csv", index=False)
print("Wrote top20_per_asset_contribs.csv with per-asset contributions for the top 20 saved-return days.")