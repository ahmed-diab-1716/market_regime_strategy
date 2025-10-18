# one-click overnight/intraday decomposition + slippage + ban-top-k stress test
import pandas as pd
import numpy as np

# ---------- CONFIG ----------
DATA_FILE = "data.csv"
CLUSTERS_FILE = "backtest_features_with_clusters.csv"
# if you want to compare to saved strategy_return, set this to your file (optional)
SAVED_RETURNS_FILE = "backtest_portfolio_returns.csv"  # optional; will be used only if present

# Slippage scenarios to test (per-asset, in fractional units = 1 bp = 0.0001)
SLIPPAGE_LEVELS = [0.0003, 0.0005, 0.0010]  # 3 bps, 5 bps, 10 bps

# Slippage application modes:
#  - "on_change": apply slippage_per_asset only on days where previous_cluster != trade_cluster
#  - "always": apply slippage_per_asset every day (assume paying small price impact on every trade)
SLIP_MODES = ["on_change", "always"]

# How many top days to "ban" in stress test
BAN_TOP_K = [5, 10, 20]

# ---------- UTIL / WEIGHTS (edit if your mapping differs) ----------
def weights_for_cluster(c):
    try:
        c = int(c)
    except:
        return {}
    if c == 2: return {"GLD":0.25,"HYG":0.25,"TLT":0.25,"SPY":0.25}
    if c == 0: return {"SPY":0.25,"QQQ":0.25,"XLE":0.25,"IWM":0.25}
    if c == 1: return {"TLT":0.3,"GLD":0.3,"UUP":0.4}
    return {}

# ---------- LOAD FILES ----------
print("Loading files...")
data = pd.read_csv(DATA_FILE, parse_dates=["Date"])
clusters = pd.read_csv(CLUSTERS_FILE, parse_dates=["Date"])
saved_returns = None
try:
    saved_returns = pd.read_csv(SAVED_RETURNS_FILE, parse_dates=["Date"])
    print(f"Loaded saved returns from {SAVED_RETURNS_FILE}")
except Exception:
    saved_returns = None
    print("Saved returns file not found or unreadable (this is optional).")

# ---------- TICKER AUTO-DETECTION ----------
# prefer tickers derived from columns in features file (columns like 'SPY_daily_return')
ticker_candidates = []
for col in clusters.columns:
    if col.endswith("_daily_return"):
        ticker_candidates.append(col.replace("_daily_return", ""))

# fallback: detect all *_Open columns in data
if len(ticker_candidates) == 0:
    ticker_candidates = [c[:-5] for c in data.columns if c.endswith("_Open")]

tickers = sorted(set(ticker_candidates))
print("Detected tickers:", tickers)

# Validate data contains needed Open/Close for tickers
missing = []
for t in tickers:
    if f"{t}_Open" not in data.columns or f"{t}_Close" not in data.columns:
        missing.append(t)
if missing:
    raise Exception(f"Missing Open/Close columns for tickers: {missing}. Adjust DATA_FILE or tickers list.")

# ---------- BUILD overnight / intraday returns ----------
# Overnight: Open_t / Close_{t-1} - 1
# Intraday: Close_t / Open_t - 1
rets = pd.DataFrame({"Date": data["Date"]})
for t in tickers:
    rets[f"{t}_overnight"] = data[f"{t}_Open"] / data[f"{t}_Close"].shift(1) - 1
    rets[f"{t}_intraday"] = data[f"{t}_Close"] / data[f"{t}_Open"] - 1

# Merge clusters (we assume clusters.trade_cluster indicates the cluster used to trade at the open)
# If your pipeline uses cluster.shift(1) to produce trade_cluster, that's fine — we use trade_cluster as the active cluster for the day
df = pd.merge(rets, clusters[["Date","trade_cluster"]], on="Date", how="inner").sort_values("Date").reset_index(drop=True)

# compute previous_cluster (useful for slippage-on-change)
df["previous_cluster"] = df["trade_cluster"].shift(1)

# ---------- compute per-day portfolio contributions ----------
def compute_portfolio_returns(df):
    overnight_portfolio = []
    intraday_portfolio = []
    total_portfolio = []
    n_assets_traded = []

    for _, row in df.iterrows():
        cl = row["trade_cluster"]
        w = weights_for_cluster(cl)
        if not w:
            overnight_portfolio.append(0.0)
            intraday_portfolio.append(0.0)
            total_portfolio.append(0.0)
            n_assets_traded.append(0)
            continue
        # normalize weights
        s = sum(w.values())
        if s == 0:
            overnight_portfolio.append(0.0)
            intraday_portfolio.append(0.0)
            total_portfolio.append(0.0)
            n_assets_traded.append(0)
            continue
        if abs(s-1.0) > 1e-9:
            w = {k:v/s for k,v in w.items()}

        on_ret = 0.0
        in_ret = 0.0
        cnt = 0
        for sym, weight in w.items():
            on_r = row.get(f"{sym}_overnight", 0.0)
            in_r = row.get(f"{sym}_intraday", 0.0)
            if pd.isna(on_r): on_r = 0.0
            if pd.isna(in_r): in_r = 0.0
            on_ret += weight * on_r
            in_ret += weight * in_r
            cnt += 1
        overnight_portfolio.append(on_ret)
        intraday_portfolio.append(in_ret)
        total_portfolio.append(on_ret + in_ret)
        n_assets_traded.append(cnt)

    df["overnight_ret"] = overnight_portfolio
    df["intraday_ret"] = intraday_portfolio
    df["total_ret"] = total_portfolio
    df["n_assets_traded"] = n_assets_traded
    return df

df = compute_portfolio_returns(df)

# ---------- SUMMARY: baseline (no slippage) ----------
def print_summary(df, label="BASE"):
    print(f"\n--- Summary: {label} ---")
    print("Days:", df.shape[0])
    print("Mean overnight:", round(df["overnight_ret"].mean(),6))
    print("Mean intraday:", round(df["intraday_ret"].mean(),6))
    print("Mean combined:", round(df["total_ret"].mean(),6))
    print("Fraction positive overnight days:", round((df["overnight_ret"]>0).mean(),3))
    print("Fraction positive intraday days:", round((df["intraday_ret"]>0).mean(),3))

    df["wealth_overnight"] = (1 + df["overnight_ret"]).cumprod()
    df["wealth_intraday"] = (1 + df["intraday_ret"]).cumprod()
    df["wealth_total"] = (1 + df["total_ret"]).cumprod()

    print("Final wealth overnight-only:", round(df["wealth_overnight"].iloc[-1],6))
    print("Final wealth intraday-only:", round(df["wealth_intraday"].iloc[-1],6))
    print("Final wealth combined:", round(df["wealth_total"].iloc[-1],6))

    # per-cluster overnight/intraday means
    try:
        grp = df.groupby("trade_cluster")[["overnight_ret","intraday_ret","total_ret"]].mean()
        print("\nPer-cluster mean returns (overnight, intraday, total):")
        print(grp)
    except Exception:
        pass

    # top contributors
    top_total = df.nlargest(20, "total_ret")[["Date","overnight_ret","intraday_ret","total_ret"]]
    print("\nTop 20 days by total_ret (date, overnight, intraday, total):")
    print(top_total.to_string(index=False))
    return df

df = print_summary(df, label="NO SLIPPAGE (BASE)")

# ---------- SLIPPAGE SENSITIVITY ----------
print("\n\n=== SLIPPAGE SENSITIVITY ===")
results_slip = []
for slip in SLIPPAGE_LEVELS:
    for mode in SLIP_MODES:
        df2 = df.copy()
        if mode == "on_change":
            # only apply slippage where previous_cluster differs from trade_cluster
            change_mask = (df2["previous_cluster"].notna()) & (df2["previous_cluster"] != df2["trade_cluster"])
            df2["overnight_ret_with_slip"] = df2["overnight_ret"] - df2["n_assets_traded"] * slip * change_mask.astype(int)
            df2["intraday_ret_with_slip"] = df2["intraday_ret"]  # assume intraday no extra slip here for simplicity
        else:  # "always"
            df2["overnight_ret_with_slip"] = df2["overnight_ret"] - df2["n_assets_traded"] * slip
            df2["intraday_ret_with_slip"] = df2["intraday_ret"]

        df2["total_ret_with_slip"] = df2["overnight_ret_with_slip"] + df2["intraday_ret_with_slip"]
        wealth = (1 + df2["total_ret_with_slip"]).cumprod().iloc[-1]
        results_slip.append((slip, mode, wealth))
        print(f"slip={slip:.6f}, mode={mode}, final wealth={wealth:.6f}")

# ---------- BAN TOP-K DAYS STRESS TEST ----------
print("\n\n=== BAN-TOP-K DAYS (zero-out top contributors) ===")
# compute daily multiplicative contribution (wealth change) and use to rank
df["daily_mult"] = (1 + df["total_ret"]).cumprod().pct_change().fillna(df["wealth_total"].iloc[0])
# simpler ranking by total_ret
for k in BAN_TOP_K:
    df_ban = df.copy()
    topk_dates = df_ban.nlargest(k, "total_ret")["Date"].tolist()
    df_ban.loc[df_ban["Date"].isin(topk_dates), "total_ret_ban"] = 0.0
    df_ban["total_ret_ban"] = df_ban["total_ret_ban"].fillna(df_ban["total_ret"])
    wealth_ban = (1 + df_ban["total_ret_ban"]).cumprod().iloc[-1]
    print(f"Ban top {k} days (zero those day returns) -> final wealth = {wealth_ban:.6f}")

# ---------- OPTIONAL: Compare to saved strategy_return if present ----------
if saved_returns is not None:
    merged = pd.merge(saved_returns, df[["Date","total_ret"]], on="Date", how="inner")
    merged["diff"] = merged["strategy_return"] - merged["total_ret"]
    print("\n\n=== COMPARE saved strategy_return vs recomputed total_ret ===")
    print("Rows compared:", merged.shape[0])
    print("abs diff > 1e-9 count:", (merged["diff"].abs() > 1e-9).sum())
    print("diff stats (min, mean, max):", merged["diff"].min(), merged["diff"].mean(), merged["diff"].max())
    if (merged["diff"].abs() > 1e-9).sum() > 0:
        print("Sample diffs:")
        print(merged.loc[merged["diff"].abs() > 1e-9].head(10).to_string(index=False))

print("\n\n=== DONE ===")