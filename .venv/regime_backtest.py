import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns


# --------------------------
# 1. Load models and data
# --------------------------
scaler = joblib.load("scaler.pkl")
pca_model = joblib.load("pca_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
cluster_assigner = joblib.load("knn_model.pkl")

features_raw = pd.read_csv("raw_features.csv")
data = pd.read_csv("data.csv")
numeric_features = pd.read_csv("numeric_features.csv")  # columns used in training

features_raw["Date"] = pd.to_datetime(features_raw["Date"])
data["Date"] = pd.to_datetime(data["Date"])

kept_cols = numeric_features.columns.tolist()

# --------------------------
# 2. Select out-of-sample period
# --------------------------
train_end = pd.Timestamp("2019-12-31")
backtest_features = features_raw[features_raw["Date"] > train_end].copy()
backtest_data = data[data["Date"] > train_end].copy()

# --------------------------
# Scale, PCA, assign clusters

backtest_numeric = backtest_features[kept_cols].copy()
print(backtest_numeric.shape)
scaled_backtest = pd.DataFrame(data=scaler.transform(backtest_numeric),columns=kept_cols,index=backtest_numeric.index)
explained_variance = pca_model.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)    
wanted_variance = 0.5
n_components = np.argmax(cumulative_explained_variance > wanted_variance) + 1
pca_backtest = pd.DataFrame(data=pca_model.transform(scaled_backtest),columns=[f"PC{i+1}" for i in range(0,n_components)],index=scaled_backtest.index)


backtest_features["cluster"] = cluster_assigner.predict(pca_backtest)


# Shift clusters to trade next day to ensure no data leakage or look ahead bias

backtest_features["trade_cluster"] = backtest_features["cluster"].shift(1)
backtest_features = backtest_features.dropna(subset=["trade_cluster"])
backtest_features["trade_cluster"]=backtest_features["trade_cluster"]
backtest_features = backtest_features.dropna(subset=["trade_cluster"])
clusters = backtest_features["trade_cluster"].values
for i in range(1, len(clusters)-1):
    if clusters[i-1] == clusters[i+1] != clusters[i]:
        clusters[i] = clusters[i-1]
backtest_features["trade_cluster"] = clusters

# Computing Close-to-Close returns

portfolio_assets = ["SPY_Close", "QQQ_Close", "TLT_Close", "IWM_Close","GLD_Close","HYG_Close","UUP_Close","XLE_Close","XLF_Close","USO_Close"]
returns_df = backtest_data[["Date"] + portfolio_assets].sort_values("Date").reset_index(drop=True)
print(returns_df.head())


# Computing daily percentage returns
returns_pct = returns_df.copy()
returns_pct[portfolio_assets] = returns_pct[portfolio_assets].pct_change().fillna(0)

# Merging clusters
returns_with_clusters = returns_pct.merge(
    backtest_features[["Date", "trade_cluster"]],
    on="Date",
    how="inner"
)

returns_with_clusters["previous_cluster"] = returns_with_clusters["trade_cluster"].shift(1)

momentum_window = 7
benchmarks = ["SPY_Close", "QQQ_Close", "IWM_Close","TLT_Close"]

# Compute cumulative return *up to yesterday* (shift by 1)
momentum_df = (1 + returns_with_clusters[benchmarks]).rolling(momentum_window).apply(np.prod, raw=True) - 1
momentum_df = momentum_df.shift(1)  # shift to avoid using today's return

# Take the average momentum across benchmarks
returns_with_clusters["market_momentum"] = momentum_df.mean(axis=1)

# Classify momentum regime
returns_with_clusters["momentum_state"] = np.where(
    returns_with_clusters["market_momentum"] > 0, "strong", "weak"
)

returns_with_clusters["previous_momentum_state"] = returns_with_clusters["momentum_state"].shift(1)


# Constructing trading strategy based on analysis of regimes
def cluster_strategy(row):
    cluster = int(row["trade_cluster"])
    previous_cluster = row["previous_cluster"]
    momentum_state = row.get("momentum_state", "weak")
    previous_momentum_state = row["previous_momentum_state"]
    day_return = 0

    # Define base cluster portfolios
    if cluster == 2:
        
        # sideways / mixed
        if momentum_state == "strong":
            day_return = 0.25 * row["SPY_Close"] + 0.25 * row["QQQ_Close"] + 0.25 * row["IWM_Close"] + 0.25 * row["GLD_Close"]
        else:
            day_return = (row["SPY_Close"] * -0.2 + row["XLE_Close"] * 0.2 + row["IWM_Close"] * 0.15 + row["GLD_Close"] * 0.15 + row["XLF_Close"] * 0.15 + row["UUP_Close"] * 0.15)
        
    elif cluster == 1:
        # bull market
        if momentum_state == "strong":
            day_return = 0.3 * row["SPY_Close"] + 0.3 * row["QQQ_Close"] + 0.2 * row["XLE_Close"] + 0.2 * row["IWM_Close"]
        else:
            day_return = 0.8*(-0.25 * row["SPY_Close"] + 0.2 * row["QQQ_Close"] + 0.2 * row["XLE_Close"] + 0.2 * row["IWM_Close"] + 0.15 * row["UUP_Close"])

    elif cluster == 0:
        # defensive market
        if momentum_state == "strong":
            # start rotating off safe assets slightly
            day_return =  0.3 * row["SPY_Close"] + 0.2 * row["GLD_Close"] + 0.2 * row["TLT_Close"] + 0.2 * row["UUP_Close"]
        else:
            day_return = (0.3 * row["TLT_Close"] + 0.3 * row["GLD_Close"] + 0.4 * row["UUP_Close"])

    else:
        day_return = 0

    # Add a small commission when cluster switches
    if (pd.notna(previous_cluster) and previous_cluster != cluster) or (pd.notna(previous_momentum_state) and previous_momentum_state != momentum_state):
        day_return -= 0.0001

    return day_return


returns_with_clusters["strategy_return"] = returns_with_clusters.apply(cluster_strategy, axis=1)


# Computing cumulative returns

returns_with_clusters["cumulative_return"] = (1 + returns_with_clusters["strategy_return"]).cumprod()
returns_with_clusters["SPY_cumulative_return"] = (1 + returns_with_clusters["SPY_Close"]).cumprod() 
returns_with_clusters["strat_2"] = (1 + 0.25 * returns_with_clusters["GLD_Close"] + 0.25 * returns_with_clusters["HYG_Close"] +
0.25 * returns_with_clusters["TLT_Close"] + 0.25 * returns_with_clusters["SPY_Close"]).cumprod()
returns_with_clusters["strat_1"] = (1 + 0.25 * returns_with_clusters["SPY_Close"] + 0.25 * returns_with_clusters["QQQ_Close"] +
0.25 * returns_with_clusters["XLE_Close"] + 0.25 * returns_with_clusters["IWM_Close"]).cumprod()

returns_with_clusters["Date"] = pd.to_datetime(returns_with_clusters["Date"])
returns_with_clusters = returns_with_clusters.sort_values("Date")
print(returns_with_clusters.head(20))

plt.figure(figsize=(12,6))
plt.plot(returns_with_clusters["Date"], returns_with_clusters["cumulative_return"], label="Backtest Strategy")
#plt.plot(returns_with_clusters["Date"], returns_with_clusters["SPY_cumulative_return"], label="SPY_return",alpha=0.8)
plt.plot(returns_with_clusters["Date"], returns_with_clusters["strat_2"], label="sideways strat",alpha=0.8)
plt.plot(returns_with_clusters["Date"], returns_with_clusters["strat_1"], label="growth strat",alpha=0.8)
plt.title("Out-of-Sample Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.show()


backtest_features.to_csv("backtest_features_with_clusters.csv", index=False)
returns_with_clusters[["Date", "strategy_return", "cumulative_return"]].to_csv(
    "backtest_portfolio_returns.csv", index=False)

plot_df = returns_df[["Date","SPY_Close"]].merge(backtest_features[["Date","trade_cluster"]],on="Date",how="inner")

plt.figure(figsize=(14,6))
sns.scatterplot(
    x="Date",
    y="SPY_Close",
    hue="trade_cluster",
    data=plot_df,
    palette="tab10",
    s=5
)
plt.title("SPY Closing Price Colored by Market Regime")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend(title="Cluster")
plt.show()

plot_df = returns_df[["Date","SPY_Close"]].merge(returns_with_clusters[["Date","momentum_state"]],on="Date",how="inner")

plt.figure(figsize=(14,6))
sns.scatterplot(
    x="Date",
    y="SPY_Close",
    hue="momentum_state",  
    data=plot_df,
    palette={"strong": "green", "weak": "red"}, 
    s=5,                      
    alpha=0.7
)
plt.title("SPY Closing Price Colored by Momentum State")
plt.xlabel("Date")
plt.ylabel("SPY Close Price")
plt.legend(title="Momentum")
plt.show()

def sharpe(returns, freq=252):
    return np.sqrt(freq) * returns.mean() / returns.std() if returns.std() != 0 else np.nan

print("Strategy Sharpe:", sharpe(returns_with_clusters["strategy_return"]))
print("SPY Sharpe:", sharpe(returns_with_clusters["SPY_Close"]))
mdd = (1 + returns_with_clusters["cumulative_return"])
drawdown = mdd / mdd.cummax() - 1
print("Max drawdown:", drawdown.min())


"""
Overall we have constructed a low risk market strategy with modest annual returns. I think the key to improving the returns would be more technical analysis of
the cluster 1. One could consider perhaps clustering elements of the transition cluster again and seeing what appears.
The model seems to perform excellently in crisis and the combination of the clustering with the momentum signal allows one to benefit off of these situations greatly.
also a more powerful classifier as opposed to KNN may also be more effective in assigning clusters
"""