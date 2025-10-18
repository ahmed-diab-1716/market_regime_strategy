import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np


"""
In this file I attempt to analyse my clusters to gain insights as to the properties of each cluster (regime) in order to
determine an effective trading strategy
"""
#Importing data from the training file and cleaning the dataframes

pca_data = pd.read_csv("pca_data.csv",index_col=0).reset_index()
features = pd.read_csv("raw_features.csv",index_col=0).reset_index()
data = pd.read_csv("data.csv",index_col=0).reset_index()
clusters_df = pd.read_csv("clusters.csv",index_col=0).reset_index()
clusters_df["smoothed_cluster"] = clusters_df["smoothed_cluster"].ffill().bfill()
pca = joblib.load("pca_model.pkl")


clusters_df["Date"] = pd.to_datetime(clusters_df["Date"])
data["Date"] = pd.to_datetime(data["Date"])
features["Date"] = pd.to_datetime(features["Date"])
pca_data["Date"] = pd.to_datetime(pca_data["Date"])



pca_df = (pca_data.merge(clusters_df, on="Date"))


#PC1 vs PC2 Plot coloured by cluster

plt.figure(figsize=(10, 14))
sns.scatterplot(
    data=pca_df,
    x="PC1", y="PC2",
    hue="cluster",
    palette="tab10",
    s=5
)
plt.title("PCA: PC1 vs PC2 Colored by Cluster")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.show()

"""
PC2 vs PC3 Plot coloured by cluster
"""
plt.figure(figsize=(10, 14))
sns.scatterplot(
    data=pca_df,
    x="PC2", y="PC3",
    hue="cluster",
    palette="tab10",
    s=5
)
plt.title("PCA: PC2 vs PC3 Colored by Cluster")
plt.xlabel("Principal Component 2")
plt.ylabel("Principal Component 3")
plt.legend(title="Cluster")
plt.show()

"""
PC1 vs PC3 Plot coloured by cluster
"""
plt.figure(figsize=(10, 14))
sns.scatterplot(
    data=pca_df,
    x="PC1", y="PC3",
    hue="cluster",
    palette="tab10",
    s=5
)
plt.title("PCA: PC1 vs PC3 Colored by Cluster")
plt.xlabel("Principal Component 2")
plt.ylabel("Principal Component 3")
plt.legend(title="Cluster")
plt.show()

"""
The 3 PCA Plots above seem to indicate that PC1 clearly separates clusters 1 and 2 from cluster 0
while PC2 is the main driving factor separating clusters 1 and 2
PC3 has little influence in separating clusters 0 and 1 but there may be slight separation based on PC3
"""


# Make a DataFrame containing the info for plotting
plot_df = (data.merge(clusters_df,on="Date"))

plt.figure(figsize=(14,6))
sns.scatterplot(
    x="Date",
    y="SPY_Close",
    hue="cluster",
    data=plot_df,
    palette="tab10",
    s=5
)
plt.title("SPY Closing Price Colored by Market Regime")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend(title="Cluster")
plt.show()

features_with_clusters = features.merge(clusters_df, on="Date")
loadings = pd.read_csv("loadings.csv")
loadings.rename(columns={loadings.columns[0]:"Feature"},inplace=True)
PC1_key_features = loadings.loc[loadings['PC1'].abs().nlargest(15).index, ['Feature', 'PC1']]
PC2_key_features = loadings.loc[loadings['PC2'].abs().nlargest(15).index, ['Feature', 'PC2']]
PC3_key_features = loadings.loc[loadings['PC3'].abs().nlargest(15).index, ['Feature', 'PC3']]
print(PC1_key_features)
print(PC2_key_features)
print(PC3_key_features)

"""
By analysing the biggest loadings of PC1, PC2 and PC3 we can see that:
PC1 seems to indicate overall market trend
PC2 is a more subtle PC driven by interasset correlations, which could be used to perform some pairs trading for example
PC3 is driven by short term market behaviour

This makes sense as to why PC3 has limited power in separating the clusters as short term market behaviour is much more noisy
and so will have less of an impact on long term market
"""

"""
We now analyse the cluster means, standard deviations and skew to identify how the PCs and certain assets behave across clusters
"""

numeric_features_with_clusters = features_with_clusters.select_dtypes(include=[np.number]).copy()
pca_df.drop(columns=["smoothed_cluster","index","Date"],inplace=True)
cluster_means = pca_df.groupby("cluster").mean()

print(cluster_means)

"""
We see that cluster 0 has a very negative PC1 mean, implying that cluster 0 is a "Bear" market regime
we see that clusters 1 and 2 seem to be more of a bull market with low volatility and greater returns

Observing the PC2 means we can see that in cluster 0 assets are highly uncorrelated, which reflects what happens in a chaotic high volatility bear regime
cluster 1 has lots of highly correlated assets implying the market is growing
whereas cluster 2 there many more key assets are negatively correlated implying more of a sideways regime.

moreover in PC3 we see that cluster 1 seems to have a much more negative short term returns implying that cluster 1 may be more volatile compared to cluster 2

We note that PC4 onwards the cluster centroids get closer and likely become less informative

"""

cols_of_interest = ["SPY_daily_return","TLT_daily_return","GLD_daily_return","QQQ_daily_return","HYG_daily_return","UUP_daily_return","IWM_daily_return","XLE_daily_return"]

cluster_means = numeric_features_with_clusters.groupby("cluster").mean()
print(cluster_means[cols_of_interest])

cluster_stds = numeric_features_with_clusters.groupby("cluster").std()
print(cluster_stds[cols_of_interest])

cluster_skews = numeric_features_with_clusters.groupby("cluster").skew()
print(cluster_skews[cols_of_interest])

"""
looking at the mean returns of genuine assets cluster 1 appears to indicate the most growth oriented market despite the findings of the PC1, this is likely due
to the slightly higher mean volatility of cluster 1.
cluster 0 is characteristically high volatility so we must be careful of changing rapidly changing market patterns.

"""
"""
There is potential to exploit changes in correlation between clusters 1 and 2 but in essence a simple strategy might be:

Cluster 0:
limit invested capital to limit losses but look to gain with assets such as TLT bonds and GLD. Try to catch the rebound early by analysing short term returns
and come off safety assets

Cluster 1:
Invest into growth assets (SPY, XLE, IWM) and hold (low skew market should mean things are ok)

Cluster 2:
more tricky due to negative skew but look to exploit non correlated assets in the transition between cluster 2 and cluster 1.
More diverse portfolio, benefit from smaller gains and make trades more frequently
"""