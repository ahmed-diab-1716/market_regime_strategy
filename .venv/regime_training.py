
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture

# Reading in the data from yfinance and cleaning up the DataFrame (removing multiindices)

tickers = ["SPY","TLT","GLD","QQQ","HYG","UUP","IWM","USO","XLF","XLE","^VIX"]
interval = "1d"
period = "18y"


data = yf.download(tickers = tickers, interval = interval, period = period, group_by="ticker", auto_adjust = True)

if len(tickers) > 1:
    data.columns = ["_".join(col).strip() for col in data.columns.values]
data.reset_index(inplace=True)

print(data.isna().sum())

print(data.head())

#Initial Feature Creation

new_cols = {}   

for ticker in tickers:
        prefix = ticker + "_"
        new_cols[prefix + "daily_return"] = data[prefix + "Close"].pct_change()
features = pd.DataFrame(new_cols,index=data.index)


# Function which creates rolling features based based on daily returns and returns a dataframe of the features.

def create_rolling_features(df,window=21,tickers=tickers):
    new_cols = {}
    str_window = "_" + str(window) +"_day"
    
    for ticker in tickers:
        prefix = ticker + "_"
        
        if window > 7:
            new_cols[prefix + "ewm_return" + str_window] = df[prefix + "daily_return"].ewm(span=window,adjust=False).mean()
            new_cols[prefix + "rolling_vol" + str_window] = df[prefix + "daily_return"].rolling(window).std()
        
        if window < 21:
            new_cols[prefix + "mean_return" + str_window + "/4window_return"] = df[prefix + "daily_return"].rolling(window).mean()/df[prefix + "daily_return"].rolling(4 * window).mean()
            new_cols[prefix + "return_delta" + str_window] = df[prefix +"daily_return"] - df[prefix + "daily_return"].shift(window)
        else:
            new_cols[prefix + "vol_adjusted_mean" + str_window] = df[prefix +"daily_return"].rolling(window).mean()/df[prefix +"daily_return"].rolling(window).std()

    rolling_features = pd.DataFrame(new_cols,index=data.index)
    rolling_features = rolling_features.dropna()
    return rolling_features

#Creating correlation features based on the assets chosen.

def create_correlation_features(df,window = 21,tickers = tickers):
    
    new_cols = {}
    str_window = "_" + str(window) +"_day"
    
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i+1:]:
            col_name = f"{t1}_{t2}_corr"+str_window
            new_cols[col_name] = (
                df[f"{t1}_daily_return"]
                .rolling(window)
                .corr(features[f"{t2}_daily_return"])
            )
    corr_features = pd.DataFrame(data=new_cols,index=df.index)
    
    derived_cols = {}
    
    for col in corr_features.columns:
        derived_cols[col + "_mean"] = corr_features[col].ewm(span=window,adjust=False).mean()

    derived_features = pd.DataFrame(data=derived_cols, index=df.index)
    derived_features = derived_features.loc[df.index].dropna()
    print(derived_features.head())
    return derived_features

 

#Generating features from the functions and combining them into one dataframe.

rolling_features_21_day = create_rolling_features(features,21)
rolling_features_7_day = create_rolling_features(features,7)
rolling_features_3_day = create_rolling_features(features,3)
rolling_features_63_day = create_rolling_features(features,63)
correlation_features_21_days = create_correlation_features(features,21)


features = pd.concat([features,
                      rolling_features_7_day,
                      rolling_features_21_day,
                      rolling_features_63_day,
                      rolling_features_3_day,
                      correlation_features_21_days],axis=1).copy()

#Adding some extra features



#Cleaning the features dataframe and splitting my data at the training date

features["Date"] = pd.to_datetime(data["Date"])
features = features.dropna()
features_raw = features.copy()
features = features[features["Date"]<= "2019-12-31"]

print(features.head())

numeric_features = features.select_dtypes(include=[np.number]).copy()



#Feature Selection

"""
Function which takes all my features and removes one of each pair of highly correlated features
choosing to keep the feature with the higher coefficient of variation value.
"""

def drop_highly_correlated_keep_high_var(df, threshold=0.9):

    corr_matrix = df.corr().abs()  # absolute correlation
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # upper triangle
    
    to_drop = set()
    
    for col in upper.columns:
        # Find highly correlated features
        high_corr = [row for row in upper.index if upper.loc[row, col] > threshold]
        for row in high_corr:
            if row not in to_drop and col not in to_drop:
                # Keep the one with higher variance
                if abs((df[row].var()/df[row].mean())) >= abs((df[col].var()/df[col].mean())):
                    to_drop.add(col)
                else:
                    to_drop.add(row)
    
    print(f"Dropping {len(to_drop)} columns due to high correlation: {list(to_drop)}")
    return df.drop(columns=list(to_drop))

numeric_features = drop_highly_correlated_keep_high_var(numeric_features, threshold=0.9)

"""
Function to remove features with very little variation relative to mean as these are unlikely to
offer much value during the clustering process.
"""

def select_high_cv_features(df, cv_threshold=0.15, exclude_cols=None):

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    cv = df[numeric_cols].std() / df[numeric_cols].mean().abs()  # abs to avoid negative mean
    selected = cv[cv > cv_threshold].index.tolist()
    cv.to_csv("cv.csv")
    
    print(f"Selected {len(selected)} features with CV > {cv_threshold}: {selected}")
    return selected

kept_cols = list(select_high_cv_features(numeric_features,0.1))
numeric_features = numeric_features[kept_cols].copy()

"""
K-Means is very sensitive to scale so we scale the features to have mean 0 and variance 1 in order to ensure all
features have the same weighting in the clustering process.
"""

scaler = StandardScaler()
scaled_numeric_features = pd.DataFrame(
    scaler.fit_transform(numeric_features),
    columns=numeric_features.columns,
    index=numeric_features.index
)

"""
We apply PCA to significantly reduce the number of dimensions to limit effects of the curse of dimensionality.
Also will allow us to obtain more valuable insights as to which features drive cluster separation.
"""

pca = PCA(random_state=0)
pca.fit(scaled_numeric_features)

explained_variance = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)    
wanted_variance = 0.5
n_components = np.argmax(cumulative_explained_variance > wanted_variance) + 1
pca = PCA(n_components=n_components)

pca_data = pd.DataFrame(pca.fit_transform(scaled_numeric_features),columns=[f"PC{i+1}" for i in range(0,n_components)],index=scaled_numeric_features.index)
loadings = pd.DataFrame(pca.components_.T,columns=pca_data.columns,index=scaled_numeric_features.columns)

print(loadings)

optimal_k = 3

kmeans = BisectingKMeans(n_clusters=optimal_k,max_iter=1000,init="k-means++",n_init=50,random_state=0,bisecting_strategy="largest_cluster")
clusters = pd.Series(kmeans.fit_predict(pca_data),index=features.index)
cluster_df = pd.DataFrame({
    "Date": features["Date"],
    "cluster": clusters,
    "smoothed_cluster": clusters.rolling(14).apply(lambda x: x.mode()[0])
})


cluster_assigner = KNeighborsClassifier(n_neighbors=1)
cluster_assigner.fit(pca_data,cluster_df["cluster"])

pca_data["Date"] = features["Date"]


#Saving things for use in the other files

"""

cluster_df.to_csv("clusters.csv",index=False)
numeric_features.to_csv("numeric_features.csv",index=False)
features_raw.to_csv("raw_features.csv",index=False)
pca_data.to_csv("pca_data.csv")
data.to_csv("data.csv")
loadings.to_csv("loadings.csv")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(pca,"pca_model.pkl")
joblib.dump(kmeans,"kmeans_model.pkl")
joblib.dump(cluster_assigner,"knn_model.pkl")

"""





