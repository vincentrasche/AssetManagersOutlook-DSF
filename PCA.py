import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# -------------------------------
# Load and Prepare Data
# -------------------------------
# Load the detailed report data CSV.
df = pd.read_csv("report_data.csv")

# Use the grouped asset counts if available; otherwise use asset_counts.
if 'grouped_asset_counts' in df.columns:
    asset_col = 'grouped_asset_counts'
elif 'asset_counts' in df.columns:
    asset_col = 'asset_counts'
    print("Using 'asset_counts' column since 'grouped_asset_counts' was not found.")
else:
    raise KeyError("Neither 'grouped_asset_counts' nor 'asset_counts' found in the CSV file.")

# Convert the asset counts column from string to dictionary.
df[asset_col] = df[asset_col].apply(ast.literal_eval)

# Expand the asset counts dictionary into separate columns.
asset_df = df[asset_col].apply(pd.Series)

# Concatenate these new columns with the original DataFrame.
df_expanded = pd.concat([df, asset_df], axis=1)

# -------------------------------
# Aggregate Data by Firm (Ignoring Time)
# -------------------------------
# Sum the asset counts across all reports for each firm.
firm_assets = df_expanded.groupby("firm")[asset_df.columns].sum()

# -------------------------------
# Standardization and PCA
# -------------------------------
# Standardize the firm-level asset counts.
scaler = StandardScaler()
firm_assets_scaled = scaler.fit_transform(firm_assets)

# Apply PCA to reduce data to two dimensions.
pca = PCA(n_components=2)
pca_components = pca.fit_transform(firm_assets_scaled)

# Create a DataFrame for the PCA results.
df_pca = pd.DataFrame(pca_components, index=firm_assets.index, columns=["PC1", "PC2"]).reset_index()

# Print explained variance ratios.
print("Explained Variance Ratio (PC1, PC2):", pca.explained_variance_ratio_)

# Create a DataFrame for the loadings (components)
# pca.components_ is a (n_components, n_features) array.
loadings = pd.DataFrame(pca.components_.T,
                        index=firm_assets.columns,
                        columns=["PC1", "PC2"])
print("\nPrincipal Component Loadings:")
print(loadings)

# -------------------------------
# Clustering (K-Means)
# -------------------------------
# We choose a number of clusters, here for example 3.
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_pca[["PC1", "PC2"]])
df_pca["cluster"] = clusters

# Print out cluster centroids (in the PCA space)
print("\nK-Means Cluster Centroids (in PCA space):")
print(kmeans.cluster_centers_)

# -------------------------------
# Plot the PCA Projection with Clusters
# -------------------------------
plt.figure(figsize=(10, 8))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for cluster_id in range(k):
    subset = df_pca[df_pca["cluster"] == cluster_id]
    plt.scatter(subset["PC1"], subset["PC2"], color=colors[cluster_id % len(colors)], label=f"Cluster {cluster_id}")
    for idx, row in subset.iterrows():
        plt.text(row["PC1"], row["PC2"], row["firm"], fontsize=8, ha='right')

plt.title("PCA Grouping of Firms Based on Overall Asset Coverage")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# -------------------------------
# Descriptive Summary of Principal Components and Clusters
# -------------------------------
print("\nSummary:")
print("Principal Component 1 explains {:.1f}% of the variance and is characterized by the following loadings:".format(
    pca.explained_variance_ratio_[0]*100))
print(loadings["PC1"].sort_values(ascending=False))
print("\nPrincipal Component 2 explains {:.1f}% of the variance and is characterized by the following loadings:".format(
    pca.explained_variance_ratio_[1]*100))
print(loadings["PC2"].sort_values(ascending=False))

# Optionally, print the number of firms per cluster.
cluster_counts = df_pca["cluster"].value_counts().sort_index()
print("\nNumber of firms per cluster:")
print(cluster_counts)
