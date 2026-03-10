# ==========================================
# CUSTOMER SEGMENTATION ML PROJECT (ADVANCED)
# ==========================================

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import scipy.cluster.hierarchy as sch

print("\nLibraries Imported Successfully!")

# -------------------------------
# 2. LOAD DATASET
# -------------------------------
print("\nLoading Dataset...")

data = pd.read_csv("retail_customer_segmentation.csv")

df = pd.DataFrame(data)

print("Dataset Loaded Successfully!")

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows")
print(df.head())

# -------------------------------
# 3. DATASET INFORMATION
# -------------------------------
print("\nDataset Information")
print(df.info())

print("\nColumn Names")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------------
# 4. HANDLE MISSING VALUES
# -------------------------------
df = df.dropna()

print("\nMissing Values Removed")
print("New Shape:", df.shape)

# -------------------------------
# 5. FEATURE SELECTION
# -------------------------------
features = ['annual_income', 'purchase_frequency', 'months_active']

X = df[features].values

print("\nSelected Features")
print(df[features].head())

# -------------------------------
# 6. FEATURE DISTRIBUTION
# -------------------------------
print("\nGenerating Feature Distributions")

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
sns.histplot(df['annual_income'], kde=True)
plt.title("Annual Income")

plt.subplot(1,3,2)
sns.histplot(df['purchase_frequency'], kde=True)
plt.title("Purchase Frequency")

plt.subplot(1,3,3)
sns.histplot(df['months_active'], kde=True)
plt.title("Months Active")

plt.tight_layout()
plt.show()

# -------------------------------
# 7. CORRELATION HEATMAP
# -------------------------------
plt.figure(figsize=(8,6))

sns.heatmap(
    df.select_dtypes(include=[np.number]).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.show()

# -------------------------------
# 8. FEATURE SCALING
# -------------------------------
print("\nScaling Features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaling Completed")

# -------------------------------
# 9. ELBOW METHOD
# -------------------------------
print("\nRunning Elbow Method")

inertia = []

K_range = range(1,10)

for k in K_range:

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(K_range, inertia, marker='o')

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.show()

# -------------------------------
# 10. FIND OPTIMAL CLUSTERS
# -------------------------------
print("\nFinding Best Cluster Number")

best_k = 2
best_score = -1

for k in range(2,10):

    model = KMeans(n_clusters=k, random_state=42, n_init=10)

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print("K =",k," Silhouette Score =",score)

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest Cluster Number:", best_k)

# -------------------------------
# 11. KMEANS CLUSTERING
# -------------------------------
print("\nRunning KMeans Clustering")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

kmeans_score = silhouette_score(X_scaled, df['KMeans_Cluster'])

print("KMeans Silhouette Score:", kmeans_score)

# -------------------------------
# 12. KMEANS VISUALIZATION
# -------------------------------
plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df['annual_income'],
    y=df['purchase_frequency'],
    hue=df['KMeans_Cluster'],
    palette='viridis'
)

plt.title("KMeans Customer Segmentation")

plt.show()

# -------------------------------
# 13. HIERARCHICAL DENDROGRAM
# -------------------------------
print("\nGenerating Dendrogram")

plt.figure(figsize=(10,5))

sch.dendrogram(
    sch.linkage(
        X_scaled,
        method='ward'
    )
)

plt.title("Hierarchical Clustering Dendrogram")

plt.show()

# -------------------------------
# 14. HIERARCHICAL CLUSTERING
# -------------------------------
print("\nRunning Hierarchical Clustering")

hc = AgglomerativeClustering(
    n_clusters=best_k,
    linkage='ward'
)

df['Hierarchical_Cluster'] = hc.fit_predict(X_scaled)

hc_score = silhouette_score(
    X_scaled,
    df['Hierarchical_Cluster']
)

print("Hierarchical Silhouette Score:", hc_score)

# -------------------------------
# 15. HIERARCHICAL VISUALIZATION
# -------------------------------
plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df['annual_income'],
    y=df['purchase_frequency'],
    hue=df['Hierarchical_Cluster'],
    palette='magma'
)

plt.title("Hierarchical Clustering")

plt.show()

# -------------------------------
# 16. DBSCAN CLUSTERING
# -------------------------------
print("\nRunning DBSCAN")

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

print("DBSCAN Clusters:", np.unique(df['DBSCAN_Cluster']))

noise_points = np.sum(df['DBSCAN_Cluster'] == -1)

print("Noise Points Detected:", noise_points)

# -------------------------------
# 17. DBSCAN VISUALIZATION
# -------------------------------
plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df['annual_income'],
    y=df['purchase_frequency'],
    hue=df['DBSCAN_Cluster'],
    palette='tab10'
)

plt.title("DBSCAN Clustering")

plt.show()

# -------------------------------
# 18. PCA VISUALIZATION
# -------------------------------
print("\nApplying PCA")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df['PCA1'],
    y=df['PCA2'],
    hue=df['KMeans_Cluster'],
    palette='viridis'
)

plt.title("Customer Segmentation using PCA")

plt.show()

# -------------------------------
# 19. CLUSTER SUMMARY
# -------------------------------
cluster_summary = df.groupby(
    'KMeans_Cluster'
)[features].mean()

print("\nCluster Summary")
print(cluster_summary)

# -------------------------------
# 20. CLUSTER INTERPRETATION
# -------------------------------
print("\nCluster Interpretation")

for i in cluster_summary.index:

    income = cluster_summary.loc[i,'annual_income']
    freq = cluster_summary.loc[i,'purchase_frequency']

    if income > df['annual_income'].mean():
        income_level = "High Income"
    else:
        income_level = "Low Income"

    if freq > df['purchase_frequency'].mean():
        purchase_level = "Frequent Buyers"
    else:
        purchase_level = "Rare Buyers"

    print(f"Cluster {i} → {income_level}, {purchase_level}")

# -------------------------------
# 21. PAIRPLOT VISUALIZATION
# -------------------------------
print("\nGenerating Pairplot")

sns.pairplot(
    df,
    vars=features,
    hue="KMeans_Cluster",
    palette="viridis"
)

plt.show()

# -------------------------------
# 22. SAVE MODEL
# -------------------------------
joblib.dump(
    kmeans,
    "kmeans_customer_segmentation_model.pkl"
)

print("\nModel Saved Successfully!")

# -------------------------------
# 23. SAVE RESULTS
# -------------------------------
df.to_csv(
    "customer_segmentation_results.csv",
    index=False
)

print("Results Saved Successfully!")

# -------------------------------
# 24. COMPLETION MESSAGE
# -------------------------------
print("\n===================================")
print("Customer Segmentation Project Completed")
print("===================================")