# ==========================================
# CUSTOMER SEGMENTATION PROJECT (CORRECTED)
# ==========================================

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import scipy.cluster.hierarchy as sch

print("Libraries Imported Successfully!")

# -------------------------------
# 2. LOAD DATASET
# -------------------------------
data = pd.read_csv("retail_customer_segmentation.csv")

print("\nData Imported Successfully!")
print("Shape:", data.shape)

df = pd.DataFrame(data)

print("\nFirst 5 Rows:")
print(df.head())

# -------------------------------
# 3. CHECK DATA
# -------------------------------
print("\nDataset Info:")
print(df.info())

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# 4. HANDLE MISSING VALUES
# -------------------------------
df = df.dropna()

print("\nAfter Removing Missing Values:")
print(df.shape)

# -------------------------------
# 5. FEATURE SELECTION
# -------------------------------
features = ['annual_income','purchase_frequency','months_active']

X = df[['annual_income','purchase_frequency','months_active']].values

print("\nSelected Features:")
print (df[features].head())
print(X[:5])

# -------------------------------
# 6. FEATURE SCALING
# -------------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData Scaling Completed")

# -------------------------------
# 7. ELBOW METHOD
# -------------------------------
print("\nRunning Elbow Method...")

inertia = []
K_range = range(1,10)

for k in K_range:
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    kmeans.fit(X_scaled)
    
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(K_range, inertia, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.show()

# -------------------------------
# 8. KMEANS CLUSTERING
# -------------------------------
print("\nRunning KMeans Clustering")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, df['KMeans_Cluster'])

print("Silhouette Score:", score)


#correlation heatmap
plt.figure(figsize=(10,8))

sns.heatmap(df.select_dtypes(include=[np.number]).drop(columns=['customer_id']).corr(),annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# -------------------------------
# 9. KMEANS VISUALIZATION
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
# 11. APPLY HIERARCHICAL
# -------------------------------
hc = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

df['Hierarchical_Cluster'] = hc.fit_predict(X_scaled)

score2 = silhouette_score(
    X_scaled,
    df['Hierarchical_Cluster']
)

print("Hierarchical Silhouette Score:", score2)

# -------------------------------
# 12. HIERARCHICAL VISUALIZATION
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
# 13. DBSCAN CLUSTERING
# -------------------------------
print("\nRunning DBSCAN")

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

print("DBSCAN Clusters:", np.unique(df['DBSCAN_Cluster']))

# -------------------------------
# 14. DBSCAN VISUALIZATION
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
# 15. CLUSTER SUMMARY
# -------------------------------
cluster_summary = df.groupby(
    'KMeans_Cluster'
)[features].mean()

print("\nCluster Summary:")
print(cluster_summary)

# -------------------------------
# 16. SAVE RESULTS
# -------------------------------
df.to_csv(
    "customer_segmentation_results.csv",
    index=False
)

print("\nResults Saved Successfully!")

# -------------------------------
# 17. COMPLETION MESSAGE
# -------------------------------
print("\nCustomer Segmentation Project Completed Successfully!")



# -------------------------------
# 10. HIERARCHICAL CLUSTERING
# -------------------------------
print("\nGenerating Dendrogram")

plt.figure(figsize=(10,5))

sch.dendrogram(
    sch.linkage(
        X_scaled,
        method='ward'
    )
)

plt.title("Dendrogram")

plt.show()