import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Encode Gender (categorical to numerical)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Female=0, Male=1

# Select features
X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensions to 2 for better clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Use optimal clusters based on the Elbow graph (e.g., 6)
optimal_clusters = 6
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_pca)

# Visualize the clusters
plt.figure(figsize=(10, 8))
for cluster in range(optimal_clusters):
    plt.scatter(
        X_pca[data['Cluster'] == cluster, 0],
        X_pca[data['Cluster'] == cluster, 1],
        label=f"Cluster {cluster}"
    )
plt.title('Customer Segments (PCA Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()

# Save clustered data to a CSV file
data.to_csv("clustered_customers.csv", index=False)
print("Clustered data saved to: clustered_customers_pca.csv")