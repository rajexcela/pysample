import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
# Step 1: Data Collection
# Generate sample data (Replace with your actual data)
data = np.random.rand(100, 5)  # 100 samples, 5 features
# Step 2: Dimensionality Reduction using PCA
pca = PCA(n_components=2)  # Reduce to 2 components
reduced_data = pca.fit_transform(data)

# Step 3: Clustering using K-Medoids
k = 3  # Number of clusters
kmedoids = KMedoids(n_clusters=k, random_state=0)
kmedoids.fit(reduced_data)
# Step 4: Clustering using K-Means
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(reduced_data)
# Step 5: Load Balancing
# Assign cluster labels to data points
kmedoids_labels = kmedoids.labels_
kmeans_labels = kmeans.labels_
# Perform load balancing based on the cluster assignments
# Step 6: Evaluation
# Calculate Silhouette scores to evaluate clustering quality
silhouette_kmedoids = silhouette_score(reduced_data, kmedoids_labels)
silhouette_kmeans = silhouette_score(reduced_data, kmeans_labels)

# Print the Silhouette scores
print("Silhouette Score - K-Medoids:", silhouette_kmedoids)
print("Silhouette Score - K-Means:", silhouette_kmeans)


# Plot the reduced data points with cluster labels
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmedoids_labels)
plt.title('Clustering Result - K-Medoids')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
# Plot the reduced data points with cluster labels
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels)
plt.title('Clustering Result - K-Means')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

plt.show()

