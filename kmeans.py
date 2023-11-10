import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Titanic dataset
data = pd.read_csv('./dataset/train.csv')  # Replace 'path_to_titanic_dataset.csv' with the actual path

# Perform necessary data preprocessing
# For example, removing NaN values, encoding categorical variables, etc.

# Select relevant features for clustering (numerical features in this case)
# Here, considering 'Age' and 'Fare' columns for clustering
selected_features = data[['Age', 'Fare']]

# Handle missing values if any
selected_features = selected_features.dropna()

# Standardize the features
scaler = StandardScaler()
selected_features_scaled = scaler.fit_transform(selected_features)

# Apply PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(selected_features_scaled)

# Apply K-Means clustering
k = 4  # Considering 4 clusters as an example
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(selected_features_scaled)

# Predict clusters for the reduced data
clusters = kmeans.predict(selected_features_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 6))

# Plot each cluster with different colors
for i in range(k):
    cluster_samples = X_pca[clusters == i]
    plt.scatter(cluster_samples[:, 0], cluster_samples[:, 1], label=f'Cluster {i}')

plt.title('Titanic Dataset Clusters')
plt.legend()
plt.show()
