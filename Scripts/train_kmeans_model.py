from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def train_model(data_path, model_path, elbow_method_path, clustered_data_path):
    data = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    print(scaled_data.head())

    inertia = []
    range_clusters = range(1, 11)
    for k in range_clusters:
        kmeans = KMeans(n_clusters = k, random_state = 42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    print("Inertia values for different number of clusters:")
    for k, val in zip(range_clusters, inertia):
        print(f'Clusters: {k}, Inertia: {val}')

    # Plot the elbow method:
    plt.plot(range_clusters, inertia, marker = 'o')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid()
    plt.savefig(elbow_method_path)
    plt.show()

    optimal_clusters = 4  # Assuming the optimal number of clusters is determined from the Elbow Method (e.g., 4 clusters)
    kmeans = KMeans(n_clusters = optimal_clusters, random_state = 42) # Apply K-means clustering with the optimal number of clusters
    kmeans.fit(scaled_data)
    data['Cluster_kmeans'] = kmeans.labels_
    data.to_csv(data, clustered_data_path)

    # Analyze the clusters
