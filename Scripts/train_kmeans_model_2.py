from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# The same process of clustering, but using the original data and not the AVG data.
def train_model(data_path, model_path, elbow_method_path, clustered_data_path, clusters_plot_path):
    data = pd.read_csv(data_path)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    print(scaled_data)

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
    plt.show()
    plt.savefig(elbow_method_path)

    optimal_clusters = 6 # Assuming the optimal number of clusters is determined from the Elbow Method (e.g., 6 clusters)
    kmeans = KMeans(n_clusters = optimal_clusters, random_state = 42) # Apply K-means clustering with the optimal number of clusters
    kmeans.fit(scaled_data)

    # save model for 9 clusters
    with open(model_path, 'wb') as file:
        pickle.dump(kmeans, file)

    data['Cluster_kmeans'] = kmeans.labels_
    data.to_csv(clustered_data_path, index = False)

    # Analyze the clusters
    cluster_summary = data.groupby('Cluster_kmeans').mean()
    print("\nCluster Summary:")
    print(cluster_summary)

    pca = PCA(n_components = 17)
    pca_data = pca.fit_transform(data)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c = kmeans.labels_) # plot first two components relationship
    plt.title('Clustering of Student Performance')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    plt.savefig(clusters_plot_path)

if __name__ == "__main__":
    train_model("../Data/processed_data.csv", "../Models/linear_regression_model_2.pkl", "../Results/Plots/elbow_method_2.png", "../Results/clustered_data_2.csv", "../Results/Plots/clusters_2.png")
# Process works better with avg data.