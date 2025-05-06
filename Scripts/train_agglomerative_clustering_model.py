from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import pandas as pd
import pickle

def train_model(data_path, model_path, clustered_data_path, dendrogram_path, clusters_plot_path):
    data = pd.read_csv(data_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    print(scaled_data)

    dendrogram = sch.dendrogram(sch.linkage(scaled_data, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Students')
    plt.ylabel('Euclidean distances')
    plt.savefig(dendrogram_path)
    plt.show()

    # Fit Hierarchical Clustering model
    hc = AgglomerativeClustering(n_clusters = 4, metric = 'euclidean', linkage = 'ward')
    hc.fit(scaled_data)

    # save model for 4 clusters
    with open(model_path, 'wb') as file:
        pickle.dump(hc, file)

    data['Cluster_hc'] = hc.labels_
    data.to_csv(clustered_data_path, index = False)

    # Analyze the clusters
    cluster_summary = data.groupby('Cluster_hc').mean()
    print("\nCluster Summary:")
    print(cluster_summary)

    pca = PCA(n_components = 4)
    pca_data = pca.fit_transform(data)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c = hc.labels_)
    plt.title('Clustering of Student Performance')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()
    plt.savefig(clusters_plot_path)

if __name__ == "__main__":
    train_model("../Data/avg_data.csv", "../Models/agglomerative_clustering_model.pkl", "../Results/hc_clustered_data.csv", "../Results/Plots/dendrogram.png", "../Results/Plots/hc_clusters.png")
