import sys
sys.path.append('../')
import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from itertools import product
from utils.data_preprocessing import Dataset

class BIRCH_Clustering:

    def __init__(self, X, param_grid):
        print("---Running BIRCH Clustering---")
        self.X = X
        self.param_grid = param_grid

    def grid_search(self):
        best_score = -1
        best_params = {}
        best_num_clusters = 0

        # Perform grid search
        for params in product(*self.param_grid.values()):
            param_dict = dict(zip(self.param_grid.keys(), params))
            birch = Birch(threshold=param_dict['threshold'], branching_factor=param_dict['branching_factor'])
            cluster_labels = birch.fit_predict(self.X)

            # Ignore single clusters (noise)
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(self.X, cluster_labels)

                # Update best parameters if silhouette score is higher
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_params = param_dict
                    best_num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        return best_params, best_score, best_num_clusters

    def birch_clustering(self, best_params):
        birch_model = Birch(**best_params)
        cluster_labels = birch_model.fit_predict(self.X)

        # Extract clusters
        unique_labels = np.unique(cluster_labels)
        clusters = [self.X[cluster_labels == label] for label in unique_labels if label != -1]  # Ignore noise points (-1 label)

        # Compute centroids for valid clusters
        centroids = [np.mean(cluster, axis=0) for cluster in clusters]

        return centroids, clusters

if __name__ == "__main__":
    # Load your dataset and perform preprocessing if needed

    data_path = '../data/raw/vowel.arff'  # Change the path to your ARFF file
    dataset = Dataset(data_path)
    X = dataset.processed_data.drop(columns=['y_true']).values  # Use processed_data from the Dataset object

    # Define the parameter grid for grid search
    param_grid = {
        'threshold': [0.2, 0.5, 1, 1.5, 2],
        'branching_factor': [3, 5, 7, 10, 20, 30, 40]
    }

    # Instantiate BIRCH_Clustering class
    birch_clustering = BIRCH_Clustering(X, param_grid)

    # Perform grid search
    best_params, best_score, best_num_clusters = birch_clustering.grid_search()
    print("Best Parameters:", best_params)
    print("Best Silhouette Score:", best_score)
    print("Num of clusters (excluding noise):", best_num_clusters)

    # Perform clustering using the best parameters
    centroids, clusters = birch_clustering.birch_clustering(best_params)
