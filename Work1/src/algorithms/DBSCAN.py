import os
import sys
sys.path.append('../')
import numpy as np
from pathlib import Path
from utils.data_preprocessing import Dataset
from sklearn.metrics import silhouette_score, pairwise_distances
from itertools import product
from sklearn.cluster import DBSCAN

class DBSCAN_Clustering:

    def __init__(self, X, param_grid):
        print("---Running DBSCAN---")
        self.X = X
        self.param_grid = param_grid

    def grid_search(self):
        best_score = -1
        best_params = {}
        best_num_clusters = 0

        # Perform grid search
        for params in product(*self.param_grid.values()):
            param_dict = dict(zip(self.param_grid.keys(), params))
            dbscan = DBSCAN(**param_dict)
            distance_matrix = pairwise_distances(self.X, metric=param_dict['metric'])
            labels = dbscan.fit_predict(distance_matrix)

            # Ignore single clusters (noise)
            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(self.X, labels)
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                # Update best parameters if silhouette score is higher
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_params = param_dict
                    best_num_clusters = num_clusters

        return best_params, best_score, best_num_clusters

    def dbscan_clustering(self, best_params):
        dbscan_model = DBSCAN(**best_params)
        distance_matrix = pairwise_distances(self.X, metric=best_params['metric'])
        labels = dbscan_model.fit_predict(distance_matrix)

        # Extract clusters
        unique_labels = np.unique(labels)
        clusters = [self.X[labels == label] for label in unique_labels if label != -1]  # Ignore noise points (-1 label)

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
        'eps': [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples': [ 5, 10, 15, 20, 25, 30, 35, 40],
        'metric': ['euclidean', 'manhattan', 'cosine']
    }

    # Instantiate DBSCAN_Clustering class
    dbscan_clustering = DBSCAN_Clustering(X, param_grid)

    # Perform grid search
    best_params, best_score, best_num_clusters = dbscan_clustering.grid_search()
    print("Best Parameters:", best_params)
    print("Best Silhouette Score:", best_score)
    print("Num of clusters (excluding noise):", best_num_clusters)

    # Perform clustering using the best parameters
    centroids, clusters = dbscan_clustering.dbscan_clustering(best_params)
