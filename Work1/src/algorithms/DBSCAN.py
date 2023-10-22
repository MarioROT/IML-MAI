
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append('../')
from sklearn import metrics
from sklearn.cluster import DBSCAN
from itertools import product
import numpy as np
from utils.data_preprocessing import Dataset


class DBSCANClustering:
    def __init__(self, X, y):
        print("---Running DBSCAN Clustering---")
        self.X = X
        self.y_true = y
        self.eps_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
        self.min_samples_values = [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

        self.X = StandardScaler().fit_transform(self.X)

        self.best_params = {
            'homogeneity': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None},
            'completeness': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None},
            'v_measure': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None},
            'adjusted_rand': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None},
            'adjusted_mutual_info': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None},
            'silhouette': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None}
        }

    def _calculate_metrics(self, labels):
        metrics_scores = {
            'homogeneity': metrics.homogeneity_score(self.y_true, labels),
            'completeness': metrics.completeness_score(self.y_true, labels),
            'v_measure': metrics.v_measure_score(self.y_true, labels),
            'adjusted_rand': metrics.adjusted_rand_score(self.y_true, labels),
            'adjusted_mutual_info': metrics.adjusted_mutual_info_score(self.y_true, labels),
            'silhouette': metrics.silhouette_score(self.X, labels)
        }
        return metrics_scores

    def search_best_params(self):
        for eps, min_samples in product(self.eps_values, self.min_samples_values):
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(self.X)
            unique_labels = np.unique(db.labels_)

            if len(unique_labels) == 1:
                continue

            labels = db.labels_
            metrics_scores = self._calculate_metrics(labels)

            for metric_name, score in metrics_scores.items():
                if score > self.best_params[metric_name]['score']:
                    self.best_params[metric_name]['score'] = score
                    self.best_params[metric_name]['eps'] = eps
                    self.best_params[metric_name]['min_samples'] = min_samples
                    self.best_params[metric_name]['num_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)

    def print_best_params(self):
        for metric_name, params in self.best_params.items():
            print(f"Best Parameters for {metric_name.capitalize()}:")
            print(f"EPS: {params['eps']}, Min Samples: {params['min_samples']}")
            print(f"Best {metric_name.capitalize()} Score: {params['score']:.3f}")
            print(f"Number of Clusters: {params['num_clusters']}")
            print("------")

if __name__ == "__main__":
    # Load Dataset:
    data_path = '../../data/raw/vowel.arff'  # Change the path to your ARFF file
    dataset = Dataset(data_path)
    X = dataset.processed_data.drop(columns=['y_true']).values  # Use processed_data from the Dataset object
    y = dataset.y_true

    DBSCANClustering = DBSCANClustering(X, y)
    DBSCANClustering.search_best_params()
    DBSCANClustering.print_best_params()

