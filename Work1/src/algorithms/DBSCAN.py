
import os
import sys
import pandas as pd
from matplotlib import pyplot as plt
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
        self.eps_values = [3, 5, 7, 10]
        self.min_samples_values = [3, 10, 15, 20, 25, 30, 35, 40, 45]
        self.similarity_metrics = ['euclidean', 'cosine']
        self.distance_algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']

        self.X = StandardScaler().fit_transform(self.X)

        self.best_params = {
            'homogeneity': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None,
                            'similarity_metric': None, 'distance_algorithm': None},
            'completeness': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None,
                             'similarity_metric': None, 'distance_algorithm': None},
            'v_measure': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None,
                          'similarity_metric': None, 'distance_algorithm': None},
            'adjusted_rand': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None,
                              'similarity_metric': None, 'distance_algorithm': None},
            'adjusted_mutual_info': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None,
                                     'similarity_metric': None, 'distance_algorithm': None},
            'silhouette': {'score': -1, 'eps': None, 'min_samples': None, 'num_clusters': None,
                           'similarity_metric': None, 'distance_algorithm': None}
        }
        self.metric_scores_per_eps = {metric: [] for metric in
                                      self.best_params.keys()}  # Store metric scores for different eps values

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
        for eps, min_samples, similarity_metric, distance_algorithm in product(self.eps_values, self.min_samples_values,
                                                                                self.similarity_metrics, self.distance_algorithms):
            if similarity_metric == 'cosine' and distance_algorithm in ['ball_tree', 'kd_tree']:
                continue
            db = DBSCAN(eps=eps, min_samples=min_samples, metric=similarity_metric, algorithm=distance_algorithm).fit(self.X)
            unique_labels = np.unique(db.labels_)

            if len(unique_labels) == 1:
                continue

            labels = db.labels_
            metrics_scores = self._calculate_metrics(labels)

            for metric_name, score in metrics_scores.items():
                self.metric_scores_per_eps[metric_name].append(score)
                if score > self.best_params[metric_name]['score']:
                    self.best_params[metric_name]['score'] = score
                    self.best_params[metric_name]['eps'] = eps
                    self.best_params[metric_name]['min_samples'] = min_samples
                    self.best_params[metric_name]['num_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
                    self.best_params[metric_name]['similarity_metric'] = similarity_metric
                    self.best_params[metric_name]['distance_algorithm'] = distance_algorithm


    def print_best_params(self):
        print(self.best_params)
        for metric_name, params in self.best_params.items():
            print(f"Best Parameters for {metric_name.capitalize()}:")
            print(f"EPS: {params['eps']}, Min Samples: {params['min_samples']}")
            print(f"Similarity Metric: {params['similarity_metric']}, Distance Algorithm: {params['distance_algorithm']}")
            print(f"Best {metric_name.capitalize()} Score: {params['score']:.3f}")
            print(f"Number of Clusters: {params['num_clusters']}")
            print("------")




if __name__ == "__main__":
    # Load Dataset:
    data_path = '../../data/raw/kr-vs-kp.arff'  # Change the path to your ARFF file
    dataset = Dataset(data_path)
    X = dataset.processed_data.drop(columns=['y_true']).values  # Use processed_data from the Dataset object
    y = dataset.y_true

    DBSCANClustering = DBSCANClustering(X, y)
    DBSCANClustering.search_best_params()
    DBSCANClustering.print_best_params()



