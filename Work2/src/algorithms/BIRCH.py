from itertools import product
import pandas as pd
from sklearn import metrics
from sklearn.cluster import Birch
import sys

sys.path.append('../')
from utils.data_preprocessing import Dataset


class BIRCHClustering:
    def __init__(self, X, y):
        print("---Running BIRCH Clustering---")
        self.X = X
        self.y_true = y
        self.threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.branching_factor_values = [10, 20, 30, 40, 50]
        self.labels_ = None
        self.best_params = {
            'Homogeneity': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'Completeness': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'V_measure': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'Adjusted_rand': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'Adjusted_mutual_info': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'Silhouette': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None}
        }
        self.results_dict = {
            'Homogeneity': {},
            'Completeness': {},
            'V_measure': {},
            'Adjusted_rand': {},
            'Adjusted_mutual_info': {},
            'Silhouette': {}
        }

    def _calculate_metrics(self, labels):
        metrics_scores = {
            'Homogeneity': metrics.homogeneity_score(self.y_true, labels),
            'Completeness': metrics.completeness_score(self.y_true, labels),
            'V_measure': metrics.v_measure_score(self.y_true, labels),
            'Adjusted_rand': metrics.adjusted_rand_score(self.y_true, labels),
            'Adjusted_mutual_info': metrics.adjusted_mutual_info_score(self.y_true, labels),
            'Silhouette': metrics.silhouette_score(self.X, labels)
        }
        return metrics_scores
    
    def fit(self, data, threshold=0.5, branching_factor=50):
        birch = Birch(threshold=threshold, branching_factor=branching_factor)
        birch.fit(data)
        self.labels_ = birch.labels_
        return self.labels_
        
    def predict(self,data):
        return self.labels_
                    
    def search_best_params(self):
        for threshold, branching_factor in product(self.threshold_values, self.branching_factor_values):
            birch = Birch(threshold=threshold, branching_factor=branching_factor)
            birch.fit(self.X)
            labels = birch.labels_
            metrics_scores = self._calculate_metrics(labels)

            for metric_name, score in metrics_scores.items():
                if score > self.best_params[metric_name]['score']:
                    self.best_params[metric_name]['score'] = score
                    self.best_params[metric_name]['threshold'] = threshold
                    self.best_params[metric_name]['branching_factor'] = branching_factor
                    self.best_params[metric_name]['num_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)

    def print_best_params(self):
        for metric_name, params in self.best_params.items():
            threshold = params['threshold']
            branching_factor = params['branching_factor']
            num_clusters = params['num_clusters']
            score = params['score']

            self.results_dict[metric_name] = {
                'Threshold': threshold,
                'Branching Factor': branching_factor,
                'Number of Clusters': num_clusters,
                'Score': round(score, 3)
            }

        print(pd.DataFrame(self.results_dict))


if __name__ == "__main__":
    # Load Dataset:
    data_path = '../../data/raw/iris.arff'  # Change the path to your ARFF file
    dataset = Dataset(data_path, method="numerical")
    X = dataset.processed_data.drop(columns=['y_true']).values  # Use processed_data from the Dataset object
    y = dataset.y_true

    BIRCHClustering = BIRCHClustering(X, y)
    BIRCHClustering.search_best_params()
    BIRCHClustering.print_best_params()
