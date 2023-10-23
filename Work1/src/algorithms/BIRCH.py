from itertools import product

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
        self.best_params = {
            'homogeneity': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'completeness': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'v_measure': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'adjusted_rand': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'adjusted_mutual_info': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None},
            'silhouette': {'score': -1, 'threshold': None, 'branching_factor': None, 'num_clusters': None}
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
            print(f"Best Parameters for {metric_name.capitalize()}:")
            print(f"Threshold: {params['threshold']}, Branching Factor: {params['branching_factor']}")
            print(f"Best {metric_name.capitalize()} Score: {params['score']:.3f}")
            print(f"Number of Clusters: {params['num_clusters']}")
            print("------")

if __name__ == "__main__":
    # Load Dataset:
    data_path = '../../data/raw/vowel.arff'  # Change the path to your ARFF file
    dataset = Dataset(data_path)
    X = dataset.processed_data.drop(columns=['y_true']).values  # Use processed_data from the Dataset object
    y = dataset.y_true

    BIRCHClustering = BIRCHClustering(X, y)
    BIRCHClustering.search_best_params()
    BIRCHClustering.print_best_params()
