import numpy as np
import pandas as pd
from sklearn import metrics
from prettytable import PrettyTable


def map_clusters_to_labels(labels, true_labels):
    """Map clusters to most common true labels."""
    cluster_to_label_mapping = {}
    for cluster in np.unique(labels):
        true_classes_in_cluster = true_labels[labels == cluster]
        most_common = np.bincount(true_classes_in_cluster).argmax()
        cluster_to_label_mapping[cluster] = most_common

    predicted_classes = np.array([cluster_to_label_mapping[cluster] for cluster in labels])
    return predicted_classes

compute_accuracy = lambda predictions, true_labels: np.mean(predictions == true_labels)


def performance_eval(X, predictions, true_labels, verbose = True):
    predictions = map_clusters_to_labels(predictions, true_labels)

    table = PrettyTable(['Metric', 'Result'])
    results = {'accuracy': compute_accuracy(predictions, true_labels),
               'homogeneity': metrics.homogeneity_score(true_labels, predictions),
               'completeness': metrics.completeness_score(true_labels, predictions),
               'v_measure': metrics.v_measure_score(true_labels, predictions),
               'adjusted_rand': metrics.adjusted_rand_score(true_labels, predictions),
               'adjusted_mutual_info': metrics.adjusted_mutual_info_score(true_labels, predictions),
               'silhouette': metrics.silhouette_score(X, predictions) if len(np.unique(predictions))>1 else np.nan,
               'davies': metrics.davies_bouldin_score(X, predictions) if len(np.unique(predictions))>1 else np.nan}

    table.add_rows([[k,v] for k,v in results.items()])

    if verbose:
        print(table)
    return results

class params_grid_eval:
    def __init__(self,
                 X,
                 y,
                 sort_order,
                 name = None):
        self.X = X
        self.y = y
        self.sort_order = sort_order
        self.name = name
        self.results = {'group':[],
                        'accuracy': [],
                        'homogeneity': [],
                        'completeness': [],
                        'v_measure': [],
                        'adjusted_rand': [],
                        'adjusted_mutual_info': [],
                        'silhouette': [], 
                        'davies':[]}
        self.asc_map = {'accuracy': False,
                        'homogeneity': False,
                        'completeness': False,
                        'v_measure': True,
                        'adjusted_rand': True,
                        'adjusted_mutual_info': False,
                        'silhouette': True,
                        'davies': True}
        self.asc_list = [self.asc_map[m] for m in self.sort_order]

    def add_params_group(self, group, predictions):
        single_results = performance_eval(self.X, predictions, self.y, False)
        self.results['group'].append(group)
        for k,v in single_results.items():
            self.results[k].append(v)

    def process_results(self, verbose=True):
        if verbose:
            print(f'---- Results for the algorithm: {self.name}')
        self.results = pd.DataFrame(self.results)
        # self.results.set_index('group')
        self.results = self.results.sort_values(by=self.sort_order, ascending = self.asc_list)
        self.results = self.results.round(3)
        df_d = self.results.to_dict('split')
        results_dict = {'Group': df_d['columns']}
        for g,v in zip(df_d['index'], df_d['data']):
            results_dict[g] = v

        table = PrettyTable([[*v] for k,v in results_dict.items()][0])
        table.add_rows([[*v] for k,v in results_dict.items()][1:])
        if verbose:
            print(table)
        
        return self.results

