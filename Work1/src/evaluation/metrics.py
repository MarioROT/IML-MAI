import numpy as np
from prettytable import PrettyTable

def compute_accuracy(labels, true_labels):
    """Compute accuracy of predicted labels."""
    cluster_to_label_mapping = {}
    for cluster in np.unique(labels):
        true_classes_in_cluster = true_labels[labels == cluster]
        most_common = np.bincount(true_classes_in_cluster).argmax()
        cluster_to_label_mapping[cluster] = most_common

    predicted_classes = np.array([cluster_to_label_mapping[cluster] for cluster in labels])
    accuracy = np.mean(predicted_classes == true_labels)
    return accuracy

def performance_eval(predictions, true_labels):
    table = PrettyTable(['Metric', 'Result'])
    results = {}
    results ['Accuracy'] = compute_accuracy(predictions, true_labels)
    table.add_rows([[k,v] for k,v in results.items()])
    
    print(table)
    return results

