import numpy as np

def map_clusters_to_labels(labels, true_labels):
    """Map clusters to most common true labels."""
    cluster_to_label_mapping = {}
    for cluster in np.unique(labels):
        true_classes_in_cluster = true_labels[labels == cluster]
        most_common = np.bincount(true_classes_in_cluster).argmax()
        cluster_to_label_mapping[cluster] = most_common

    predicted_classes = np.array([cluster_to_label_mapping[cluster] for cluster in labels])
    accuracy = np.mean(predicted_classes == true_labels)
    return accuracy

