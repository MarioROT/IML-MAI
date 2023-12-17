import numpy as np
import pandas as pd
from KIBL import KIBL
from sklearn.metrics import euclidean_distances
from scipy.stats import norm
from collections import Counter

class MCNN:
    def __init__(self, data: pd.DataFrame, k_neighbors: int):
        self.data = data
        self.k_neighbors = k_neighbors
        self.prototypes = None

    def mcnn_algorithm(self):
        data = self.data
        features = data.loc[:, data.columns != 'y_true']
        labels = data.loc[:,'y_true']

        # Step 1: Initialize with one prototype from each class
        unique_classes = np.unique(labels)
        self.prototypes = pd.DataFrame(columns=data.columns)

        for class_label in unique_classes:
            class_instances = features[labels == class_label]
            centroid = self.compute_centroid(class_instances)
            closest_index = class_instances.iloc[[self.find_closest_instance(centroid, class_instances)]].index
            self.prototypes = pd.concat([self.prototypes, data.loc[closest_index]], ignore_index=True)

        i = 0
        # Step 2. Iterative refinement until all instances are correctly classified
        while True:
            # Train a k-nearest neighbors classifier with the current prototypes
            classifier = KIBL(X=self.prototypes, K=self.k_neighbors)

            # Predict using the current prototypes
            classifier.kIBLAlgorithm(data)
            predictions = classifier.predictions

            # Check for misclassifications
            misclassified_instances = data.loc[predictions != data['y_true']]
            misclassified_features = misclassified_instances.loc[:, data.columns != 'y_true']
            misclassified_labels = misclassified_instances.loc[:, 'y_true']

            if len(misclassified_instances) == 0:
                break  # All instances are correctly classified

            # Add representative instances for the current class for class_label in unique classes:
            for class_label in unique_classes:
                # Get misclassified instances for the current class
                class_misclassified_instances = misclassified_features[misclassified_labels == class_label]

                if len(class_misclassified_instances) > 0:
                    centroid = self.compute_centroid(class_misclassified_instances)
                    closest_index = class_misclassified_instances.iloc[
                        [self.find_closest_instance(centroid, class_misclassified_instances)]].index
                    self.prototypes = pd.concat([self.prototypes, misclassified_instances.loc[closest_index]],
                                                ignore_index=True)
            i += 1
            print(f'Iteration: {i + 1} - Misclassified Instances {len(misclassified_instances)}')

        # Step 3: Deletion Operator
        classifier = KIBL(X=self.prototypes, K=1, store_used_neighbors=True)
        classifier.kIBLAlgorithm(data)

        # Identify prototypes that participate in classification
        used_neighbors, counts = np.unique(classifier.used_neighbors, return_counts=True)

        participating_prototypes = used_neighbors[counts > 1]

        # Filter prototypes to keep only those that participate in classification
        self.prototypes = self.prototypes.loc[participating_prototypes]

    def compute_centroid(self, X):
        # Calculate the centroid of a set of instances
        return X.mean()

    def find_closest_instance(self, centroid, instances):
        # Find the instance closest to the centroid
        distances = np.linalg.norm(instances - centroid, axis=1)
        closest_index = np.argmin(distances)
        return closest_index



class ENN:
    def __init__(self, data, k_neighbors):
        self.data = data
        self.k_neighbors = k_neighbors

    def edited_nearest_neighbors(self):
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values

        kibl_instance = KIBL(X=self.data, K=self.k_neighbors)

        # Step 1: Train a K-IBL model
        # Uncomment the line below if you have a method to train the K-IBL model
        # kibl_instance.kIBLAlgorithm(self.data)

        # Step 2: Identify instances with different predicted class than the majority of their k-nearest neighbors
        to_remove = []

        for i in range(self.data.shape[0]):  # Iterate over instances in the original data
            instance = self.data.iloc[i]
            neighbors = kibl_instance.get_neighbors(self.data, instance)

            # Check if the predicted class is different from the majority class in the neighbors
            neighbors_labels = [row[-1] for row in neighbors]
            majority_class = Counter(neighbors_labels).most_common(1)[0][0]
            predicted_class = kibl_instance.predict(neighbors)
            if predicted_class != majority_class:
                to_remove.append(i)

        # Step 3: Remove instances with different predicted class
        data_resampled = np.delete(X, to_remove, axis=0)
        labels_resampled = np.delete(y, to_remove)

        X_resampled = pd.DataFrame(data_resampled, columns=self.data.columns[:-1])
        y_resampled = pd.Series(labels_resampled, name=self.data.columns[-1])

        return X_resampled, y_resampled

# Assuming you have a KIBL class with necessary methods (get_neighbors, predict, etc.)
class KIBL:
    def __init__(self, X, K):
        # Initialization logic for KIBL class
        pass

    def get_neighbors(self, data, instance):
        # Logic to get k-nearest neighbors for a given instance
        pass

    def predict(self, neighbors):
        # Logic to predict the class based on neighbors
        pass

# Example usage:
# enn_instance = ENN(data=your_data, k_neighbors=your_k_value)
# X_resampled, y_resampled = enn_instance.edited_nearest_neighbors()


class IB3:
    def __init__(self, confidence_accept=0.9, confidence_drop=0.7):
        self.S = []  # Set of stored instances
        self.confidence_accept = confidence_accept
        self.confidence_drop = confidence_drop
        self.class_freq = {}  # Frequency of each class
        self.total_instances = 0

    def fit(self, X, y):
        for instance, label in zip(X, y):
            self.total_instances += 1
            self.class_freq[label] = self.class_freq.get(label, 0) + 1

            nearest_instance, nearest_label = self.find_nearest_acceptable_instance(instance)
            if nearest_label != label:
                self.S.append({'instance': instance, 'label': label, 'correct': 0, 'total': 0})

            for stored_instance in self.S:
                if self.is_at_least_as_close(instance, stored_instance['instance'], nearest_instance):
                    stored_instance['total'] += 1
                    if stored_instance['label'] == label:
                        stored_instance['correct'] += 1

            self.S = [s for s in self.S if self.is_acceptable(s)]

    def find_nearest_acceptable_instance(self, instance):
        if not self.S:
            return None, None

        distances = euclidean_distances([instance], [s['instance'] for s in self.S])[0]
        nearest_index = np.argmin(distances)
        return self.S[nearest_index]['instance'], self.S[nearest_index]['label']

    def is_at_least_as_close(self, instance, candidate, nearest):
        if nearest is None:
            return True
        return np.linalg.norm(instance - candidate) <= np.linalg.norm(instance - nearest)

    def is_acceptable(self, instance):
        n = instance['total']
        if n == 0:
            return False

        p = instance['correct'] / n
        z_accept = norm.ppf(self.confidence_accept)
        z_drop = norm.ppf(self.confidence_drop)

        # Upper and lower bounds for acceptability and dropping
        lower_bound_accept = self.confidence_interval(p, n, z_accept)[0]
        upper_bound_drop = self.confidence_interval(p, n, z_drop)[1]

        class_freq = self.class_freq[instance['label']] / self.total_instances
        upper_bound_class = self.confidence_interval(class_freq, self.total_instances, z_accept)[1]
        lower_bound_class = self.confidence_interval(class_freq, self.total_instances, z_drop)[0]

        return lower_bound_accept > upper_bound_class or p > upper_bound_drop < lower_bound_class

    def confidence_interval(self, p, n, z):
        # Calculating the confidence interval
        factor = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
        lower = (p + z ** 2 / (2 * n) - factor) / (1 + z ** 2 / n)
        upper = (p + z ** 2 / (2 * n) + factor) / (1 + z ** 2 / n)
        return lower, upper

    def remove_low_confidence_instances(self):
        """Remove instances with low confidence from the stored instances."""
        self.S = [instance for instance in self.S if self.is_acceptable(instance)]


def preprocess_with_IB3(X, y):
    ib3 = IB3()
    ib3.fit(X, y)
    X_refined, y_refined = zip(*[(s['instance'], s['label']) for s in ib3.S])
    return np.array(X_refined), np.array(y_refined)
