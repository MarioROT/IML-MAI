import time
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from algorithms.feature_selection import FeatureSelection
# from algorithms.instance_selection import InstanceSelection
from sklearn.metrics import euclidean_distances
from sklearn.metrics import euclidean_distances
from scipy.stats import norm
from datetime import datetime
from collections import Counter


class KIBL:
    def __init__(self,
                 X=None,
                 K=3,
                 voting='MP',
                 retention='NR',
                 feature_selection='ones',
                 k_fs='nonzero',
                 instance_selection='None',
                 normalize=False,
                 store_used_neighbors=False,
                 save=False):

        print("-------Performing K-IBL-------")
        self.X = X
        self.K = K
        self.voting = voting  # MP: Modified Plurality , BC: Borda Count
        self.retention = retention  # NR: Never Retains, . Always retain (AR) Different Class retention (DF). Degree of Disagreement (DD).
        self.weights_m = feature_selection
        self.k_weights = k_fs
        self.instance_selection = instance_selection
        self.normalize = normalize
        self.store_used_neighbors = store_used_neighbors
        self.save = save

    # Normalizing the train data
    def normalize_fun(self, X):
        data_normalized = pd.DataFrame(X)
        for column in data_normalized.columns:
            if is_numeric_dtype(data_normalized[column]):
                values = data_normalized[column].values
                min_v = values.min()
                max_v = values.max()

                for i in range(len(data_normalized[column])):
                    value = data_normalized.loc[i, column]
                    new_v = (value - min_v) / (max_v - min_v)
                    data_normalized.loc[i, column] = new_v

            return data_normalized

    # Calculating the euc distance between two instances
    def euc_distance(self, test_row, train_row):
        dist = np.sum(self.weights * ((test_row - train_row) ** 2))
        return np.sqrt(dist)

    # Getting the K nearest neighbors with all the data from the learning base
    def get_neighbors(self, train, test_row):
        distances = []
        for i, train_row in enumerate(train.values):
            x_train = train_row[:-1]
            dist = self.euc_distance(test_row[:-1], x_train)
            distances.append((train_row, dist, i))
        distances.sort(key=lambda tup: tup[1])
        neighbors = np.array(distances[:self.K], dtype=object)[:, 0]
        if self.store_used_neighbors:
            self.used_neighbors.append(np.array(distances[:self.K], dtype=object)[:, 2])
        return neighbors

    # Predict the type according to the majority of neighbors
    def predict(self, neighbors):
        neighbors_labels = [row[-1] for row in neighbors]
        neighbour_class, counts = np.unique(neighbors_labels, return_counts=True)
        length = len(neighbors_labels)

        if self.voting == 'MP':
            if np.count_nonzero(counts == counts.max()) == 1:
                prediction = max(set(neighbors_labels), key=neighbors_labels.count)
            else:
                while np.count_nonzero(counts == counts.max()) > 1:
                    neighbors_labels = neighbors_labels[:-1]
                    neighbour_class, counts = np.unique(neighbors_labels, return_counts=True)
                prediction = max(set(neighbors_labels), key=neighbors_labels.count)
            return prediction

        elif self.voting == 'BC':
            points = np.arange(length - 1, -1, -1, dtype=int)
            values_array = np.array(neighbors_labels)
            scores = {}

            for label in set(values_array):
                labels = np.where(values_array == label, 1, 0)
                score = np.dot(labels, points)
                scores[label] = score

            max_points = max(scores.values())
            max_point_class = [label for label, points in scores.items() if points == max_points]

            if len(max_point_class) == 1:
                return max_point_class[0]
            else:
                return next(label for label in values_array if label in max_point_class)

    def compute_weights(self, data, method):
        features = data.loc[:, data.columns != 'y_true']
        labels = data.loc[:, 'y_true']
        return FeatureSelection(features, labels, method, self.k_weights).compute_weights()

    def evaluate_accuracy(self, predictions, true_labels):
        correct_count = sum(1 for pred, true_label in zip(predictions, true_labels) if pred == true_label)
        return correct_count / len(true_labels)

    def evaluate_efficiency(self, problem_solving_times):
        return np.mean(problem_solving_times)

    def kIBLAlgorithm(self, test_data):
        self.used_neighbors = [] if self.store_used_neighbors else False

        if self.normalize == True:
            data_normalized = self.normalize_fun(np.vstack([self.X, test_data]))
            train_normalized = data_normalized[:self.X.shape[0]]
            test_normalized = data_normalized[-test_data.shape[0]:]
        else:
            train_normalized = self.X
            test_normalized = test_data

        print('----Data normalized----')

        if self.instance_selection in ['MCNN', 'ENN', 'IB3']:
            train_normalized = InstanceSelection(train_normalized, self.K, self.instance_selection,
                                                 self.save).refine_dataset()

        self.weights = self.compute_weights(train_normalized, self.weights_m)

        true_labels = test_normalized.iloc[:, -1]
        predictions = []
        problem_solving_times = []
        start_total_time = time.time()
        for i, instance in enumerate(test_normalized.values):
            if i % 1000 == 0:
                print(f'iteration:{i}')
            start_time = time.time()
            neighbors = self.get_neighbors(train_normalized, instance)
            predict = self.predict(neighbors)
            predictions.append(predict)

            if self.retention == 'NR':
                retained_instance = None
            elif self.retention == 'AR':
                retained_instance = instance
            elif self.retention == 'DF':
                if instance[-1] != predict:
                    retained_instance = instance
                else:
                    retained_instance = None
            elif self.retention == 'DD':
                neighbors_labels = [row[-1] for row in neighbors]
                neighbour_class, counts = np.unique(neighbors_labels, return_counts=True)
                majority_cases = max(counts)
                dd = (self.K - majority_cases) / ((len(neighbour_class) - 1) * majority_cases)

                if dd >= 0.5:
                    retained_instance = instance
                else:
                    retained_instance = None

            if retained_instance is not None:
                self.X = pd.concat([self.X, pd.DataFrame(retained_instance.reshape(1, -1), columns=self.X.columns)],
                                   ignore_index=True)
                train_normalized = pd.concat([train_normalized, pd.DataFrame(retained_instance.reshape(1, -1),
                                                                             columns=train_normalized.columns)],
                                             ignore_index=True)

            end_time = time.time()
            problem_solving_times.append(end_time - start_time)

        self.predictions = predictions
        # self.used_neighbors = np.unique(self.used_neighbors)

        accuracy = self.evaluate_accuracy(predictions, true_labels)
        print('----Accuracy Completed----')

        efficiency = self.evaluate_efficiency(problem_solving_times)
        print('----Efficiency Completed----')
        end_total_time = time.time()
        total_time = end_total_time - start_total_time

        return accuracy, efficiency, total_time


class InstanceSelection():
    def __init__(self,
                 data: pd.DataFrame,
                 k_neighbors: int,
                 method: str,
                 save: str):
        self.data = data
        self.k_neighbors = k_neighbors
        self.met_name = method
        self.methods = {'MCNN': self.modified_condensed_nearest_neighbors,
                        'ENN': self.edited_nearest_neighbors,
                        'IB3': preprocess_with_IB3}
        self.method = self.methods[method]
        self.save = save

    def refine_dataset(self):
        refined_dataset = self.method(self.data, self.k_neighbors)
        if self.save:
            name_date = str(datetime.now()).replace(" ", "_").replace(":", "_")
            refined_dataset.to_csv(self.save + f'RefinedDataset_{self.met_name}_{name_date}.csv', index=False)
        return refined_dataset

    @staticmethod
    def modified_condensed_nearest_neighbors(data, k_neighbors):
        features = data.loc[:, data.columns != 'y_true']
        labels = data.loc[:, 'y_true']

        # Step 1: Initialize with one prototype from each class
        unique_classes = np.unique(labels)
        prototypes = pd.DataFrame(columns=data.columns)

        for class_label in unique_classes:
            class_instances = features[labels == class_label]
            centroid = InstanceSelection.compute_centroid(class_instances)
            closest_index = class_instances.iloc[
                [InstanceSelection.find_closest_instance(centroid, class_instances)]].index
            prototypes = pd.concat([prototypes, data.loc[closest_index]], ignore_index=True)

        # prototypes = pd.read_csv('pen-based_FinalProts.csv', index_col=0)
        i = 0
        # Step 2. Interative refinement until all instances are correctly classified
        while True:
            # Train a k-nearest neighbors classifier with the current prototypes
            classifier = KIBL(X=prototypes, K=k_neighbors)

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
                    centroid = InstanceSelection.compute_centroid(class_misclassified_instances)
                    closest_index = class_misclassified_instances.iloc[
                        [InstanceSelection.find_closest_instance(centroid, class_misclassified_instances)]].index
                    prototypes = pd.concat([prototypes, misclassified_instances.loc[closest_index]], ignore_index=True)
            i += 1
            print(f'Iteration: {i + 1} - Misclassified Instances {len(misclassified_instances)}')

        # prototypes = pd.read_csv('pen-based MCNN.csv')

        # Step 3: Deletion Operator
        classifier = KIBL(X=prototypes, K=1, store_used_neighbors=True)
        classifier.kIBLAlgorithm(data)

        # Identify prototypes that participate in classification
        used_neighbors, counts = np.unique(classifier.used_neighbors, return_counts=True)

        participating_prototypes = used_neighbors[counts > 1]

        # Filter prototypes to keep only those that participate in classificationp.unique(classifier.used_neighbors, return_counts=True)[1]nnp.unique(classifier.used_neighbors, return_counts=True)[1A]
        final_prototypes = prototypes.loc[participating_prototypes]

        return final_prototypes

    @staticmethod
    def edited_nearest_neighbors(data, k_neighbors):
        kibl_instance = KIBL(X=data, K=k_neighbors)

        # Step 1: Train a K-IBL model
        #kibl_instance.kIBLAlgorithm(data)

        # Step 2: Identify instances with different class than the majority of their k-nearest neighbors
        new_train_data = []

        for i in range(data.shape[0]):  # Iterate over instances in the original data
            instance = data.iloc[i]
            neighbors = kibl_instance.get_neighbors(data, instance)
            neighbors_labels = [row[-1] for row in neighbors]

            # Step 3: Keep instances
            if Counter(neighbors_labels).most_common(1)[0][0] == instance[-1]:
                new_train_data.append(instance)

        return pd.DataFrame(new_train_data, columns=data.columns)

    @staticmethod
    def compute_centroid(X):
        # compute the centroid of a set of instances
        return X.mean()

    @staticmethod
    def find_closest_instance(centroid, instances):
        # Find the instance closest to the centroid
        distances = np.linalg.norm(instances - centroid, axis=1)
        closest_index = np.argmin(distances)
        return closest_index


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


def preprocess_with_IB3(data, k):
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    ib3 = IB3()
    ib3.fit(X.values, y.values)
    X_refined, y_refined = zip(*[(s['instance'], s['label']) for s in ib3.S])

    data_refined = pd.DataFrame(np.vstack([np.array(X_refined).T, np.array(y_refined)]).T, columns=data.columns)
    return data_refined

