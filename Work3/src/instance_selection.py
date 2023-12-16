from collections import Counter

import numpy as np
import pandas as pd
from KIBL import KIBL
from utils.data_preprocessing import Dataset
import time
from sklearn.metrics import euclidean_distances

class InstanceSelection():
    def __init__(self,
                 data: pd.DataFrame,
                 k_neighbors: int):
        self.data = data
        self.k_neighbors = k_neighbors

    def mcnn_algorithm(self):
        data = self.data
        features = data.loc[:, data.columns != 'y_true']
        labels = data.loc[:,'y_true']

        # Step 1: Initialize with one prototype from each class
        unique_classes = np.unique(labels)
        prototypes = pd.DataFrame(columns=data.columns)

        for class_label in unique_classes:
            class_instances = features[labels == class_label]
            centroid = self.compute_centroid(class_instances)
            closest_index = class_instances.iloc[[self.find_closest_instance(centroid, class_instances)]].index
            prototypes = pd.concat([prototypes, data.loc[closest_index]], ignore_index = True)

        # prototypes = pd.read_csv('pen-based_FinalProts.csv', index_col=0)
        i = 0
        # Step 2. Interative refinement until all instances are correctly classified
        while True:
            # Train a k-nearest neighbors classifier with the current prototypes
            classifier = KIBL(X=prototypes, K=self.k_neighbors)

            # Predict using the current prototypes
            classifier.kIBLAlgorithm(data)
            predictions = classifier.predictions

            # Check for misclassifications
            misclassified_instances = data.loc[predictions != data['y_true']]
            misclassified_features = misclassified_instances.loc[:, data.columns != 'y_true']
            misclassified_labels = misclassified_instances.loc[:, 'y_true']
            
            if len(misclassified_instances) == 0:
                break # All instances are correctly classified
            
            # Add representative instances for the current class for class_label in unique classes:
            for class_label in unique_classes:
                # Get misclassified instances for the current class
                class_misclassified_instances = misclassified_features[misclassified_labels == class_label]

                if len(class_misclassified_instances) > 0:
                    centroid = self. compute_centroid(class_misclassified_instances)
                    closest_index = class_misclassified_instances.iloc[[self.find_closest_instance(centroid, class_misclassified_instances)]].index
                    prototypes = pd.concat([prototypes, misclassified_instances.loc[closest_index]], ignore_index = True)
            i += 1
            print(f'Iteration: {i+1} - Misclassified Instances {len(misclassified_instances)}')

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

    def edited_nearest_neighbors(self):
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values

        kibl_instance = KIBL(X=self.data, K=self.k_neighbors)

        # Step 1: Train a K-IBL model
        kibl_instance.kIBLAlgorithm(self.data)

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
        

    def compute_centroid(self, X):
        # Calculate the centroid of a set of instances
        return X.mean()

    def find_closest_instance(self, centroid, instances):
        # Find the instance closest to the centroid
        distances = np.linalg.norm(instances - centroid, axis=1)
        closest_index = np.argmin(distances)
        return closest_index




DATASET_NAME = "pen-based"
TRAIN_DATASETS_PATH = []
for fold in range(0, 10):
    TRAIN_DATASETS_PATH.append(f'../data/folded/{DATASET_NAME}/{DATASET_NAME}.fold.00000{fold}.train.arff')

print(TRAIN_DATASETS_PATH)

print(f"Dataset {DATASET_NAME}")
for i, fold in enumerate(TRAIN_DATASETS_PATH):
    print(f"------------fold{i}------------------")
    data = Dataset(fold)
    train = data.processed_data
    print(train)

    start = time.time()
    instance_selection = InstanceSelection(data=train, k_neighbors=3)
    x_resampled, y_resampled = instance_selection.edited_nearest_neighbors()
    data_resampled = pd.concat([x_resampled, y_resampled], axis=1)
    data_resampled.to_csv(f"../data/resampled-enn2/{DATASET_NAME}/fold{i}.csv")
    print(data_resampled)

    end = time.time()

    print(f"execution time: {(end-start)/60} min")


