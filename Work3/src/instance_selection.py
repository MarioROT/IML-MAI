import numpy as np
import pandas as pd
from KIBL import KIBL

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
                    closest_index = self.find_closest_instance(centroid, class_misclassified_instances)
                    prototypes = pd.concat([prototypes, misclassified_instances.loc[closest_index]], ignore_index = True)
            
        # Step 3: Deletion Operator
        classifier = KIBL(X=prototypes, K=1)
        classifier.kIBLAlgorithm(data)
        predictions = classifier.predictions

        # Identify prototypes that participate in classification
        participating_prototypes = predictions.unique()

        # Filter prototypes to keep only those that participate in classification
        final_prototypes = prototypes[np.isin(np.arange(len(prototypes)), participating_prototypes)]

        return final_prototypes
        

    def compute_centroid(self, X):
        # Calculate the centroid of a set of instances
        return X.mean()

    def find_closest_instance(self, centroid, instances):
        # Find the instance closest to the centroid
        distances = np.linalg.norm(instances - centroid, axis=1)
        closest_index = np.argmin(distances)
        return closest_index



