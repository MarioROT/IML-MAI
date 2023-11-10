import argparse
import timeit
import numpy as np
import pandas as pd
from scipy.io import arff
import sys
from sklearn.decomposition import TruncatedSVD

sys.path.append('../')

from utils.data_preprocessing import Dataset
from algorithms.BIRCH import BIRCHClustering
from algorithms.PCA import CustomPCA
from algorithms.TruncatedSVD import find_best_n_components
from algorithms.kmeans import KMeans
from evaluation.metrics import performance_eval


def run_clustering(X, y, method, threshold):
    """
    Run Birch and K-means clustering on the provided data.

    Parameters:
    - X: Features matrix
    - y: Target variable
    - method: Clustering method ('BIRCH' or 'KMeans')
    - threshold: Threshold for dimensionality reduction

    Returns:
    - None
    """
    if method == 'BIRCH':
        clustering = BIRCHClustering(X, y)
        clustering.search_best_params()
        clustering.print_best_params()
    elif method == 'KMeans':
        # Run K-means clustering
        kmeans = KMeans(k=3)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Evaluate performance
        print(f"K-means clustering ({method}):")
        performance_eval(X, labels, y)
    else:
        raise ValueError("Invalid clustering method.")

    # Add K-means implementation for method 'KMeans' if needed
    # ...

def run_dimensionality_reduction(X, method, threshold):
    """
    Perform dimensionality reduction on the provided data.

    Parameters:
    - X: Features matrix
    - method: Dimensionality reduction method ('PCA' or 'TruncatedSVD')
    - threshold: Threshold for dimensionality reduction

    Returns:
    - Transformed data after dimensionality reduction
    """
    if method == 'PCA':
        pca = CustomPCA(X, threshold=threshold)
        pca.fit()
        return pca.X_transformed
    elif method == 'TruncatedSVD':
        best_n_components, _ = find_best_n_components(X, threshold=threshold)
        svd = TruncatedSVD(n_components=best_n_components)
        return svd.fit_transform(X)
    else:
        raise ValueError("Invalid dimensionality reduction method.")

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Clustering and Dimensionality Reduction Script')

    # Add arguments
    parser.add_argument('--dataset', type=str, choices=['waveform', 'kr-vs-kp', 'vowel'], default='waveform',
                        help='Specify the dataset (waveform, kr-vs-kp, vowel)')
    parser.add_argument('--type_dataset', type=str, choices=['numerical', 'categorical', 'mixed'], default='numerical',
                        help='Specify the type of dataset (numerical, categorical, mixed)')

    args = parser.parse_args()

    DATASET = args.dataset
    TYPE_DATASET = args.type_dataset
    data_path = f"../data/raw/{DATASET}.arff"

    # Load ARFF dataset and metadata
    data, meta = arff.loadarff(data_path)

    # Preprocessing data
    dataset = Dataset(data_path, method=TYPE_DATASET)
    X_original = dataset.processed_data.drop(columns=['y_true']).values
    y_original = dataset.y_true

    THRESHOLD = 85

    ##################################################################################################################
    print("----------Running clustering without using dimensionality reduction----------")
    run_clustering(X_original, y_original, method='BIRCH', threshold=THRESHOLD)
    # TODO: Add K-means clustering without dimensionality reduction if needed
    print("----------Running Kmeans without using dimensionality reduction-----")
    run_clustering(X_original, y_original, method='KMeans', threshold=THRESHOLD)

    ##################################################################################################################
    ##################################################################################################################
    print("----------Running clustering using PCA----------")
    X_PCA = run_dimensionality_reduction(X_original, method='PCA', threshold=THRESHOLD)
    run_clustering(X_PCA, y_original, method='BIRCH', threshold=THRESHOLD)
    # TODO: Add K-means clustering using PCA if needed
    print("----------Running KMeans clustering using PCA----------")
    run_clustering(X_PCA, y_original, method='KMeans', threshold=THRESHOLD)

    ##################################################################################################################
    ##################################################################################################################
    print("----------Running clustering using TruncatedSVD----------")
    X_SVD = run_dimensionality_reduction(X_original, method='TruncatedSVD', threshold=THRESHOLD)
    run_clustering(X_SVD, y_original, method='BIRCH', threshold=THRESHOLD)
    # TODO: Add K-means clustering using TruncatedSVD if needed
    print("----------Running KMeans clustering using SVD----------")
    run_clustering(X_SVD, y_original, method='KMeans', threshold=THRESHOLD)
