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
from algorithms.PCA import PCA
from algorithms.TruncatedSVD import find_best_n_components

DATASET = "waveform"
data_path = f"../data/raw/{DATASET}.arff"

# Load ARFF dataset and metadata
data, meta = arff.loadarff(data_path)

# Preprocessing data
dataset = Dataset(data_path, method="numerical")
X_original = dataset.processed_data.drop(columns=['y_true']).values
y_original = dataset.y_true

THRESHOLD = 85

##################################################################################################################
print("----------Running clustering without using dimensionality reduction----------")
"""
Run Birch
"""
BIRCHClustering_original = BIRCHClustering(X_original, y_original)
BIRCHClustering_original.search_best_params()
BIRCHClustering_original.print_best_params()


"""
Run K-means
#TODO
"""


##################################################################################################################
##################################################################################################################
"""
Perform reduction of dimensionality using PCA
"""
pca = PCA(X_original, threshold=THRESHOLD)
pca.fit()
X_PCA = pca.X_transformed  # Transformed data after PCA

print("----------Running clustering using PCA----------")
"""
Run Birch
"""
BIRCHClustering_PCA = BIRCHClustering(X_PCA, y_original)
BIRCHClustering_PCA.search_best_params()
BIRCHClustering_PCA.print_best_params()


"""
Run K-means
#TODO
"""

##################################################################################################################
##################################################################################################################
"""
Perform reduction of dimensionality using TruncatedSVD
"""
best_n_components, _ = find_best_n_components(X_original, threshold=THRESHOLD)
svd = TruncatedSVD(n_components=best_n_components)
X_SVD = svd.fit_transform(X_original)

print("----------Running clustering using TruncatedSVD----------")
"""
Run Birch
"""
BIRCHClustering_SVD = BIRCHClustering(X_SVD, y_original)
BIRCHClustering_SVD.search_best_params()
BIRCHClustering_SVD.print_best_params()


"""
Run K-means
#TODO
"""