import sys
import seaborn as sns
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

sys.path.append('../')
from utils.data_preprocessing import Dataset
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from numpy.linalg import eig


class CustomPCA:
    def __init__(self, X, k=None, threshold=85):
        print("----Performing PCA----")
        self.X = X
        self.k = k
        self.threshold_exp_var = threshold  # 85%
        self.X_transformed = None
        self.cum_explained_variance = 0

    def fit(self):
        #print("Original dataset: ", self.X)
        # Compute the mean centered vector of the data
        mean_centered_vector = (self.X - np.mean(self.X, axis=0))
        # Calculate covariance matrix
        cov_matrix = np.cov(mean_centered_vector, rowvar=False)
        #print("Covariance matrix:", cov_matrix)

        # Eigenvalues and eigenvectors
        eig_vals, eig_vecs = eig(cov_matrix)

        # Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs * signs[np.newaxis, :]
        eig_vecs = eig_vecs.T

        #print('Eigenvalues \n', eig_vals)
        #print('Eigenvectors \n', eig_vecs)

        # Rearrange the eigenvectors and eigenvalues
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i, :]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])

        #print("Eigenvector Matrix (sorted by eigenvalues):", eig_pairs)

        # Choose principal components
        eig_vals_total = sum(eig_vals)
        explained_variance = [(i / eig_vals_total) * 100 for i in eig_vals_sorted]
        explained_variance = np.round(explained_variance, 2)
        self.cum_explained_variance = np.cumsum(explained_variance)

        # Determine k based on cumulative explained variance. Higher than 'threshold_exp_var'
        if self.k is None:
            self.k = np.argmax(self.cum_explained_variance >= self.threshold_exp_var) + 1
        W = eig_vecs_sorted[:self.k, :]

        # Project data
        self.X_transformed = self.X.dot(W.T)
        #print("Original data shape: ", self.X.shape)
        #print("Transformed dataset: ", self.X_transformed)
        #print(
            #f"Transformed data shape: {self.X_transformed.shape} captures {self.cum_explained_variance[self.X_transformed.shape[1] - 1]:0.2f}% of total "
            #f"variation (own PCA)")

    def plot_components(self, dataset_name):
        plt.figure(figsize=(12, 8))
        n_samples, n_features = self.X.shape
        plt.plot(np.arange(1, n_features + 1), self.cum_explained_variance, '-o')
        plt.xticks(np.arange(1, n_features + 1))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('PCA: Explained Variance Ratio vs. Number of Components')
        plt.savefig(f'../Results/images/{dataset_name}_PCAcomponents.png')

    def plot_2D(self, dataset_name, y):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.X_transformed[:, 0], self.X_transformed[:, 1], c=y, cmap='viridis')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'2 components')
        plt.savefig(f'../Results/images/{dataset_name}_PCA2D.png')
