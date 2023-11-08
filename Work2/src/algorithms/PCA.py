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


class PCA:
    def __init__(self, X, k=None):
        print("----Performing PCA----")
        self.X = X
        self.k = k
        self.threshold_exp_var = 85 #85%

    def fit(self):
        # Compute the mean centered vector of the data
        mean_centered_vector = (self.X - np.mean(self.X, axis=0))
        # Calculate covariance matrix
        cov_matrix = np.cov(mean_centered_vector, rowvar=False)
        print("Covariance matrix:", cov_matrix)

        # Eigenvalues and eigenvectors
        eig_vals, eig_vecs = eig(cov_matrix)

        # Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs * signs[np.newaxis, :]
        eig_vecs = eig_vecs.T

        print('Eigenvalues \n', eig_vals)
        print('Eigenvectors \n', eig_vecs)

        # Rearrange the eigenvectors and eigenvalues
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i, :]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])

        print("Eigenvector Matrix (sorted by eigenvalues):", eig_pairs)

        # Choose principal components
        eig_vals_total = sum(eig_vals)
        explained_variance = [(i / eig_vals_total) * 100 for i in eig_vals_sorted]
        explained_variance = np.round(explained_variance, 2)
        cum_explained_variance = np.cumsum(explained_variance)

        # Plot
        plt.plot(np.arange(1, n_features + 1), cum_explained_variance, '-o')
        plt.xticks(np.arange(1, n_features + 1))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.show()

        # Determine k based on cumulative explained variance
        if self.k is None:
            self.k = np.argmax(cum_explained_variance >= self.threshold_exp_var) + 1
        W = eig_vecs_sorted[:self.k, :]

        # Project data
        X_proj = self.X.dot(W.T)
        print("Original data shape: ", self.X.shape)
        print(f"Transformed data shape: {X_proj.shape} captures {cum_explained_variance[X_proj.shape[1]-1]} of total "
              f"variation")

        """
        # Plot
        plt.scatter(X_proj[:, 0], X_proj[:, 1])
        plt.xlabel('PC1')
        plt.xticks([])
        plt.ylabel('PC2')
        plt.yticks([])
        plt.title('2 components, captures {} of total variation'.format(cum_explained_variance[1]))
        plt.show()

        plt.close()"""


if __name__ == "__main__":
    DATASET = "kr-vs-kp"
    data_path = f"../../data/raw/{DATASET}.arff"

    # Load ARFF dataset and metadata
    data, meta = arff.loadarff(data_path)

    # Plot original data
    df = pd.DataFrame(data)
    print(df)

    X_original = df.drop(columns=["class"])
    y_true = df["class"]
    features = meta.names()

    n_samples, n_features = X_original.shape

    print('Number of samples:', n_samples)
    print('Number of features:', n_features)

    """sns.set(style="ticks")
    sns.pairplot(X_original, corner=True)
    plt.show()"""


    # Preprocessing data
    dataset = Dataset(data_path, method="categorical")
    X = dataset.processed_data.drop(columns=['y_true']).values
    y = dataset.y_true

    X_std = StandardScaler().fit_transform(X)

    pca = PCA(X_std)
    pca.fit()

