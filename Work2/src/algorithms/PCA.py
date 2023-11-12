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
        self.X_reconstructed = None
        self.X_transformed = None
        self.cum_explained_variance = 0

    def fit(self):
        print("Original dataset: ", self.X)
        # Compute the mean centered vector of the data
        mean_centered_vector = (self.X - np.mean(self.X, axis=0))
        # Calculate covariance matrix
        cov_matrix = np.cov(mean_centered_vector, rowvar=False)
        print("Covariance matrix:", cov_matrix)

        # Eigenvalues and eigenvectors
        self.eig_vals, self.eig_vecs = eig(cov_matrix)

        # Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive
        max_abs_idx = np.argmax(np.abs(self.eig_vecs), axis=0)
        signs = np.sign(self.eig_vecs[max_abs_idx, range(self.eig_vecs.shape[0])])
        self.eig_vecs = self.eig_vecs * signs[np.newaxis, :]
        self.eig_vecs = self.eig_vecs.T

        print('Eigenvalues \n', self.eig_vals)
        print('Eigenvectors \n', self.eig_vecs)

        # Rearrange the eigenvectors and eigenvalues
        self.eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[i, :]) for i in range(len(self.eig_vals))]
        self.eig_pairs.sort(key=lambda x: x[0], reverse=True)
        self.eig_vals_sorted = np.array([x[0] for x in self.eig_pairs])
        self.eig_vecs_sorted = np.array([x[1] for x in self.eig_pairs])

        print("Eigenvector Matrix (sorted by eigenvalues):", self.eig_pairs)

        # Choose principal components
        self.eig_vals_total = sum(self.eig_vals)
        explained_variance = [(i / self.eig_vals_total) * 100 for i in self.eig_vals_sorted]
        explained_variance = np.round(explained_variance, 2)
        self.cum_explained_variance = np.cumsum(explained_variance)

        # Determine k based on cumulative explained variance. Higher than 'threshold_exp_var'
        if self.k is None:
            self.k = np.argmax(self.cum_explained_variance >= self.threshold_exp_var) + 1
        W = self.eig_vecs_sorted[:self.k, :]

        # Project data
        self.X_transformed = self.X.dot(W.T)
        self.X_reconstructed = self.X_transformed.T.dot(W.T) + np.mean(self.X, axis=0)
        print("Original data shape: ", self.X.shape)
        print("Transformed dataset: ", self.X_transformed)
        print(
            f"Transformed data shape: {self.X_transformed.shape} captures {self.cum_explained_variance[self.X_transformed.shape[1] - 1]:0.2f}% of total "
            f"variation (own PCA)")

    def visualize(self, labels, axes=[0, 1, 2, 3], figsize=(10, 10), data2plot='Original', exclude = [], layout=None, axis=None, title_size = 12, save=None):
        data = {'Original':self.X, 'Reconstructed':self.X_reconstructed, 'Transformed':self.X_transformed}[data2plot]
        title = data2plot + ' Data'  

        plots ={}
        for comb in [com for sub in range(1,4) for com in combinations(axes, sub + 1)]:
            if str(len(comb))+'d' not in plots.keys():
                plots[str(len(comb))+'d'] = []
            plots[str(len(comb))+'d'].append(comb)

        plots['scree'] = [(True)]  
        plots = {k:v for k,v in plots.items() if k not in exclude}

        layout = [len(plots), max(len(l) for l in plots.values())] if not layout else layout

        cg = custom_grids([],layout[0], layout[1], figsize=figsize, axis=axis, title_size=title_size, use_grid_spec = False)
        cg.show()

        for k,group in plots.items():
            for i,v in enumerate(group):
                if k in ['2d', '3d', '4d']:
                    if k == '2d':
                        ax = cg.add_plot(title+ ' ' + self.Version, clear_ticks=True, axlabels=v)
                        ax.scatter(data[:, v[0]], data[:, v[1]], c=labels)
                    else: 
                        ax = cg.add_plot(title+ ' ' + self.Version,projection=True, clear_ticks=True, axlabels=v, row_last= True if i == len(group)-1 else False)
                        ax.scatter(data[:, v[0]], data[:, v[1]], data[:, v[2]], c=labels, s=15 if k == '3d' else data[:, v[3]] * 10)
                elif k == 'scree':
                    p = cg.add_plot('Explained Variance {}'.format(self.Version), axlabels=['Number of Components','Cumulative explained variance'], last=True)
                    p.plot(np.cumsum(self.explained_variance_ratio), marker='.', color=colors[1])
                    p.bar(list(range(0, self.n_features)), self.explained_variance_ratio, color=colors[2])
        plt.show()

        if save:
            plt.savefig(save)
