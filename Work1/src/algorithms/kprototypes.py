import numpy as np
import pandas as pd

class KPrototypes:
    def __init__(self, k, max_iters=100, gamma=None, gamma_factor=0.5):
        self.k = k
        self.max_iters = max_iters
        self.gamma = gamma
        self.gamma_factor = gamma_factor
        self.iter = 0


    def fit(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        # 2. Select k random instances {s1, s2,… sk} as seeds.
        self.centroids = data.iloc[np.random.choice(data.shape[0], self.k, replace=False)].copy()
        prev_labels = np.zeros(data.shape[0])

        for it in range(self.max_iters):
            self.iter = it 
            # 3. Decide the class membership(assigning each data point to the closest centroid)
            labels = self._assign_to_clusters(data, prev_labels)

            # Check for convergence
            if np.all(labels == prev_labels):
                break
            prev_labels = labels

            # 4. Update the seeds to the centroid of each cluster
            self.updte_centroids(data, labels)
            self.labels_ = self._assign_to_clusters(data, prev_labels)

    def predict(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        return self._assign_to_clusters(data, predict = True)

    def _assign_to_clusters(self, data, prev_labels = False, predict = False):
        num_feats = data.select_dtypes(exclude='object').values
        cat_feats = data.select_dtypes(exclude=num_feats.dtype).values
        if num_feats.size != 0:
            num_cents = self.centroids.select_dtypes(exclude='object').values
            num_distances = np.linalg.norm(num_feats[:, np.newaxis,] - num_cents, axis=2)
            distances_to_centroids = np.zeros(num_distances.shape)
        if cat_feats.size != 0:
           cat_cents = self.centroids.select_dtypes(exclude = 'number').values
           cat_distances = np.sum(np.not_equal(cat_feats[:, np.newaxis],cat_cents).astype(np.uint), axis=2)
           distances_to_centroids = np.zeros(cat_distances.shape)
        if num_feats.size != 0 and cat_feats.size != 0:
            if not self.gamma and not predict:
                if not predict:
                    self.gammas = self.compute_gamma(num_feats, prev_labels)
                for c, gamma in self.gammas.items():
                    distances_to_centroids[prev_labels == c] = num_distances[prev_labels == c] + (gamma * cat_distances[prev_labels == c])
            elif self.gamma: 
                distances_to_centroids = num_distances + (self.gamma * cat_distances)
            else:
                gamma = self.compute_gamma(num_feats, np.zeros(num_feats.shape[0]))
                distances_to_centroids = num_distances + (self.gamma * cat_distances)
        elif cat_feats.size == 0:
            distances_to_centroids = num_distances
        elif num_feats.size == 0:
            distances_to_centroids = cat_distances
        return np.argmin(distances_to_centroids, axis=1)

    def updte_centroids(self, data, labels):
        for i in range(self.k):
            stats = data.iloc[labels == i].describe(include='all')
            if 'top' in stats.T.keys() and 'mean' in stats.T.keys():
                self.centroids.iloc[i] = stats.loc['top'].combine_first(stats.loc['mean']).values
            elif not 'top' in stats.T.keys():
                self.centroids.iloc[i] = stats.loc['mean']
            elif not 'mean' in stats.T.keys():
                self.centroids.iloc[i] = stats.loc['top']

    def compute_gamma(self, num_feats, prev_labels):
        gammas = {}
        for c in np.unique(prev_labels):
            gammas[c] = self.gamma_factor * np.mean(num_feats[prev_labels == c].std(axis=0))
        return gammas

