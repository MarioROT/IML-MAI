import numpy as np
class KPrototypes:
    def __init__(self, k, max_iters=100, gamma=None):
        self.cat_idx = cat_idx
        self.k = k
        self.max_iters = max_iters
        self.gamma = gamma
        self.iter = 0


    def fit(self, data):
        # 2. Select k random instances {s1, s2,… sk} as seeds.
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        prev_labels = np.zeros(data.shape[0])

        for it in range(self.max_iters):
            self.iter = it 
            # 3. Decide the class membership(assigning each data point to the closest centroid)
            labels = self._assign_to_clusters(data)

            # Check for convergence
            if np.all(labels == prev_labels):
                break
            prev_labels = labels

            # 4. Update the seeds to the centroid of each cluster
            self.centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
            self.labels_ = self._assign_to_clusters(data)

    def predict(self, data):
        return self._assign_to_clusters(data)

    def _assign_to_clusters(self, data):
        num_feats = data.select_dtypes(exclude='object').values
        cat_feats = data.select_dtypes(exclude=num_feats.dtype)
        num_distances = np.linalg.norm(data[:, np.newaxis, :self.cat_idx].astype(np.uint) - self.centroids[:,:self.cat_idx]np.astype(np.uint), axis=2)
        cat_distances = np.sum(np.not_equal(data[:, np.newaxis, self.cat_idx:],self.centroids[:,:self.centroids]).astype(np.uint8), axis=2)
        self.gamma = 
        distances_to_centroids = num_distances + cat_distances
        return np.argmin(distances_to_centroids, axis=1)

    # def updte_centroids(self, data, labels):

    def compute_gamma(data):
        if self.iter == 0:
            gama = 

