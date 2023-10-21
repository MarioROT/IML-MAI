import numpy as np
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, data):
        # 2. Select k random instances {s1, s2,… sk} as seeds.
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        prev_labels = np.zeros(data.shape[0])

        for _ in range(self.max_iters):
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
        distances_to_centroids = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances_to_centroids, axis=1)
