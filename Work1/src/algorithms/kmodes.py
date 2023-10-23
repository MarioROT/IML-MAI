import numpy as np

class KModes:

    def __init__(self, n_clusters=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        X = X.astype(np.uint)

        # Insert the first K objects into K new clusters.
        initial_clusters = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        # Calculate the initial K modes for K clusters.
        self.cluster_centers_ = np.array([self._mode(cluster.reshape(1, -1)) for cluster in initial_clusters])

        previous_labels = np.zeros(X.shape[0])
        for _ in range(self.max_iter):
            # For each object, calculate similarity and assign to the closest mode
            labels = np.argmin(self._dissimilarity(X), axis=1)

            # Check for convergence or if few objects change clusters
            if np.all(labels == previous_labels):
                break

            previous_labels = labels

            # Recalculate the cluster modes
            self.cluster_centers_ = np.array([self._mode(X[labels == i]) for i in range(self.n_clusters) if len(X[labels == i]) > 0])

        self.labels_ = labels
        return self
    
    def predict(self, X):
        X = X.astype(np.uint)
        return np.argmin(self._dissimilarity(X), axis=1)

    def _dissimilarity(self, X):
        """Compute dissimilarity matrix."""
        return np.array([[self._hamming_distance(x, mode) for mode in self.cluster_centers_] for x in X])

    @staticmethod
    def _hamming_distance(x1, x2):
        """Compute the Hamming distance between two samples."""
        return np.sum(x1 != x2)

    @staticmethod
    def _mode(data):
        """Compute mode for categorical data."""
        return [np.bincount(column).argmax() for column in data.T]
