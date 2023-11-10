import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def find_best_n_components(X, threshold=85):
    """
    Find the optimal number of components for TruncatedSVD based on the explained variance ratio.

    Parameters:
    - X: Input data, a 2D array-like or sparse matrix.
    - threshold: The minimum cumulative explained variance ratio to achieve.

    Returns:
    - best_n_components: The optimal number of components.
    - explained_variances: List of explained variances for each number of components.
    """
    if X.ndim != 2:
        raise ValueError("Input data should be a 2D array-like or sparse matrix.")

    explained_variances = []
    n_features = X.shape[1]
    n_components_range = range(1, n_features + 1)
    best_n_components = None

    for n_components in n_components_range:
        svd = TruncatedSVD(n_components=n_components)
        X_transformed = svd.fit_transform(X)
        explained_variances.append(np.sum(svd.explained_variance_ratio_) * 100)

        if np.sum(svd.explained_variance_ratio_) * 100 >= threshold:
            best_n_components = n_components
            break

    print(f"Transformed data shape: ({X.shape[0]}, {best_n_components}) captures {explained_variances[best_n_components-1]:.2f}% of total "
          f"variation (TruncatedSVD)")

    return best_n_components, explained_variances

def plot_explained_variance(explained_variances):
    """
    Plot the cumulative explained variance ratio for different numbers of components.

    Parameters:
    - explained_variances: List of explained variances for each number of components.

    Returns:
    - None
    """
    n_components_range = range(1, len(explained_variances) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range, explained_variances, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio (%)')
    plt.title('TruncatedSVD: Explained Variance Ratio vs. Number of Components')
    plt.grid(True)
    plt.show()
