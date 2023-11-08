import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import sys
sys.path.append('../')
from utils.data_preprocessing import Dataset

def find_best_n_components(X, threshold=0.85):
    explained_variances = []
    n_features = X.shape[1]
    n_components_range = range(1, n_features + 1)
    best_n_components = None

    for n_components in n_components_range:
        svd = TruncatedSVD(n_components=n_components)
        X_transformed = svd.fit_transform(X)
        explained_variances.append(np.sum(svd.explained_variance_ratio_))

        if np.sum(svd.explained_variance_ratio_) >= threshold:
            best_n_components = n_components
            break

    return best_n_components, explained_variances

def plot_explained_variance(explained_variances):
    n_components_range = range(1, len(explained_variances) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(n_components_range, explained_variances, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('TruncatedSVD: Explained Variance Ratio vs. Number of Components')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    DATASET = "waveform"
    data_path = f"../../data/raw/{DATASET}.arff"

    # Preprocessing data
    dataset = Dataset(data_path, method="numerical")
    X = dataset.processed_data.drop(columns=['y_true']).values
    y = dataset.y_true

    # Find the best number of components
    best_n_components, explained_variances = find_best_n_components(X, threshold=0.85)

    print("Number of Components:", best_n_components)

    # Plot explained variance ratio
    plot_explained_variance(explained_variances)
