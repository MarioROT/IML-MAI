import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import arff
import sys
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA

# Add the path to your utility modules if needed
sys.path.append('../')
from utils.data_preprocessing import Dataset
from algorithms.PCA import CustomPCA


def load_arff_dataset(data_path):
    data, meta = arff.loadarff(data_path)
    return data, meta


def preprocess_data(data_path, type_dataset):
    dataset = Dataset(data_path, method=type_dataset)
    X_original = dataset.processed_data.drop(columns=['y_true']).values
    y_original = dataset.y_true
    return X_original, y_original


def map_labels(dataset, y_original):
    if dataset == "waveform":
        return np.array(y_original.values).astype(int)
    elif dataset == "kr-vs-kp":
        return y_original.map({'won': 0, 'nowin': 1})
    elif dataset == "vowel":
        return y_original.map(
            {'hid': 0, 'hId': 1, 'hEd': 2, 'hAd': 3, 'hYd': 4, 'had': 5, 'hOd': 6, 'hod': 7, 'hUd': 8, 'hud': 9,
             'hed': 10})


def perform_custom_pca(X_original, threshold):
    pca = CustomPCA(X_original, threshold=threshold)
    pca.fit()
    pca.plot_components(DATASET)
    X_pca = pca.X_transformed
    num_components = pca.k
    return X_pca, num_components


def perform_sklearn_pca(X_original, num_components):
    X_pca = PCA(n_components=num_components).fit(X_original)
    explained_variance = sum(X_pca.explained_variance_ratio_) * 100
    print(
        f"Transformed data shape: ({X_pca.n_samples_}, {X_pca.n_components_}) captures {explained_variance:0.2f}% of "
        f"total variation (Sklearn PCA)")
    return X_pca


def perform_incremental_pca(X_original, num_components):
    X_pca = IncrementalPCA(n_components=num_components).fit(X_original)
    explained_variance = sum(X_pca.explained_variance_ratio_) * 100
    print(
        f"Transformed data shape: ({X_pca.n_samples_seen_}, {X_pca.n_components_}) captures {explained_variance:0.2f}"
        f"% of total variation (Sklearn IncrementalPCA)")
    return X_pca


def plot_pca_comparison(X_pca_custom, X_pca_sklearn, X_pca_incremental, y_pca):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot for Custom PCA
    axes[0].scatter(X_pca_custom[:, 0], X_pca_custom[:, 1], c=y_pca, cmap='viridis')
    axes[0].set_title('Custom PCA')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')

    # Plot for scikit-learn PCA
    axes[1].scatter(X_pca_sklearn.transform(X_original)[:, 0], X_pca_sklearn.transform(X_original)[:, 1], c=y_pca,
                    cmap='viridis')
    axes[1].set_title('Scikit-learn PCA')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')

    # Plot for scikit-learn IncrementalPCA
    axes[2].scatter(X_pca_incremental.transform(X_original)[:, 0], X_pca_incremental.transform(X_original)[:, 1],
                    c=y_pca, cmap='viridis')
    axes[2].set_title('Scikit-learn IncrementalPCA')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')

    # Adjust layout
    plt.tight_layout()

    # Save the figure or display it
    plt.savefig(f'../results/images/{DATASET}_Comparison_PCA.png')


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='PCA Comparison Script')

    # Add arguments
    parser.add_argument('--dataset', type=str, choices=['waveform', 'kr-vs-kp', 'vowel'], default='waveform',
                        help='Specify the dataset (waveform, kr-vs-kp, vowel)')
    parser.add_argument('--type_dataset', type=str, choices=['numerical', 'categorical', 'mixed'], default='numerical',
                        help='Specify the type of dataset (numerical, categorical, mixed)')

    args = parser.parse_args()

    DATASET = args.dataset
    TYPE_DATASET = args.type_dataset
    data_path = f"../data/raw/{DATASET}.arff"

    # Load ARFF dataset and metadata
    data, meta = load_arff_dataset(data_path)

    # Preprocess data
    X_original, y_original = preprocess_data(data_path, TYPE_DATASET)

    # Map labels
    y_pca = map_labels(DATASET, y_original)

    THRESHOLD = 85

    # Perform Custom PCA
    X_pca_custom, num_components = perform_custom_pca(X_original, THRESHOLD)

    # Perform sklearn PCA
    X_pca_sklearn = perform_sklearn_pca(X_original, num_components)

    # Perform sklearn IncrementalPCA
    X_pca_incremental = perform_incremental_pca(X_original, num_components)

    # Plot PCA comparison
    plot_pca_comparison(X_pca_custom, X_pca_sklearn, X_pca_incremental, y_pca)
