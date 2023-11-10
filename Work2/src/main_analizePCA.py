import numpy as np
from matplotlib import pyplot as plt
from scipy.io import arff
import sys

from sklearn.decomposition import TruncatedSVD, IncrementalPCA

sys.path.append('../')

from utils.data_preprocessing import Dataset
from algorithms.BIRCH import BIRCHClustering
from algorithms.PCA import CustomPCA
from algorithms.TruncatedSVD import find_best_n_components
from sklearn.decomposition import PCA

DATASET = "kr-vs-kp"
TYPE_DATASET = "categorical"
data_path = f"../data/raw/{DATASET}.arff"

# Load ARFF dataset and metadata
data, meta = arff.loadarff(data_path)

# Preprocessing data
dataset = Dataset(data_path, method=TYPE_DATASET)
X_original = dataset.processed_data.drop(columns=['y_true']).values
y_original = dataset.y_true


if DATASET == "waveform":
      y_pca = np.array(y_original.values).astype(int)
if DATASET == "kr-vs-kp":
      y_pca = y_original.map({'won': 0, 'nowin': 1})

THRESHOLD = 85


"""
Perform reduction of dimensionality using our PCA
"""
pca_1 = CustomPCA(X_original, threshold=THRESHOLD)
pca_1.fit()
pca_1.plot_components(DATASET)
X_pca_1 = pca_1.X_transformed
num_components = pca_1.k

"""
Perform reduction of dimensionality using sklearn PCA
"""
X_pca_2 = PCA(n_components=num_components).fit(X_original)

print(f"Transformed data shape: ({X_pca_2.n_samples_}, {X_pca_2.n_components_}) captures {sum(X_pca_2.explained_variance_ratio_)*100:0.2f}% of total "
      f"variation (Sklearn PCA)")
#plot_2D(X_pca_2, DATASET, y_pca)

"""
Perform reduction of dimensionality using sklearn IncrementalPCA
"""
X_pca_3 = IncrementalPCA(n_components=num_components).fit(X_original)
print(f"Transformed data shape: ({X_pca_3.n_samples_seen_}, {X_pca_3.n_components_}) captures {sum(X_pca_3.explained_variance_ratio_)*100:0.2f}% of total "
      f"variation (Sklearn IncrementalPCA)")


# Create subplots with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot for Custom PCA
axes[0].scatter(X_pca_1[:, 0], X_pca_1[:, 1], c=y_pca, cmap='viridis')
axes[0].set_title('Custom PCA')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# Plot for scikit-learn PCA
axes[1].scatter(X_pca_2.transform(X_original)[:, 0], X_pca_2.transform(X_original)[:, 1], c=y_pca, cmap='viridis')
axes[1].set_title('Scikit-learn PCA')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

# Plot for scikit-learn IncrementalPCA
axes[2].scatter(X_pca_3.transform(X_original)[:, 0], X_pca_3.transform(X_original)[:, 1], c=y_pca, cmap='viridis')
axes[2].set_title('Scikit-learn IncrementalPCA')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')

# Adjust layout
plt.tight_layout()

# Save the figure or display it
plt.savefig(f'../Results/images/{DATASET}_Comparison_PCA.png')
