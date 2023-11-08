from scipy.io import arff
import sys

from sklearn.decomposition import TruncatedSVD, IncrementalPCA

sys.path.append('../')

from utils.data_preprocessing import Dataset
from algorithms.BIRCH import BIRCHClustering
from algorithms.PCA import CustomPCA
from algorithms.TruncatedSVD import find_best_n_components
from sklearn.decomposition import PCA

DATASET = "waveform"
data_path = f"../data/raw/{DATASET}.arff"

# Load ARFF dataset and metadata
data, meta = arff.loadarff(data_path)

# Preprocessing data
dataset = Dataset(data_path, method="numerical")
X_original = dataset.processed_data.drop(columns=['y_true']).values
y_original = dataset.y_true

THRESHOLD = 85

"""
Perform reduction of dimensionality using our PCA
"""
pca_1 = CustomPCA(X_original, threshold=THRESHOLD)
pca_1.fit()
X_pca_1 = pca_1.X_transformed
num_components = pca_1.k

"""
Perform reduction of dimensionality using sklearn PCA
"""
X_pca_2 = PCA(n_components=num_components).fit(X_original)

print(f"Transformed data shape: ({X_pca_2.n_samples_}, {X_pca_2.n_components_}) captures {sum(X_pca_2.explained_variance_ratio_)*100:0.2f}% of total "
      f"variation (Sklearn PCA)")

"""
Perform reduction of dimensionality using sklearn IncrementalPCA
"""
X_pca_3 = IncrementalPCA(n_components=num_components).fit(X_original)
print(f"Transformed data shape: ({X_pca_3.n_samples_seen_}, {X_pca_3.n_components_}) captures {sum(X_pca_3.explained_variance_ratio_)*100:0.2f}% of total "
      f"variation (Sklearn IncrementalPCA)")
