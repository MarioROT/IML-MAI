from algorithms.kmeans import KMeans
from algorithms.kmodes import KModes
from algorithms.kprototypes import KPrototypes
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import compute_accuracy
from utils.data_preprocessing import Dataset

def process_kprototypes(data_path):
    dataset = Dataset(data_path)
    X = dataset.processed_data.iloc[:, :-1]
    true_labels = dataset.processed_data['y_true'].values

    # My KMeans
    kprot = KPrototypes(k=3)
    kprot.fit(X)
    labels = kprot.predict(X)
    accuracy = compute_accuracy(labels, true_labels)
    print(f"My KPrototypes Accuracy: {accuracy * 100:.2f}%")


def process_kmeans(data_path):
    dataset = Dataset(data_path)
    X = dataset.processed_data.iloc[:, :-1].values
    true_labels = dataset.processed_data['y_true'].values

    # My KMeans
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    accuracy = map_clusters_to_labels(labels, true_labels)
    print(f"My KMeans Accuracy: {accuracy * 100:.2f}%")

    # Scikit-learn KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sk_kmeans = SKLearnKMeans(n_clusters=3, random_state=42, n_init=10)
    sk_kmeans.fit(X)
    sk_labels = sk_kmeans.labels_
    sk_accuracy = map_clusters_to_labels(sk_labels, true_labels)
    print(f"Scikit-learn KMeans Accuracy: {sk_accuracy * 100:.2f}%")

    #print(labels)
    #print(sk_labels)
    #print(true_labels)


def process_kmodes(data_path):
    dataset = Dataset(data_path)
    X = dataset.processed_data.iloc[:, :-1].values.astype(int)
    true_labels = dataset.processed_data['y_true'].values

    # Kmodes
    kmodes = KModes(n_clusters=3, random_state=42)
    kmodes.fit(X)
    kmodes_labels = kmodes.labels_
    kmodes_accuracy = map_clusters_to_labels(kmodes_labels, true_labels)
    print(f"Kmodes Accuracy: {kmodes_accuracy * 100:.2f}%")

    #print(kmodes_labels)
    #print(true_labels)


if __name__ == "__main__":
    #print("Processing KMeans on waveform.arff")
    # process_kmeans('../data/raw/waveform.arff')
    # process_kmeans('../data/raw/iris.arff')
    process_kprototypes('../data/raw/iris.arff')

    #print("\nProcessing KModes on dataset_24_mushroom.arff")
    # process_kmodes('../data/raw/dataset_24_mushroom.arff')
