import sys
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns

# DATASET WAVEFORM
DATASET = "waveform"

# Load ARFF dataset and metadata
data, meta = arff.loadarff(f"datasets/{DATASET}.arff")

# Convert data to Pandas DataFrame
df = pd.DataFrame(data)

# Extract target variable and drop the "class" column
y_true = df["class"]
X = df.drop(columns="class")

# Number of unique classes
num_classes = y_true.nunique()

# Attribute names
attribute_names = meta.names()

# Number of samples and features
num_samples, num_features = X.shape

# Summary Statistics
statistics = X.describe()

# Class counts
class_counts = y_true.value_counts()

# Correlation Matrix
correlation_matrix = X.corr()

# Frequency of Features
feature_frequencies = X.nunique()

# Print Results
print(f"Number of Samples: {num_samples}")
print(f"Number of Features: {num_features}")
print(f"Number of Classes: {num_classes}")
print("Features: ", attribute_names)
print("Summary Statistics:")
print(statistics)

# Plot Frequency of Classes
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar')
plt.title('Frequency of Classes')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.savefig(f'images/{DATASET}_classes.png')
plt.close()

# Plot Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig(f'images/{DATASET}_cm.png')
plt.close()

# Reduce features for pairplot
X_reduced = X.iloc[:, :20]
X_reduced["class"] = y_true

# Pairplot for Reduced Data
sns.set(style="ticks")
sns.pairplot(X_reduced, hue=X_reduced.columns[-1], markers='.', palette='Set1', plot_kws={'alpha': 0.3})
plt.savefig(f'images/{DATASET}_pairplot.png')
plt.close()
