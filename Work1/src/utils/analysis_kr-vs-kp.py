import sys
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns

# DATASET KR-VS-KP
DATASET = "kr-vs-kp"

# Load ARFF dataset and metadata
data, meta = arff.loadarff(f"datasets/{DATASET}.arff")

# Convert data to Pandas DataFrame
df = pd.DataFrame(data)
print(df)

# Extract target variable and drop the "class" column
y_true = df["class"]
X = df.drop(columns="class")

# Number of unique classes
num_classes = y_true.nunique()

# Attribute names
attribute_names = meta.names()
print("Features: ", attribute_names)

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

# Create Reduced DataFrame for Pairplot
X_reduced = X.copy()
X_reduced["class"] = y_true

print(X_reduced)

# Pairplot for Reduced Data
sns.set(style="ticks")
sns.pairplot(X_reduced, plot_kws={'alpha': 0.3})
plt.savefig(f'images/{DATASET}_pairplot.png')
plt.close()
