import sys
import pandas as pd
from matplotlib.cm import ScalarMappable
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns

# DATASET WAVEFORM
DATASET = "waveform"
data_path = f"../../data/raw/{DATASET}.arff"

# Load ARFF dataset and metadata
data, meta = arff.loadarff(data_path)

# Convert data to Pandas DataFrame
df = pd.DataFrame(data)

# Extract target variable and drop the "class" column
y_true = df["class"]
X = df.drop(columns="class")

num_classes = y_true.nunique()
attribute_names = meta.names()
num_samples, num_features = X.shape
statistics = X.describe()
class_counts = y_true.value_counts()
correlation_matrix = X.corr()
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
plt.savefig(f'../../results/images/{DATASET}_classes.png')
plt.close()

# Plot Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig(f'../../results/images/{DATASET}_cm.png')
plt.close()

# Find pair of features with the highest positive correlation
highest_positive_corr = correlation_matrix[correlation_matrix != 1.0].stack().idxmax()
feature1_pos, feature2_pos = highest_positive_corr
correlation_value_pos = correlation_matrix.loc[feature1_pos, feature2_pos]

# Find pair of features with the highest negative correlation
highest_negative_corr = correlation_matrix[correlation_matrix != -1.0].stack().idxmin()
feature1_neg, feature2_neg = highest_negative_corr
correlation_value_neg = correlation_matrix.loc[feature1_neg, feature2_neg]

print(f"Highest positive correlation: {feature1_pos} and {feature2_pos}, Correlation = {correlation_value_pos:0.3f}")
print(f"Highest negative correlation: {feature1_neg} and {feature2_neg}, Correlation = {correlation_value_neg:0.3f}")

# Create subplots with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot the scatter plot for highest positive correlation
scatter_pos = axes[0].scatter(X[feature1_pos], X[feature2_pos], c=y_true, marker='.', cmap='viridis')
axes[0].set_xlabel(feature1_pos)
axes[0].set_ylabel(feature2_pos)
axes[0].set_title(f'Scatter Plot: {feature1_pos} vs {feature2_pos}')

# Plot the scatter plot for highest negative correlation
scatter_neg = axes[1].scatter(X[feature1_neg], X[feature2_neg], c=y_true, marker='.')
axes[1].set_xlabel(feature1_neg)
axes[1].set_ylabel(feature2_neg)
axes[1].set_title(f'Scatter Plot: {feature1_neg} vs {feature2_neg}')

plt.tight_layout()
plt.savefig(f'../../results/images/{DATASET}_scatter_plots.png')
