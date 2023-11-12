import sys
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns

# DATASET VOWEL
DATASET = "vowel"

# Load ARFF dataset and metadata
data, meta = arff.loadarff(f"../../data/raw/{DATASET}.arff")

# Convert data to Pandas DataFrame
df = pd.DataFrame(data)
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Map class labels
mapping_dict = {'hid': 0, 'hId': 1, 'hEd': 2, 'hAd': 3, 'hYd': 4, 'had': 5, 'hOd': 6, 'hod': 7, 'hUd': 8, 'hud': 9,
                'hed': 10}
y_true = df["Class"].map(mapping_dict)

# Process features
X = df.drop(columns=["Class", "Train_or_Test", "Speaker_Number"])
X['Sex'] = X['Sex'].map({'Male': 1, 'Female': 0})
X_2 = X.drop(columns=["Sex"])

num_classes = y_true.nunique()
attribute_names = meta.names()
num_samples, num_features = X.shape
statistics = X.describe()
class_counts = y_true.value_counts()
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
plt.savefig(f'../../results/images/{DATASET}_classes.png')
plt.close()

# Correlation Matrix
correlation_matrix = X_2.corr()

# Plot Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig(f'../../results/images/{DATASET}_cm.png')
plt.close()

# Select the relevant columns
selected_columns = ['Feature_1', 'Feature_7', 'Feature_8']

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()
for i, feature in enumerate(selected_columns):
    sns.scatterplot(x=df['Sex'], y=df[feature], data=df, hue='Class', palette='viridis', ax=axes[i])
    axes[i].set_title(f'Scatter Plot: {feature} and Sex')
    axes[i].set_xlabel('Sex')
    axes[i].set_ylabel(feature)
plt.tight_layout()
plt.savefig(f'../../results/images/{DATASET}_scatter_plots.png')

