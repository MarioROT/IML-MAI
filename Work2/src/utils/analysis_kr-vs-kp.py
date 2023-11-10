import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns

# DATASET KR-VS-KP
DATASET = "kr-vs-kp"
data_path = f"../../data/raw/{DATASET}.arff"

# Load ARFF dataset and metadata
data, meta = arff.loadarff(data_path)

# Convert data to Pandas DataFrame
df = pd.DataFrame(data)

# Decode bytes to utf-8 if needed
df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Extract target variable and drop the "class" column
y_true = df["class"]
X = df.drop(columns="class")

num_classes = y_true.nunique()
attribute_names = meta.names()
num_samples, num_features = X.shape
statistics = X.describe()
class_counts = y_true.value_counts()

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
plt.savefig(f'../../Results/images/{DATASET}_classes.png')
plt.close()

# Heatmap for correlation between features and y_true
plt.figure(figsize=(15, 6))
heatmap_data = pd.crosstab(df['class'], [df['bxqsq'], df['reskr'], df['rimmx'], df['simpl']])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues')
plt.title('Correlation between features and y_true')
plt.savefig(f'../../Results/images/{DATASET}_original_features.png')
