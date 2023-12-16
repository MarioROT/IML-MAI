import numpy as np
from sklearn.metrics import euclidean_distances
from scipy.stats import norm
import numpy as np
import pandas as pd
from KIBL import KIBL
from utils.data_preprocessing import Dataset



class IB3:
    def __init__(self, confidence_accept=0.9, confidence_drop=0.7):
        self.S = []  # Set of stored instances
        self.confidence_accept = confidence_accept
        self.confidence_drop = confidence_drop
        self.class_freq = {}  # Frequency of each class
        self.total_instances = 0

    def fit(self, X, y):
        for instance, label in zip(X, y):
            self.total_instances += 1
            self.class_freq[label] = self.class_freq.get(label, 0) + 1

            nearest_instance, nearest_label = self.find_nearest_acceptable_instance(instance)
            if nearest_label != label:
                self.S.append({'instance': instance, 'label': label, 'correct': 0, 'total': 0})

            for stored_instance in self.S:
                if self.is_at_least_as_close(instance, stored_instance['instance'], nearest_instance):
                    stored_instance['total'] += 1
                    if stored_instance['label'] == label:
                        stored_instance['correct'] += 1

            self.S = [s for s in self.S if self.is_acceptable(s)]

    def predict(self, X):
        predictions = []
        for instance in X:
            nearest_instance, nearest_label = self.find_nearest_acceptable_instance(instance)
            predictions.append(nearest_label)
        return predictions

    def find_nearest_acceptable_instance(self, instance):
        if not self.S:
            return None, None

        distances = euclidean_distances([instance], [s['instance'] for s in self.S])[0]
        nearest_index = np.argmin(distances)
        return self.S[nearest_index]['instance'], self.S[nearest_index]['label']

    def is_at_least_as_close(self, instance, candidate, nearest):
        if nearest is None:
            return True
        return np.linalg.norm(instance - candidate) <= np.linalg.norm(instance - nearest)


    def is_acceptable(self, instance):
        n = instance['total']
        if n == 0:
            return False

        p = instance['correct'] / n
        z_accept = norm.ppf(self.confidence_accept)
        z_drop = norm.ppf(self.confidence_drop)

        # Upper and lower bounds for acceptability and dropping
        lower_bound_accept = self.confidence_interval(p, n, z_accept)[0]
        upper_bound_drop = self.confidence_interval(p, n, z_drop)[1]

        class_freq = self.class_freq[instance['label']] / self.total_instances
        upper_bound_class = self.confidence_interval(class_freq, self.total_instances, z_accept)[1]
        lower_bound_class = self.confidence_interval(class_freq, self.total_instances, z_drop)[0]

        return lower_bound_accept > upper_bound_class or p > upper_bound_drop < lower_bound_class

    def confidence_interval(self, p, n, z):
        # Calculating the confidence interval
        factor = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
        lower = (p + z ** 2 / (2 * n) - factor) / (1 + z ** 2 / n)
        upper = (p + z ** 2 / (2 * n) + factor) / (1 + z ** 2 / n)
        return lower, upper
        
    def remove_low_confidence_instances(self):
        """Remove instances with low confidence from the stored instances."""
        self.S = [instance for instance in self.S if self.is_acceptable(instance)]



def preprocess_with_IB3(X, y):
    ib3 = IB3()
    ib3.fit(X, y)
    X_refined, y_refined = zip(*[(s['instance'], s['label']) for s in ib3.S])
    return np.array(X_refined), np.array(y_refined)


# Load data
data = Dataset('../data/folded/Nueva carpeta/pen-based', cat_transf='onehot', folds=True)

train, test = data[0]

# Separate features and labels
X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
print("X_train",X_train)
# Apply this to your dataset
X_refined, y_refined = preprocess_with_IB3(X_train.values, y_train.values)

print("X_refined",X_refined)

# Initialize and train the IB3 model
#ib3 = IB3()
#ib3.fit(X_train.values, y_train.values)

# Remove low confidence instances
#ib3.remove_low_confidence_instances()

# Make predictions on the test set
#predictions = ib3.predict(X_test.values)
#print("predictions",predictions)





