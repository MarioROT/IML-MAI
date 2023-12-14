import pandas as pd
import numpy as np


class FeatureSelection():

    def __init__(self,
                 features: pd.DataFrame,
                 labels: pd.Series,
                 method: str,
                 selection = 'nonzero', 
                 method_params = False):
        
        self.features = features
        self.labels = labels
        methods = {'ones':self.ones,
                   'information_gain': self.information_gain,
                   'correlation': self.correlation}
        self.method = methods[method]
        if selection == 'nonzero' or isinstance(selection, int):
            self.selection = selection
        elif isinstance(selection, str) and '%' in selection:
            self.selection = round(features.shape[1] * int(selection[:-1]) / 100)

        self.selection = 'nonzero' if method in ['ones'] else self.selection
        self.method_params = method_params if method_params else {}

    def compute_weights(self):
        scored_feats = self.method(self.features, self.labels,**self.method_params)
        if self.selection == 'nonzero':
            return [1 if v>0 else 0 for v in scored_feats.values()]
        else:
            feats_ranked = dict(sorted(scored_feats.items(), key=lambda item:item[1])[::-1][:self.selection])
            return [1 if k in feats_ranked.keys() else 0 for k in scored_feats.keys()]

    @staticmethod
    def ones(features, labels):
        return dict(zip(features.columns,np.ones_like(features.iloc[0])))
        
    @staticmethod
    def correlation(features, labels):
        return abs(pd.concat([features,labels], axis=1).corr()['y_true'][:-1].values) 
        # elif method == 'information_gain':
        #     return mutual_info_classif(data, np.squeeze(labels))

    @staticmethod
    def information_gain(features, labels):
        """Calculate Information Gain for a feature in a dataset."""
        total_entropy = FeatureSelection.entropy(labels)
        features['y_true'] = labels
        inf_gains = {}

        for feature in np.delete(features.columns, np.where(features.columns == 'y_true')):
            values = features[feature].unique()
            
            weighted_entropy = 0
            for value in values:
                subset = features[features[feature] == value]
                weighted_entropy += len(subset) / len(features) * FeatureSelection.entropy(subset['y_true'])
            
            inf_gains[feature] = total_entropy - weighted_entropy
        return inf_gains

    @staticmethod
    def entropy(labels):
        """Calculate entropy for a set of labels."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy_value = -np.sum(probabilities * np.log2(probabilities))
        return entropy_value

