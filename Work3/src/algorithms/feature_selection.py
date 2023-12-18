import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, chi2
from sklearn_relief import ReliefF

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
                   'IG': self.information_gain,
                   'CR': self.correlation,
                   'C2S': self.chi_square_statistic,
                   'VT': self.variance_threshold,
                   'MI': self.mutual_info_cls,
                   'C2': self.chi2_skl,
                   'RF':self.reliefF_sk
                   }
        self.method = methods[method]
        if selection == 'nonzero' or isinstance(selection, int):
            self.selection = selection
        elif isinstance(selection, str) and '%' in selection:
            self.selection = round(features.shape[1] * int(selection[:-1]) / 100)

        self.selection = 'nonzero' if method in ['ones', 'variance_threshold'] else self.selection
        self.method_params = method_params if method_params else {}

    def compute_weights(self):
        scored_feats = self.method(self.features, self.labels,**self.method_params)
        if self.selection == 'nonzero':
            return [1 if v>0 else 0 for v in scored_feats.values()]
        else:
            feats_ranked = dict(sorted(scored_feats.items(), key=lambda item:item[1])[::-1][:self.selection])
            return [1 if k in feats_ranked.keys() else 0 for k in scored_feats.keys()]

    def reliefF_sk(self, features, labels):
        rel = ReliefF(n_features = self.selection)
        sel_feats = rel.fit_transform(features, labels)
        return {k:1 if features[k].values.tolist() in self.T.tolist() else 0 for k in features.columns}
    
    @staticmethod
    def ones(features, labels):
        return dict(zip(features.columns,np.ones_like(features.iloc[0])))
        
    @staticmethod
    def correlation(features, labels):
        weights = abs(pd.concat([features,labels], axis=1).corr()['y_true'][:-1].values) 
        return {k:v for k,v in zip(features.columns, weights)}

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

    @staticmethod
    def chi_square_statistic(features, labels):
        """Calculate Chi-square statistic for a feature in a dataset."""

        chi2s = {}

        for feature in features.columns:
            contingency_table = pd.crosstab(features[feature], labels)
            # Observed frequencies
            observed_frequencies = contingency_table.values
            # Chi-square statistic, p-value, degrees of freedom, and expected frequencies
            chi2v, p, dof, expected = chi2_contingency(observed_frequencies)

            chi2s[features] = chi2v
        
        return chi2

    @staticmethod
    def variance_threshold(features, labels):
        selector = VarianceThreshold()
        selector.fit(features.values)
        choosen = selector.get_feature_names_out(features.columns)
        return {k:1 if k in choosen else 0 for k in features.columns}

    @staticmethod
    def mutual_info_cls(features, labels):
        infs = mutual_info_classif(features.values, labels.values)
        return {k:v for k,v in zip(features.columns, infs)}

    @staticmethod
    def chi2_skl(features, labels):
        stats = chi2(features.values, labels.values)
        return {k:v for k,v in zip(features.columns, stats)}






