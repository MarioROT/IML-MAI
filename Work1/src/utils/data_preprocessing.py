from typing import Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder
from scipy.io import arff

class Dataset():
    """"Class aimed to preprocess datasets comming from arff files, you can access to each 
    datapoint and its corresponding label using the class instantiation. The preprocessing
    method reads the raw data from the original files and separate the features, the labes 
    and the metadata. It also assings an ID (numeric value) to each class. Then, it evaluates
    if there are missing values on the data, and run an standarization process where data
    imputations, scale adjustments, and label encoding are made if necessary. General statistics 
    of the dataset can be obtained by calling the statstics method. The processed data can be
    save using te save method."""
    def __init__(self, 
                 data_path: str,
                 with_mean: bool = True, 
                 with_std: bool = True,
                 method: str = 'numerical',
                 cat_transf: str = 'onehot'):
        self.data_path = Path(data_path)
        self.wmean = with_mean
        self.wstd = with_std
        self.method = method
        self.cat_transf = cat_transf
        self.raw_data = None
        self.metadata = None
        self.df = None 
        self.y_true = None
        self.processed_data = None
        self.classes_relation = None

        self.preprocessing()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.processed_data.iloc[idx,:-1], self.processed_data.iloc[idx,-1]

    def import_raw_dataset(self):
        data, self.metadata = arff.loadarff(self.data_path)
        data = pd.DataFrame(data)
        self.raw_data = self.string_decode(data)
        return self.raw_data, self.metadata

    def preprocessing(self):
        print(f'---Preprocessing {self.data_path.name} dataset---')
        self.df, meta = self.import_raw_dataset()
        num_samples_initial, num_features_initial = self.df.shape
        self.y_true = self.get_predicted_value(self.df)
        self.classes_relation = {k: v for v, k in enumerate(set(self.y_true))}
        self.df = self.remove_predicted_value(self.df)
        #If dataset vowels
        if "Train_or_Test" in self.df.columns:
            self.df = self.df.drop(columns=["Train_or_Test"])

        nulls = self.check_null_values(self.df)
        self.processed_data = self.standardization(self.df, self.method, self.cat_transf, self.wmean, self.wstd, len(self.y_true.unique()))
        num_samples_final, num_features_final = self.processed_data.shape
        self.processed_data['y_true'] = self.encode_labels(self.y_true, self.classes_relation)
        print(f"Initial Dataset: {num_samples_initial} samples, {num_features_initial} features")
        print(f"Final Dataset: {num_samples_final} samples, {num_features_final} features")
        return self.processed_data

    def save(self, filename, dir = ''):
        self.processed_data.to_csv(dir + filename + '.csv')

    def statistics(self, data_type):
        data = self.raw_data.iloc[:,:-1] if data_type == 'raw' else self.processed_data.iloc[:,:-1]
        labels = self.raw_data.iloc[:,-1] if data_type == 'raw' else self.processed_data.iloc[:,-1]
        gen_stats = {'n_classes': len(set(labels)),
                     'n_features': len(data.columns),
                     'n_instances': len(data)}
        stats = {'Nulls':data.isnull().sum(0).values,
                 'Min': data.min().values,
                 'Max': data.max().values,
                 'Mean': data.mean().values,
                 'StD': data.std().values,
                 'Variance': data.var().values}
        stats = pd.DataFrame.from_dict(stats,orient = 'index', columns=data.columns)
        return gen_stats, stats
        
    @staticmethod
    def string_decode(df: pd.DataFrame):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')
        return df
    @staticmethod
    def remove_predicted_value(df: pd.DataFrame):
        return df.iloc[:, :-1]
    @staticmethod
    def get_predicted_value(df: pd.DataFrame):
        return df.iloc[:, -1]
    @staticmethod
    def check_null_values(df: pd.DataFrame):
        return df.isnull().sum()
    @staticmethod
    def encode_labels(labels, classes_relation):
        num_classes = [classes_relation[item] for item in labels]
        return num_classes
    @staticmethod
    def standardization(df: pd.DataFrame, method, cat_transf, wmean=True, wstd=True, num_cat=2):
        # numerical features
        num_features = df.select_dtypes(include=np.number).columns
        cat_features = df.select_dtypes(exclude=np.number).columns

        if method == 'numerical':
            # numerical features
            num_transformer = Pipeline(steps=[
                ('replace_nan', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler(with_mean=wmean,with_std=wstd))])
            # categorical features
            if cat_transf == 'onehot':
                cat_transformer = Pipeline(steps=[
                    ('replace_nan', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder())])
            elif cat_transf == 'ordinal':
                cat_transformer = Pipeline(steps=[
                    ('replace_nan', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OrdinalEncoder())])
        elif method == 'categorical':
            # numerical features
            num_transformer = Pipeline(steps=[
                ('replace_nan', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler(with_mean=wmean,with_std=wstd)),
                ('discretizer', KBinsDiscretizer(num_cat, encode='ordinal'))])
            # categorical features
            cat_transformer = Pipeline(steps=[
                ('replace_nan', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())])
        elif method == 'mixed':
            num_transformer = Pipeline(steps=[
                ('replace_nan', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler(with_mean=wmean,with_std=wstd))])
            # categorical features
            cat_transformer = Pipeline(steps=[
                ('replace_nan', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder())])

        # transform columns
        ct = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)])
        X_trans = ct.fit_transform(df)


        # dataset cases
        # case 1: categorical and numerical features
        if len(cat_features) != 0 and len(num_features) != 0:
            columns_encoder = ct.transformers_[1][1]['encoder']. \
                get_feature_names_out(cat_features)
            columns = num_features.union(pd.Index(columns_encoder), sort=False)
        # case 2: only categorical features
        elif len(cat_features) != 0 and len(num_features) == 0:
            columns = ct.transformers_[1][1]['encoder']. \
                get_feature_names_out(cat_features)
            columns = pd.Index(columns)

        # case 3: only numerical features
        elif len(cat_features) == 0 and len(num_features) != 0:
            columns = num_features
        else:
            print('There is a problem with features')

        # processed dataset
        processed_df = pd.DataFrame(X_trans, columns=columns)
        return processed_df

if __name__ == '__main__':
    data_path = Path('../data/raw/iris.csv')
    dataset = Dataset(data_path)
