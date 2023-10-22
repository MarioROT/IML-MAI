import argparse
import timeit

from utils.data_preprocessing import Dataset
from utils.best_params_search import BestParamsSearch
from algorithms.kmeans import KMeans
from algorithms.kmodes import KModes
from algorithms.kprototypes import KPrototypes
from algorithms.fcm_py import FCM
from algorithms.DBSCAN import DBSCANClustering
from algorithms.BIRCH import BIRCHClustering
from sklearn.cluster import DBSCAN, Birch
from evaluation.metrics import performance_eval

# Arguments parser from terminal
parser = argparse.ArgumentParser()

parser.add_argument("-ds", "--datasets", nargs='+', help = "['iris', 'vowel', 'cmc']", default=['iris', 'cmc', 'vowel'])
parser.add_argument("-ags", "--algorithms", nargs='+', help = "['kmeans', 'kmodes', 'kprot','fcm', 'dbscan', 'birch']", default=['kmeans', 'kmodes', 'kprot','fcm', 'dbscan', 'birch'])
parser.add_argument("-bp", "--best_params", help = "[True,False]", default=True, type=bool)
parser.add_argument("-dsm", "--dataset_method", help = "['numeric', 'categorical', 'mixed']", default='numeric', type=str)
parser.add_argument("-ce", "--cat_encoding", help = "['onehot', 'ordinal']", default='onehot', type=str)

args = parser.parse_args()

# Settings
algorithms = {'kmeans':KMeans,
              'kmodes':KModes,
              'kprot': KPrototypes,
              'fcm': FCM,
              'dbscan': DBSCANClustering,
              'birch': BIRCHClustering}

# algorithm_params = {'kmeans':{'k':3},
#                     'kmodes':{'k':3},
#                     'kprot':{'k':3},
#                     'fcm':{'C':2},
#                     'dbscan': {'eps':5, 'min_samples':20, 'metric':'euclidean'},
#                     'birch': {'threshold': 1, 'branching_factor': 20}}

algorithm_params = {'kmeans':{'k':[3,5,7]},
                    'kmodes':{'n_clusters':[3,5,7]},
                    'kprot':{'k':[3,5,7]},
                    'fcm':{'C':[3,5,7]},
                    'dbscan': {'eps':5, 'min_samples':20, 'metric':'euclidean'},
                    'birch': {'threshold': 1, 'branching_factor': 20}}

# Algoithms execution over datasets
for dataset in args.datasets:
    data = Dataset(f'data/raw/{dataset}.arff', method=args.dataset_method, cat_transf=args.cat_encoding)
    X = data.processed_data.iloc[:,:-1].values
    Y = data.processed_data.iloc[:,-1].values

    print(f'\n------- Results for {dataset} dataset:')
    for agm in args.algorithms:
        if agm in ['kmeans', 'kmodes', 'kprot','fcm']:
            if args.best_params:
                algorithm_params[agm] = BestParamsSearch(algorithms[agm], algorithm_params[agm], X, Y, ['accuracy'], f' {agm}')[0]
                print(f'--- Best params: {algorithm_params[agm]}')
            else:
                algorithm = algorithms[agm](**algorithm_params[agm])
                algorithm.fit(X)
                predictions = algorithm.predict(X)
                print(f'\n-- Algorithm {agm}')
                performance_eval(X, predictions, Y)

        if agm in ['dbscan', 'birch']:
            if args.best_params:
                algorithm = algorithms[agm](X, Y)
                algorithm.search_best_params()
                algorithm.print_best_params()
            else:
                algorithm = Birch if agm == 'birch' else DBSCAN
                algorithm = algorithm(**algorithm_params[agm])
                predictions = algorithm.fit_predict(X)
                print(f'\n-- Algorithm {agm}')
                performance_eval(X, predictions, Y)