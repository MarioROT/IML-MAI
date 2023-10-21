import argparse
import timeit

from utils.data_preprocessing import Dataset
from algorithms.kmeans import KMeans
from algorithms.kmodes import KModes
from algorithms.kprototypes import KPrototypes
from algorithms.DBSCAN import DBSCAN_Clustering
from algorithms.BIRCH import BIRCH_Clustering
from evaluation.metrics import performance_eval

# Arguments parser from terminal
parser = argparse.ArgumentParser()

parser.add_argument("-ds", "--datasets", nargs='+', help = "['iris', 'vowel', 'cmc']", default=['iris', 'cmc', 'vowel'])
parser.add_argument("-ags", "--algorithms", nargs='+', help = "['kmeans', 'kmodes', 'kprot', 'dbscan', 'birch']", default=['kmeans', 'kmodes', 'kprot', 'dbscan', 'birch'])

args = parser.parse_args()

# Configurations
algorithms = {'kmeans':KMeans,
              'kmodes':KModes,
              'kprot': KPrototypes,
              'dbscan': DBSCAN_Clustering,
              'birch': BIRCH_Clustering}

algorithm_params = {'kmeans':{'k':3},
                    'kmodes':{'k':3},
                    'kprot':{'k':3},
                    'dbscan': {'eps':5, 'min_samples':20, 'metric':'euclidean'},
                    'birch': {'threshold': 1, 'branching_factor': 20}}

for dataset in args.datasets:
    
    data = Dataset(f'../data/raw/{dataset}.arff')
    X = data.processed_data.iloc[:,:-1].values
    Y = data.processed_data.iloc[:,-1].values

    print(f'\n------- Results for {dataset} dataset:')
    for agm in args.algorithms:
        if agm in ['kmeans', 'kmodes', 'kprot']:
            algorithm = algorithms[agm](**algorithm_params[agm])
            algorithm.fit(X)
            predictions = algorithm.predict(X)
            print(f'\n-- Algorithm {agm}')
            performance_eval(predictions, Y)

        if agm == 'dbscan':
            algorithm = algorithms[agm](X, **algorithm_params[agm])
            if best_params:
                best_params, best_score, best_num_clusters = algorithm.grid_search()
                algorithm_params[agm] = best_params
            centroids, clusters = algorithm.dbscan_clustering(**algorithm_params[agm]) 
            print(f'\n-- Algorithm {agm}')
            performance_eval(clusters, Y)

        if agm == 'brich':
            algorithm = algorithms[agm](X, **algorithm_params[agm])
            if best_params:
                best_params, best_score, best_num_clusters = algorithm.grid_search()
                algorithm_params[agm] = best_params
            centroids, clusters = algorithm.birch_clustering(**algorithm_params[agm]) 
            print(f'\n-- Algorithm {agm}')
            performance_eval(clusters, Y)

