import timeit
from utils.data_preprocessing import Dataset
from evaluation.metrics import performance_eval
import argparse
from KIBL import KIBL
from instance_selection import InstanceSelection
from utils.StatTest import Friedman_Nem
import numpy as np
import pandas as pd
from utils.best_params_search import BestParamsSearch

# Arguments parser from terminal
# parser = argparse.ArgumentParser()

# parser.add_argument("-bp", "--best_params", help = "[True,False]", default=True, type=bool)
# parser.add_argument("-ds", "--datasets", nargs='+', help = "[ 'vowel', 'kr-vs-kp']", default=[ 'vowel', 'kr-vs-kp'])
# parser.add_argument("-k", "--nearest_neighbors", help = "[3, 5, 7]", default=[3,5,7], type=int)
# parser.add_argument("-vot", "--voting", nargs='+', help = "['Modified_Plurality','Borda_Count']", default=['Modified_Plurality','Borda_Count'])
# parser.add_argument("-ret", "--ret_policy", nargs='+', help = "['Never_Retain','Always_Retain']", default=['Modified_Plurality','Borda_Count'])
# parser.add_argument("-fs", "--feature_selection", help = "[True,False]", default=False, type=bool)
# parser.add_argument("-is", "--instance_selection", help = "[True,False]", default=False, type=bool)

# args = parser.parse_args()

# parameters=[]

data = Dataset('../data/Dummy/', cat_transf='onehot', folds=True)

# agm = 'BestParamsSearch' if args.best_params else 'Custom'

# for (train,test) in data:    
        
        
# train,test=data[0]
train, test = pd.read_csv('../data/Dummy/eq_pen-based_300_train.csv'), pd.read_csv('../data/Dummy/eq_pen-based_80_test.csv')
# IBL= KIBL(X=train, K=3, weights_m = 'information_gain', k_weights = '80%')   
# IBL= KIBL(X=train, K=3)   
# accuracy, efficiency, total_time= IBL.kIBLAlgorithm(test)
# print(accuracy, efficiency, total_time)

iss = InstanceSelection(train, 3)
iss.mcnn_algorithm()


    # train,test=data[0]
    # IBL= KIBL(X=train, K=3)   
    # accuracy, efficiency, total_time= IBL.kIBLAlgorithm(test)
    # print(accuracy, efficiency, total_time)
# param_selection='K:'+str(agmK)
# parameters.append(param_selection)

# accuracies={}
# efficiencies={}

# data = Dataset('C:/Users/52556/Desktop/Alam/ALAM UNI y otros docs/IML-MAI/Work3/data/folded/Nueva carpeta/pen-based', cat_transf='onehot', folds=True)

# for (train,test) in data:   
#     IBL= KIBL(X=train, K=3)   
#     accuracy, efficiency, total_time= IBL.kIBLAlgorithm(test)
#     
#     if not parameters[0] in accuracies.keys():
#         accuracies[parameters[0]]=[accuracy]
#         efficiencies[parameters[0]]=[efficiency]
#     else:
#         accuracies[parameters[0]].append(accuracy)
#         efficiencies[parameters[0]].append(efficiency)
#     
#     print(f'Dataset: {dataset}  Fold: {i}  Acc:{accuracy}  Ef:{efficiency}' )
#     print('Data has been stored')
#     
#  
# print(accuracies)
# print(efficiencies)

# Acc_Matrix=Friedman_Nem(accuracies)
# Eff_Matrix= Friedman_Nem(efficiencies)







# # Settings

# algorithms = {'kmeans':KMeans,
#               'kmodes':KModes,
#               'kprot': KPrototypes,
#               'fcm': FCM,
#               'dbscan': DBSCANClustering,
#               'birch': BIRCHClustering}

# algorithm_params = {'kmeans':{'k':[3,7,9,11,13,15]},
#                     'kmodes':{'n_clusters':[3,7,9,11,13,15]},
#                     'kprot':{'k':[3,7,9,11,13,15]},
#                     'fcm':{'C':[2,3,5,7]},
#                     'dbscan': {'eps':5, 'min_samples':20, 'metric':'euclidean'},
#                     'birch': {'threshold': 1, 'branching_factor': 20}}

# # Algorithms execution over datasets
# for dataset in args.datasets:
#     data = Dataset(f'../data/raw/{dataset}.arff', method=args.dataset_method, cat_transf=args.cat_encoding)
#     X = data.processed_data.iloc[:,:-1].values
#     Y = data.processed_data.iloc[:,-1].values

#     print(f'\n------- Results for {dataset} dataset:')
#     for agm in args.algorithms:
#         if agm in ['kmeans', 'kmodes', 'kprot','fcm']:
#             if args.best_params:
#                 algorithm_params[agm] = BestParamsSearch(algorithms[agm], algorithm_params[agm], X, Y, ['accuracy'], [agm, dataset, args.dataset_method, args.random_seed])[0]
#                 print(f'--- Best params: {algorithm_params[agm]}')
#             else:
#                 algorithm = algorithms[agm](**algorithm_params[agm])
#                 algorithm.fit(X)
#                 predictions = algorithm.predict(X)
#                 print(f'\n-- Algorithm {agm}')
#                 performance_eval(X, predictions, Y)

#         if agm in ['dbscan', 'birch']:
#             if args.best_params:
#                 algorithm = algorithms[agm](X, Y)
#                 algorithm.search_best_params()
#                 algorithm.print_best_params()
#             else:
#                 algorithm = Birch if agm == 'birch' else DBSCAN
#                 algorithm = algorithm(**algorithm_params[agm])
#                 predictions = algorithm.fit_predict(X)
#                 print(f'\n-- Algorithm {agm}')
#                 performance_eval(X, predictions, Y)



# -------------------------------------------------------------------------------------------#

# parameters=[]

# data = Dataset('C:/Users/52556/Desktop/Alam/ALAM UNI y otros docs/IML-MAI/Work3/data/folded/Nueva carpeta/pen-based', cat_transf='onehot', folds=True)

#     K=[3,5,7]
#     Voting=['MP','BC']
#     Retention=['NR']
    
#     for neighbors in K: 

# for (train,test) in data:    
        
        
# train,test=data[0]
# IBL= KIBL(X=train, K=3, weights_m = 'information_gain', k_weights = '80%')   
# IBL= KIBL(X=train, K=3)   
# accuracy, efficiency, total_time= IBL.kIBLAlgorithm(test)
# print(accuracy, efficiency, total_time)

# iss = InstanceSelection(train, 3)
# iss.mcnn_algorithm()
    # train,test=data[0]
    # IBL= KIBL(X=train, K=3)   
    # accuracy, efficiency, total_time= IBL.kIBLAlgorithm(test)
    # print(accuracy, efficiency, total_time)
# param_selection='K:'+str(agmK)
# parameters.append(param_selection)