import timeit
from utils.data_preprocessing import Dataset
from evaluation.metrics import performance_eval
import argparse
from KIBL import KIBL
from instance_selection import InstanceSelection
from utils.StatTest import Friedman_Nem, process_results, avg_rank
import numpy as np
import pandas as pd
from utils.best_params_search import BestParamsSearch
from datetime import datetime
import os


# Arguments parser from terminal
parser = argparse.ArgumentParser()

parser.add_argument("-bp", "--best_params", help = "[True,False]", default=True, type=bool)
parser.add_argument("-ds", "--datasets", nargs='+', help = "['vowel', 'kr-vs-kp']", default='kr-vs-kp', type=str)
parser.add_argument("-k", "--nearest_neighbors", help = "[3, 5, 7]", default=3, type=int)
parser.add_argument("-vot", "--voting", nargs='+', help = "['MP':Modified_Plurality',''BC'Borda_Count']", default='MP', type=str)
parser.add_argument("-ret", "--ret_policy", nargs='+', help = "['NR':Never_Retain,'AR':Always_Retain,'DF':Different Class Ret,'DD':Degree disagreement]", default='NR',type=str)
parser.add_argument("-fs", "--feature_selection", help = "['Ones', 'CR':Correlation, 'IG':Information Gain,'C2S':Chi Square Stat, 'VT':Variance Treshold, 'MI':Mutual Inf.,'C2': ChiSq. SKL, 'RF': Relief]", default='Ones', type=str)
parser.add_argument("-is", "--instance_selection", help = "['MCNN':Modif. Cond NN, 'ENN':Edited NNR, 'IBL3']", default='MCNN', type=str)

args = parser.parse_args()


algorithm_params = {'BestParamsSearch':{'ds':['pen-based', 'vowel', 'kr-vs-kp'],#['pen-based', 'vowel', 'kr-vs-kp']
                                       'K':[3,5,7],
                                       'voting': ['MP', 'BC'],
                                       'retention':['NR', 'AR', 'DF', 'DD']},
                    'Custom':{'ds':args.datasets,
                              'K':args.nearest_neighbors,
                              'voting':args.voting,
                              'retention':args.ret_policy}}

agm = 'BestParamsSearch' if args.best_params else 'Custom'

parameters=BestParamsSearch(algorithm_params[agm])

results = pd.DataFrame(columns=['params','folds','accuracies','efficiencies','total_times'])
name_date=str(datetime.now()).replace(" ","_").replace(":","_")
save_in = f'Work3/results/Experiment_{agm}_{name_date}/'
os.mkdir(save_in)

for k,params in parameters.items():
    data = Dataset(f"../data/folded/{params['ds']}",cat_transf='ordinal', folds=True)
    for i, (train, test) in enumerate(data):
        if 'ds' in params.keys():
            params.pop('ds')
        
        IBL=KIBL(train,**params)
        accuracy, efficiency, total_time = IBL.kIBLAlgorithm(test)

        res = {'params':[k],
               'folds':[i],
               'accuracies':[accuracy],
               'efficiencies':[efficiency],
               'total_times':[total_time]}
        
        results = pd.concat([results, pd.DataFrame(res)], ignore_index = True)
        results.to_csv(save_in + f'Results_KIBL_{name_date}.csv', index=False)
    
        print(f'Fold: {i}  Acc:{accuracy}  Ef:{efficiency} Time {total_time}' )
        print('Data has been stored')
    
acc, eff, times =process_results(results)
Acc_Matrix=Friedman_Nem(acc, title=[save_in, 'Accuracies'])
if Acc_Matrix is not None: 
    acc_ranks=avg_rank(acc,title=[save_in,'Accuracies' ],reverse=True)
Eff_Matrix= Friedman_Nem(eff, title=[save_in, 'Efficiencies'])
if Eff_Matrix is not None:
    eff_ranks=avg_rank(eff,title=[save_in, 'Efficiencies']) 
TTime_Matrix= Friedman_Nem(times, title=[save_in,'TTimes'])
if TTime_Matrix is not None:
    time_ranks=avg_rank(times,title=[save_in, 'Efficiencies'])




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