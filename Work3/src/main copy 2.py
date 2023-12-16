# import timeit
# from utils.data_preprocessing import Dataset
# from evaluation.metrics import performance_eval
# import argparse
# from KIBL import KIBL
# from instance_selection import InstanceSelection
# from utils.StatTest import Friedman_Nem
# import numpy as np
# import sys
# sys.path.append('../')
from itertools import product

# # Arguments parser from terminal
# parser = argparse.ArgumentParser()

# parser.add_argument("-bp", "--best_params", help = "[True,False]", default=True, type=bool)
# parser.add_argument("-ds", "--datasets", nargs='+', help = "['vowel', 'kr-vs-kp']", default='kr-vs-kp', type=str)
# parser.add_argument("-k", "--nearest_neighbors", help = "[3, 5, 7]", default=[3,5,7], type=int)
# parser.add_argument("-vot", "--voting", nargs='+', help = "['MP':Modified_Plurality',''BC'Borda_Count']", default='MP', type=str)
# parser.add_argument("-ret", "--ret_policy", nargs='+', help = "['NR':Never_Retain,'AR':Always_Retain,'DF':Different Class Ret,'DD':Degree disagreement]", default='NR',type=str)
# parser.add_argument("-fs", "--feature_selection", help = "['Ones', 'CR':Correlation, 'IG':Information Gain,'C2S':Chi Square Stat, 'VT':Variance Treshold, 'MI':Mutual Inf.,'C2': ChiSq. SKL, 'RF': Relief]", default='Ones', type=str)
# parser.add_argument("-is", "--instance_selection", help = "['MCNN':Modif. Cond NN, 'ENN':Edited NNR, 'IBL3']", default='MCNN', type=str)

# args = parser.parse_args()


algorithm_params = {'BestParamSearch':{'ds':['pen-based', 'vowel', 'kr-vs-kp'],
                                       'k':[3,5,7],
                                       'vp': ['MP', 'BC'],
                                       'rp':['NR', 'AR', 'DF', 'DD']}}



def BestParamsSearch(params_grid):
    param_groups = {}
    # Perform grid search
    for n, params in enumerate(product(*params_grid.values())):
        param_dict = dict(zip(params_grid.keys(), params))
        param_groups[f'{param_dict}'] = param_dict
    return param_groups

param_groups = BestParamsSearch(algorithm_params['BestParamSearch'])

print(param_groups)
print(len(param_groups))