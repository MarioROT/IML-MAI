import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from utils.data_preprocessing import Dataset
from utils.StatTest import Friedman_Nem, process_results, avg_rank
from utils.best_params_search import BestParamsSearch
from algorithms.KIBL import KIBL

# Arguments parser from terminal
parser = argparse.ArgumentParser()

parser.add_argument("-bp", "--experiment",
                    help="['BPS':BestParamsSearch, 'BFS':BestFeatureSelection, 'BIS':'BestInstanceSelection']",
                    default='BPS', type=str)
parser.add_argument("-ds", "--datasets", nargs='+', help="['pen-based','vowel', 'kr-vs-kp']",
                    default=['pen-based', 'vowel', 'kr-vs-kp'])
parser.add_argument("-k", "--nearest_neighbors", nargs='+', help="[3, 5, 7]", default=[3])
parser.add_argument("-vot", "--voting", nargs='+', help="['MP':Modified_Plurality,'BC':Borda_Count']", default=['MP'])
parser.add_argument("-ret", "--retention", nargs='+',
                    help="['NR':Never_Retain,'AR':Always_Retain,'DF':Different Class Ret,'DD':Degree disagreement]",
                    default=['NR'])
parser.add_argument("-fs", "--feature_selection", nargs='+',
                    help="['ones', 'CR':Correlation, 'IG':Information Gain,'C2S':Chi Square Stat, 'VT':Variance Treshold, 'MI':Mutual Inf.,'C2': ChiSq. SKL, 'RF': Relief]",
                    default=['ones'])
parser.add_argument("-kfs", "--k_fs", help="['nonzero', 'n%' -> e.g. '80%']", default=['80%'])
parser.add_argument("-is", "--instance_selection", nargs='+',
                    help="['None','MCNN':Modif. Cond NN, 'ENN':Edited NNR, 'IBL3']", default=['None'])

args = parser.parse_args()

experiment_params = {
    'BPS': {'ds': args.datasets, 'K': [3, 5, 7], 'voting': ['MP', 'BC'], 'retention': ['NR', 'AR', 'DF', 'DD']},
    'BFS': {'ds': args.datasets, 'K': args.nearest_neighbors, 'voting': args.voting, 'retention': args.retention,
            'feature_selection': ['ones','CR', 'IG', 'C2S', 'VT', 'MI', 'C2'], 'k_fs': args.k_fs},
    'BIS': {'ds': args.datasets, 'K': args.nearest_neighbors, 'voting': args.voting, 'retention': args.retention,
            'instance_selection': ['MCNN','ENN','IB3']},
    'Custom': {'fs': args.datasets, 'K': args.nearest_neighbors, 'voting': args.voting, 'retention': args.retention,
               'feature_selection': args.feature_selection, 'k_fs': args.k_fs,
               'instance_selection': args.instance_selection}
}

experiment = experiment_params[args.experiment]
parameters = BestParamsSearch(experiment)

results = pd.DataFrame(columns=['params', 'folds', 'accuracies', 'efficiencies', 'total_times'])
name_date = str(datetime.now()).replace(" ", "_").replace(":", "_")
save_in = f'../results/Experiment_{args.experiment}_{name_date}/'
os.mkdir(save_in)


for k, params in parameters.items():
    data = Dataset(f"../data/folded/{params['ds']}", cat_transf='onehot', folds=True)
    for i, (train, test) in enumerate(data):
        if 'ds' in params.keys():
            params.pop('ds')

        IBL = KIBL(train, save=save_in, **params)
        accuracy, efficiency, total_time = IBL.kIBLAlgorithm(test)

        res = {'params': [k],
               'folds': [i],
               'accuracies': [accuracy],
               'efficiencies': [efficiency],
               'total_times': [total_time]}

        results = pd.concat([results, pd.DataFrame(res)], ignore_index=True)
        results.to_csv(save_in + f'Results_KIBL_{name_date}.csv', index=False)

        print(f'Fold: {i}  Acc:{accuracy}  Ef:{efficiency} Time {total_time}')
        print('Data has been stored')

acc, eff, times = process_results(results)
Acc_Matrix = Friedman_Nem(acc, title=[save_in, 'Accuracies'])
if Acc_Matrix is not None:
    acc_ranks = avg_rank(acc, title=[save_in, 'Accuracies'], reverse=True)
Eff_Matrix = Friedman_Nem(eff, title=[save_in, 'Efficiencies'])
if Eff_Matrix is not None:
    eff_ranks = avg_rank(eff, title=[save_in, 'Efficiencies'])
TTime_Matrix = Friedman_Nem(times, title=[save_in, 'TTimes'])
if TTime_Matrix is not None:
    time_ranks = avg_rank(times, title=[save_in, 'Efficiencies'])





