import sys
sys.path.append('../')
from itertools import product
from evaluation.metrics import params_grid_eval


def BestParamsSearch(method, params_grid, X, y, sort_order = ['accuracy'], data=None):
    performance = params_grid_eval(X, y, sort_order, data[0])
    param_groups = {}
    best_params = {}
    best_num_clusters = 0
    # Perform grid search
    for n, params in enumerate(product(*params_grid.values())):
        param_dict = dict(zip(params_grid.keys(), params))
        algorithm = method(**param_dict)
        algorithm.fit(X)
        predictions = algorithm.predict(X)
        performance.add_params_group(f'{param_dict}', predictions)
        param_groups[f'{param_dict}'] = param_dict

    results = performance.process_results()
    results.to_csv(f'../Results/Tables/results_{data[1]}_{data[0]}_{data[2]}_{data[3]}.csv', index=False)
    best_score = results.iloc[0,1:]
    best_params = param_groups[results.iloc[0,0]]
    return best_params, best_score
