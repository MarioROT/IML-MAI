import sys
sys.path.append('../')
from itertools import product
from evaluation.metrics import params_grid_eval


def BestParamsSearch(method, params_grid, X, y, sort_order = ['accuracy'], name=None):
    performance = params_grid_eval(X, y, sort_order, name)
    param_groups = {}
    best_params = {}
    best_num_clusters = 0
    # Perform grid search
    for n, params in enumerate(product(*params_grid.values())):
        param_dict = dict(zip(params_grid.keys(), params))
        algorithm = method(**param_dict)
        algorithm.fit(X)
        predictions = algorithm.predict(X)
        performance.add_params_group(f'group {n}', predictions)
        param_groups[f'group {n}'] = param_dict

    results = performance.process_results()
    best_score = results.iloc[0,1:]
    best_params = param_groups[results.iloc[0,0]]
    return best_params, best_score
