import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from utils.custom_plots import custom_grids
from sklearn.manifold import Isomap

plt.rcParams["image.cmap"] = "tab20"
# Para cambiar el ciclo de color por defecto en Matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
#Set_ColorsIn(plt.cm.Set2.colors)
colors = plt.cm.tab20.colors


class SklearnIsoMap():

    def __init__(self,
                 data: np.array,
                 data_name:str):
        self.data = data
        self.data_name = data_name
        self.Version = 'sklearn IsoMap'
        self.labels = None
        self.transformed_data = None
        self.explained_variance_ratio = None

    def fit(self, n_components=2, eigen_solver='auto', max_iter=None, neighbors_algorithm='auto', metric='minkowski',p=2):
            self.n_features = n_components
            isomap = Isomap(n_components=n_components, eigen_solver=eigen_solver, max_iter=max_iter, neighbors_algorithm=neighbors_algorithm, metric=metric,p=p)
            self.transformed_data = isomap.fit_transform(self.data)
            return self.transformed_data


    def visualize(self, labels, axes=[0, 1, 2, 3], title='', figsize=(10, 10), data2plot='Original', exclude = [], layout=None, axis=None, title_size = 12, save=None):
        data = {'Original':self.data, 'Transformed':self.transformed_data}[data2plot]
        title = title + ' - ' + data2plot + ' Data ' + self.Version  

        plots ={}
        for comb in [com for sub in range(1,4) for com in combinations(axes, sub + 1)]:
            if str(len(comb))+'d' not in plots.keys():
                plots[str(len(comb))+'d'] = []
            plots[str(len(comb))+'d'].append(comb)
 
        plots = {k:v for k,v in plots.items() if k not in exclude}

        layout = [len(plots), max(len(l) for l in plots.values())] if not layout else layout

        cg = custom_grids([],layout[0], layout[1], title, figsize=figsize, axis=axis, title_size=title_size, use_grid_spec = False)
        if tuple(exclude) not in  permutations(['4d', '2d', '3d'],r=3):
            cg.show()

        for k,group in plots.items():
            for i,v in enumerate(group):
                if k in ['2d', '3d', '4d']:
                    if k == '2d':
                        ax = cg.add_plot(clear_ticks=True, axlabels=v)
                        ax.scatter(data[:, v[0]], data[:, v[1]], c=labels)
                    else: 
                        ax = cg.add_plot(projection=True, clear_ticks=True, axlabels=v, row_last= True if i == len(group)-1 else False)
                        ax.scatter(data[:, v[0]], data[:, v[1]], data[:, v[2]], c=labels, s=15 if k == '3d' else data[:, v[3]] * 10)
        plt.show()

        if save:
            plt.savefig(save)
