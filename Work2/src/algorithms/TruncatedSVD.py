import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from itertools import combinations
from utils.custom_plots import custom_grids

plt.rcParams["image.cmap"] = "tab20"
# Para cambiar el ciclo de color por defecto en Matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
#Set_ColorsIn(plt.cm.Set2.colors)
colors = plt.cm.tab20.colors


class SklearnTSVD():

    def __init__(self,
                 data: np.array,
                 data_name:str):
        self.data = data
        self.data_name = data_name
        self.Version = 'sklearn TruncatedSVD'
        self.labels = None
        self.transformed_data = None
        self.explained_variance_ratio = None

    def fit(self, num_components='best', algorithm='randomized', n_iter=5, n_oversamples=10, power_iteration_normalizer='auto', tol=0.0, threshold=85):
        if num_components != 'best':
            self.n_features = num_components
            tsvd = TruncatedSVD(num_components, algorithm=algorithm, n_iter=n_iter, n_oversamples=n_oversamples, power_iteration_normalizer=power_iteration_normalizer, tol=tol)
            self.transformed_data = tsvd.fit_transform(self.data)
            self.reconstructed_data=tsvd.inverse_transform(self.transformed_data)
            self.explained_variance_ratio = tsvd.explained_variance_ratio_
            return self.transformed_data
        else:
            self.transformed_data, self.num_components, self.explained_variance_ratio = self.find_best_n_components(self.Data, threshold=threshold)



    def find_best_n_components(self, X, threshold=85):
        """
        Find the optimal number of components for TruncatedSVD based on the explained variance ratio.

        Parameters:
        - X: Input data, a 2D array-like or sparse matrix.
        - threshold: The minimum cumulative explained variance ratio to achieve.

        Returns:
        - best_n_components: The optimal number of components.
        - explained_variances: List of explained variances for each number of components.
        """
        if X.ndim != 2:
            raise ValueError("Input data should be a 2D array-like or sparse matrix.")

        explained_variances = []
        n_features = X.shape[1]
        n_components_range = range(1, n_features + 1)
        best_n_components = None

        for n_components in n_components_range:
            svd = TruncatedSVD(n_components=n_components)
            X_transformed = svd.fit_transform(X)
            explained_variances.append(np.sum(svd.explained_variance_ratio_) * 100)

            if np.sum(svd.explained_variance_ratio_) * 100 >= threshold:
                best_n_components = n_components
                break

        print(f"Transformed data shape: ({X.shape[0]}, {best_n_components}) captures {explained_variances[best_n_components-1]:.2f}% of total "
              f"variation (TruncatedSVD)")

        return X_transformed, best_n_components, explained_variances

    def visualize(self, labels, axes=[0, 1, 2, 3], figsize=(10, 10), data2plot='Original', exclude = [], layout=None, axis=None, title_size = 12, save=None):
        data = {'Original':self.data, 'Transformed':self.transformed_data, 'Reconstructed':self.reconstructed_data}[data2plot]
        title = data2plot + ' Data'  

        plots ={}
        for comb in [com for sub in range(1,4) for com in combinations(axes, sub + 1)]:
            if str(len(comb))+'d' not in plots.keys():
                plots[str(len(comb))+'d'] = []
            plots[str(len(comb))+'d'].append(comb)

        plots['scree'] = [(True)]  
        plots = {k:v for k,v in plots.items() if k not in exclude}

        layout = [len(plots), max(len(l) for l in plots.values())] if not layout else layout

        cg = custom_grids([],layout[0], layout[1], figsize=figsize, axis=axis, title_size=title_size, use_grid_spec = False)
        cg.show()

        for k,group in plots.items():
            for i,v in enumerate(group):
                if k in ['2d', '3d', '4d']:
                    if k == '2d':
                        ax = cg.add_plot(title+ ' ' + self.Version, clear_ticks=True, axlabels=v)
                        ax.scatter(data[:, v[0]], data[:, v[1]], c=labels)
                    else: 
                        ax = cg.add_plot(title+ ' ' + self.Version,projection=True, clear_ticks=True, axlabels=v, row_last= True if i == len(group)-1 else False)
                        ax.scatter(data[:, v[0]], data[:, v[1]], data[:, v[2]], c=labels, s=15 if k == '3d' else data[:, v[3]] * 10)
                elif k == 'scree':
                    p = cg.add_plot('Explained Variance {}'.format(self.Version), axlabels=['Number of Components','Variance (%)'], last=True)
                    p.plot(np.cumsum(self.explained_variance_ratio), marker='.', color=colors[1])
                    p.bar(list(range(0, self.n_features)), self.explained_variance_ratio, color=colors[2])
        plt.show()

        if save:
            plt.savefig(save)
