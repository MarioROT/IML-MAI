"""
3. Analysis three data sets using PCA and IncrementalPCA from sklearn
"""

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import available_if
import matplotlib.pyplot as plt
from utils.custom_plots import custom_grids
from itertools import combinations, permutations

plt.rcParams["image.cmap"] = "tab20"
# Para cambiar el ciclo de color por defecto en Matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
#Set_ColorsIn(plt.cm.Set2.colors)
colors = plt.cm.tab20.colors


class SklearnPCA():

    def __init__(self,
               data: np.array,
               data_name: str):
        self.data = data
        self.data_name = data_name
        self.n_features = data.shape[1]
        self.eigenvalues = None
        self.eigenvectors = None
        self.transformed_data = None
        self.explained_variance_ratio = None
        self.reconstructed_data = None
        self.Version = None

    def PCA(self, num_components=None):
        self.n_features=num_components
        self.Version='sklearn PCA'
        x = self.data

        pca = PCA(num_components)
        self.transformed_data = pca.fit_transform(x)

        # Explained Variance = eigenvalues
        self.eigenvalues = pca.explained_variance_

        # Eigenvectors
        self.eigenvectors = pca.components_

        self.explained_variance_ratio = pca.explained_variance_ratio_

        # Reconstructed data
        self.reconstructed_data = pca.inverse_transform(self.transformed_data)

        print(f"Transformed data shape: ({pca.n_samples_}, {pca.n_components_}) captures {sum(pca.explained_variance_ratio_) * 100:0.2f}% of "
        f"total variation (Sklearn PCA)")

        return self.transformed_data

    def iPCA(self, num_components):
        self.n_features=num_components
        self.Version='sklearn Incremental PCA'
        x = self.data

        ipca = IncrementalPCA(num_components)
        self.transformed_data = ipca.fit_transform(x)

        # Explained Variance = eigenvalues
        self.eigenvalues = ipca.explained_variance_

        # Eigenvectors
        self.eigenvectors = ipca.components_

        self.explained_variance_ratio = ipca.explained_variance_ratio_

        # Reconstructed Data
        self.reconstructed_data = ipca.inverse_transform(self.transformed_data)

        print(f"Transformed data shape: ({X_pca.n_samples_seen_}, {X_pca.n_components_}) captures {sum(ipca.explained_variance_ratio_)*100:0.2f}"
        f"% of total variation (Sklearn IncrementalPCA)")

        return self.transformed_data

    def visualize(self, labels, axes=[0, 1, 2, 3], title='', figsize=(10, 10), data2plot='Original', exclude = [], layout=None, axis=None, title_size = 12, save=None):
        data = {'Original':self.data, 'Reconstructed':self.reconstructed_data, 'Transformed':self.transformed_data}[data2plot]
        title = title + ' - ' + data2plot + ' Data ' + self.Version  

        plots ={}
        for comb in [com for sub in range(1,4) for com in combinations(axes, sub + 1)]:
            if str(len(comb))+'d' not in plots.keys():
                plots[str(len(comb))+'d'] = []
            plots[str(len(comb))+'d'].append(comb)

        if data2plot == 'Transformed':
            plots['scree'] = [(True)]  
             
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
                elif k == 'scree' and data2plot=='Transformed':
                    cg = custom_grids([],1, 1, use_grid_spec=False)
                    cg.show()
                    p = cg.add_plot('Explained Variance {}'.format(self.Version), axlabels=['Number of Components','Variance (%)'], last=True)
                    p.plot(np.cumsum(self.explained_variance_ratio), marker='.', color=colors[1])
                    p.bar(list(range(0, self.n_features)), self.explained_variance_ratio, color=colors[2])
        plt.show()

        if save:
            plt.savefig(save)
        
    
