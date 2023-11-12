"""
3. Analysis three data sets using PCA and IncrementalPCA from sklearn
"""

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import available_if
import matplotlib.pyplot as plt
from utils.custom_plots import custom_grids
from itertools import combinations

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

        return self.transformed_data

    def incrementalPCA(self, num_components):
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

        return self.transformed_data

    def visualize(self, labels, axes=[0, 1, 2, 3], figsize=(10, 10), data2plot='Original', exclude = [], layout=None, axis=None, title_size = 12, save=None, ):
        data = {'Original':self.data, 'Reconstructed':self.reconstructed_data, 'Transformed':self.transformed_data}[data2plot]
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

        # values = []
        # for i in axes:
        #     values.append(data[:, i])
        # values = np.array(values).T
        # dims = len(axes)

        for k,group in plots.items():
            for i,v in enumerate(group):
                if k == '2d':
                    p = cg.add_plot(title+ ' ' + self.Version)
                    p.scatter(data[:, v[0]], data[:, v[1]], c=labels)
                    p.xaxis.set_ticklabels([])
                    p.yaxis.set_ticklabels([])
                    p.set_xlabel(v[0])
                    p.set_ylabel(v[1])
                elif k == '3d':
                    ax = cg.add_plot(projection=True, row_last= True if i == len(group)-1 else False)
                    ax.scatter(data[:, v[0]], data[:, v[1]], data[:, v[2]], c=labels, s=15)
                    ax.set_title(title+ ' ' + self.Version,fontsize=title_size)
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
                    ax.zaxis.set_ticklabels([])
                    ax.set_xlabel(v[0])
                    ax.set_ylabel(v[1])
                    ax.set_zlabel(v[2])
                elif k == '4d':
                    ax = cg.add_plot(projection=True, row_last= True if i == len(group)-1 else False)
                    ax.scatter(data[:, v[0]], data[:, v[1]], data[:, v[2]], c=labels, s=data[:, v[3]] * 10)
                    ax.set_title(title+ ' ' + self.Version,fontsize=title_size)
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])
                    ax.zaxis.set_ticklabels([])
                    ax.set_xlabel(v[0])
                    ax.set_ylabel(v[1])
                    ax.set_zlabel(v[2])
                elif k == 'scree':
                    p = cg.add_plot('Explained Variance {}'.format(self.Version), last=True)
                    p.plot(np.cumsum(self.explained_variance_ratio), marker='.', color=colors[1])
                    p.bar(list(range(0, self.n_features)), self.explained_variance_ratio, color=colors[2])
                    p.set_xlabel('Number of Components')
                    p.set_ylabel('Variance (%)')

        plt.show()

        if save:
            plt.savefig(save)
        
    def scree_plot(self, save=False):
        plt.plot(np.cumsum(self.explained_variance_ratio), marker='.', color=colors[1])
        plt.bar(list(range(0, self.n_features)), self.explained_variance_ratio, color=colors[2])
        plt.title('Explained Variance {}'.format(self.Version))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')

        if save:
            plt.savefig(save + 'scree_plot_{}.pdf'.format(self.data_name))

        plt.show()


    # @staticmethod
    # def scatter_4D(data, labels, axes, title, data_name, figsize=(10, 10), save=None):
    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=data[:, 3] * 10)
    #     ax.set_title(title)
    #     ax.set_xlabel(axes[0])
    #     ax.set_ylabel(axes[1])
    #     ax.set_zlabel(axes[2])

    #     if save:
    #         ax.savefig(save + 'scatter_plot_4D_{}_{}.pdf'.format(data_name, title))
    #     
    #     plt.show()


    # @staticmethod
    # def scatter_3D(data, labels, axes, title, data_name, figsize=(10, 10), save=None):
    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=15)
    #     ax.set_title(title)
    #     ax.set_xlabel(axes[0])
    #     ax.set_ylabel(axes[1])
    #     ax.set_zlabel(axes[2])

    #     if save: 
    #         fig.savefig(save + 'scatter_plot_3D_{}_{}.pdf'.format(data_name, title))
    #     
    #     plt.show()


    # @staticmethod
    # def scatter_2D(data, labels, axes, title, data_name, figsize=(10, 10), save=None):
    #     fig = plt.figure(figsize=figsize)
    #     plt.scatter(data[:, 0], data[:, 1], c=labels) #, cmap='cool' )
    #     plt.title(title)
    #     plt.xlabel(axes[0])
    #     plt.ylabel(axes[1])

    #     if save:
    #         plt.savefig(save + 'scatter_plot_2D_{}_{}.pdf'.format(data_name, title))

    #     plt.show()
