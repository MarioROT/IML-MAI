import timeit
import argparse

from algorithms.BIRCH import BIRCHClustering
from algorithms.kmeans import KMeans
from algorithms.IsoMap import SklearnIsoMap
from algorithms.PCA import CustomPCA
from algorithms.sklearnPCA import SklearnPCA
from algorithms.TruncatedSVD import SklearnTSVD

from utils.data_preprocessing import Dataset
from evaluation.metrics import performance_eval, map_clusters_to_labels

parser = argparse.ArgumentParser()

parser.add_argument("-ds", "--dataset", help = "['iris', 'vowel', 'waveform', 'kr-vs-kp']", default='iris', type=str)
parser.add_argument("-exp", "--experiment", help = "['dr', 'fv']", default='dr')
parser.add_argument("-alg", "--clust_algorithm", help = "['Kmeans','Birch']", default='Kmeans')
parser.add_argument("-fr", "--feature_reduction", help = "['PCA','iPCA','OwnPCA','TSVD']", default='OwnPCA')
parser.add_argument("-comp", "--components", help = "Integer", default=4,type=int)
parser.add_argument("-viz", "--visualization", help = "['PCA', 'Isomap']", default='PCA')
parser.add_argument("-vcomp", "--viz_components", help = "Integer", default=4,type=int)
parser.add_argument("-rs", "--random_seed", help = "an integer", default=55, type=int)

args = parser.parse_args()


clust_ags = {'Kmeans':KMeans,
             'Birch':BIRCHClustering}
clust_ags_params = {'Kmeans':{'k':args.components, 'max_iters':100},
                    'Birch':{'X':None}}

dr_ags = {'OwnPCA': CustomPCA,
          'PCA': SklearnPCA,
          'iPCA':SklearnPCA,
          'TSVD':SklearnTSVD}


data = Dataset(f'../data/raw/{args.dataset}.arff', cat_transf='onehot')
X = data.processed_data.iloc[:,:-1].values
Y = data.processed_data.iloc[:,-1].values

dr_ags_params = {'OwnPCA': {'X':X, 'data_name':args.dataset, 'k':args.components},
                 'PCA': {'data':X, 'data_name':args.dataset},
                 'iPCA':{'data':X, 'data_name':args.dataset},
                 'TSVD':{'data':X, 'data_name':args.dataset}}

viz_ags = {'OwnPCA': CustomPCA,
          'PCA': SklearnPCA,
          'iPCA':SklearnPCA,
          'Isomap':SklearnIsoMap}

viz_ags_params = {'OwnPCA': {'data_name':args.dataset, 'k':args.components},
                 'PCA': {'data_name':args.dataset},
                 'iPCA':{'data_name':args.dataset},
                 'Isomap':{'data_name':args.dataset}}


if args.experiment == 'dr':
    
    algorithm = clust_ags[args.clust_algorithm](**clust_ags_params[args.clust_algorithm])
    algorithm.fit(X)
    predictions = algorithm.predict(X)
    print(f'\n-- Original Data - Algorithm {args.clust_algorithm}')
    performance_eval(X, predictions, Y)
    
    dim_red=dr_ags[args.feature_reduction](**dr_ags_params[args.feature_reduction])
    if args.feature_reduction not in ['PCA','iPCA']:
        dim_red.fit(args.components)
    elif args.feature_reduction == 'PCA':
        dim_red.PCA(args.components)
    elif args.feature_reduction == 'iPCA':
        dim_red.iPCA(args.components)
    dim_red.visualize(Y, axes=range(4), layout=(2,5), exclude=['4d'], figsize=(30,15), title_size=8)
    dim_red.visualize(Y, range(4) if args.components > 4 else range(args.components), layout=(2,5), data2plot='Transformed', exclude=['4d'], figsize=(30,15), title_size=8)

    algorithm = clust_ags[args.clust_algorithm](**clust_ags_params[args.clust_algorithm])
    algorithm.fit(dim_red.transformed_data)
    predictions = algorithm.predict(dim_red.transformed_data)
    print(f'\n-- Reducted Data - Algorithm {args.clust_algorithm}')
    performance_eval(dim_red.transformed_data, predictions, Y)
    
else:

    algorithm = clust_ags[args.clust_algorithm](**clust_ags_params[args.clust_algorithm])
    algorithm.fit(X)
    predictions = algorithm.predict(X)
    print(f'\n-- Original Data - Algorithm {args.clust_algorithm}')
    performance_eval(X, predictions, Y)
    predictions = map_clusters_to_labels(predictions, Y)

    viz=viz_ags[args.visualization](data= X,**viz_ags_params[args.visualization])
    if args.visualization not in ['PCA','iPCA']:
        viz.fit(args.viz_components)
    elif args.visualization == 'PCA':
        viz.PCA(args.viz_components)
    elif args.visualization == 'iPCA':
        viz.iPCA(args.viz_components)
    viz.visualize(Y, range(4), layout=(2,5), exclude=['4d'], figsize=(30,15), title_size=8)
    # viz.visualize(Y, axes=range(4), layout=(2,5), data2plot='Transformed', exclude=['4d'], figsize=(30,15), title_size=8)
    viz.visualize(predictions, range(4) if args.viz_components > 4 else range(args.viz_components),'Before Reduction '+ args.clust_algorithm, data2plot='Transformed', layout=(2,5), exclude=['4d'], figsize=(30,15), title_size=8)

    
    dim_red=dr_ags[args.feature_reduction](**dr_ags_params[args.feature_reduction])
    if args.feature_reduction not in ['PCA','iPCA']:
        dim_red.fit(args.components)
    elif args.feature_reduction == 'PCA':
        dim_red.PCA(args.components)
    elif args.feature_reduction == 'iPCA':
        dim_red.iPCA(args.components)
    dim_red.visualize(Y,exclude=['4d','3d', '2d'], data2plot = 'Transformed', layout=(1,1), figsize=(30,15), title_size=8)

    algorithm = clust_ags[args.clust_algorithm](**clust_ags_params[args.clust_algorithm])
    algorithm.fit(dim_red.transformed_data)
    predictions = algorithm.predict(dim_red.transformed_data)
    print(f'\n-- Reducted Data - Algorithm {args.clust_algorithm}')
    performance_eval(dim_red.transformed_data, predictions, Y)
    predictions = map_clusters_to_labels(predictions, Y)

    viz=viz_ags[args.visualization](data= dim_red.transformed_data,**viz_ags_params[args.visualization])
    if args.visualization not in ['PCA','iPCA']:
        viz.fit(args.components)
    elif args.visualization == 'PCA':
        viz.PCA(args.components)
    elif args.visualization == 'iPCA':
        viz.iPCA(args.components)
    viz.visualize(predictions, range(4) if args.viz_components > 4 else range(args.viz_components),'After Reduction ' + args.clust_algorithm + '-' + args.feature_reduction, data2plot='Transformed', layout=(2,5), exclude=['4d'], figsize=(30,15), title_size=8)

  
    
    


