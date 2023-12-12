from algorithms.BIRCH import BIRCHClustering
from algorithms.kmeans import KMeans
from algorithms.IsoMap import SklearnIsoMap
from algorithms.PCA import CustomPCA
from algorithms.sklearnPCA import SklearnPCA
from algorithms.TruncatedSVD import SklearnTSVD
import timeit
from utils.data_preprocessing import Dataset
from evaluation.metrics import performance_eval
import argparse


# parser = argparse.ArgumentParser()

# parser.add_argument("-ds", "--dataset", help = "['iris', 'vowel', 'waveform', 'kr-vs-kp']", default='iris', type=str)
# parser.add_argument("-exp", "--experiment", help = "['dr', 'fv']", default='dr')
# parser.add_argument("-alg", "--clust_algorithm", help = "['Kmeans','Birch']", default='Kmeans')
# parser.add_argument("-fr", "--feature_reduction", help = "['PCA','iPCA','OwnPCA','TSVD']", default='OwnPCA')
# parser.add_argument("-comp", "--components", help = "Integer", default=4,type=int)
# parser.add_argument("-viz", "--visualization", help = "['PCA', 'Isomap']", default='PCA')
# parser.add_argument("-vcomp", "--viz_components", help = "Integer", default=4,type=int)
# parser.add_argument("-rs", "--random_seed", help = "an integer", default=55, type=int)

# args = parser.parse_args()


# clust_ags = {'Kmeans':KMeans,
#              'Birch':BIRCHClustering}
# clust_ags_params = {'Kmeans':{'k':args.components, 'max_iters':100},
#                     'Birch':{'X':None}}

# dr_ags = {'OwnPCA': CustomPCA,
#           'PCA': SklearnPCA,
#           'iPCA':SklearnPCA,
#           'TSVD':SklearnTSVD}


data = Dataset(f'../data/raw/iris.arff', cat_transf='onehot')
X = data.processed_data.iloc[:,:-1].values
Y = data.processed_data.iloc[:,-1].values

# dr_ags_params = {'OwnPCA': {'X':X, 'data_name':args.dataset, 'k':args.components},
#                  'PCA': {'data':X, 'data_name':args.dataset},
#                  'iPCA':{'data':X, 'data_name':args.dataset},
#                  'TSVD':{'data':X, 'data_name':args.dataset}}

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
    dim_red.visualize(Y, axes=range(X.shape[1]),exclude=['4d'], figsize=(30,15), title_size=8)
    dim_red.visualize(Y, axes=range(args.components),data2plot='Transformed', exclude=['4d'], figsize=(30,15), title_size=8)

    algorithm = clust_ags[args.clust_algorithm](**clust_ags_params[args.clust_algorithm])
    algorithm.fit(dim_red.transformed_data)
    predictions = algorithm.predict(dim_red.transformed_data)
    print(f'\n-- Reducted Data - Algorithm {args.clust_algorithm}')
    performance_eval(X, predictions, Y)
    

# else:
  
    
    


