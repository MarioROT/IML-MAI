
import timeit
from utils.data_preprocessing import Dataset
from evaluation.metrics import performance_eval
import argparse
from KIBL import KIBL


data = Dataset(f'../data/folded/Nueva carpeta/pen-based', cat_transf='onehot', folds=True)

for train,test in data:
    K=[3,5,7]
    Voting=['MP','BC']
    Retention=[]
    for neighbors in K: 
        KIBL.kIBLAlgorithm(X=train, K=)
    
    


