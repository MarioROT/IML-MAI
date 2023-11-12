from utils.data_preprocessing import Dataset
from algorithms.sklearnPCA import SklearnPCA

d = Dataset('../data/raw/iris.arff')
X = d.processed_data.iloc[:,:-1].values
y = d.processed_data.iloc[:,-1].values

skpca = SklearnPCA(X, 'iris')
skpca.iPCA(4)
skpca.visualize(y, exclude=['4d'], figsize=(30,15), title_size=8)

