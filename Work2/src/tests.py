from utils.data_preprocessing import Dataset
from algorithms.sklearnPCA import SklearnPCA
from algorithms.TruncatedSVD import SklearnTSVD
from algorithms.PCA import CustomPCA
from algorithms.IsoMap import SklearnIsoMap



d = Dataset('../data/raw/iris.arff')
X = d.processed_data.iloc[:,:-1].values
y = d.processed_data.iloc[:,-1].values



#Incremental PCA
# skpca = SklearnPCA(X, 'iris')
# skpca.iPCA(4)
# skpca.visualize(y, exclude=['4d'], figsize=(30,15), title_size=8)

#Classical PCA
skpca = SklearnPCA(X, 'iris')
skpca.PCA(2)
skpca.visualize(y, axes=[0,1], exclude=['4d'],data2plot= 'Transformed', figsize=(30,15), title_size=8)

#Own PCA
# skpca =CustomPCA(X,4)
# skpca.fit()
# skpca.visualize(y, exclude=['4d'], figsize=(30,15), title_size=8)

# Isomap
# isomap =SklearnIsoMap(X,'iris')
# isomap.fit(3)
# isomap.visualize(y,axes=[0,1,2], exclude=['4d'], data2plot='Transformed',figsize=(30,15), title_size=8)


#Truncated SVD
# sktsvd= SklearnTSVD(X,'iris')
# sktsvd.fit(4)
# sktsvd.visualize(y, exclude=['4d'], figsize=(30,15), title_size=8)


