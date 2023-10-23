In order to run the code, first is needed to install a virtual environment with the following command: 

```bash
conda env create -f environment.yml
```
Be sure to run the previous command inside the Work1 directory

Activate the environment with the command: 
```bash
conda activate IML23
```

Move to the src folder, where the main.py is located
```bash
cd src
```

```bash
- '-ds' or '--datasets': Specify the datasets you want to perform clustering on. Available options: 'iris', 'vowel', 'waveform', 'kr-vs-kp'.
- '-ags' or '--algorithms': Choose clustering algorithms to apply. Available options: 'kmeans', 'kmodes', 'kprot', 'fcm', 'dbscan', 'birch'.
- '-bp' or '--best_params': Set to 'True' if you want to search for the best algorithm parameters (default is 'True').
- '-dsm' or '--dataset_method': Specify the dataset type: 'numeric', 'categorical', or 'mixed' (default is 'numeric').
- '-ce' or '--cat_encoding': Choose categorical encoding: 'onehot' or 'ordinal' (default is 'onehot').
- '-r' or '--random_seed': Set an integer value for the random seed (default is '55').
```

You can run for example the K-prototypes and K-means with the iris dataset by using the following command:
```bash
python3 main.py -ds iris -ags kprot kmeans
```
