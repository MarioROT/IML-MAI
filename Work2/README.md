# Introduction to Machine Learning
## Work 1 - Clustering Exercise

Work1: Laboratory project deliveries for the Introduction to Machine Learning (IML) course of the Master in Artificial Intelligence at UPC.

This project contains the implementation of following algorithms:

 - DBSCAN (sklearn)
 - Birch (sklearn)
 - K-Means, K-modes (our code)
 - K-Prototypes (our code)
 - Fuzzy C-Means (our code)



### Directory Structure

```
work1/
│
├── data/
│   ├── raw/
│   │   ├── waveform.arff
│   │   ├── dataset_24_mushroom.arff
│   │   ├── iris.arff
│   │   ├── vowel.arff
│   │   └── kr-vs-kp.arff
│
├── src/
│   ├── algorithms/
│   │   ├── fcm_py.py - Fuzzy C-Means clustering algorithm implementation
│   │   ├── kmeans.py - K-Means clustering algorithm implementation
│   │   ├── BIRCH.py - BIRCH clustering algorithm implementation
│   │   ├── kprototypes.py - K-Prototypes clustering algorithm implementation
│   │   ├── DBSCAN.py - DBSCAN clustering algorithm implementation
│   │   └── kmodes.py - K-Modes clustering algorithm implementation
│   ├── evaluation/
│   │   └── metrics.py - Evaluation metrics for clustering algorithms
│   ├── utils/
│   │   ├── analysis_kr-vs-kp.py - Analysis utility for the kr-vs-kp dataset
│   │   ├── data_preprocessing.py - Data preprocessing utility
│   │   ├── analysis_vowel.py - Analysis utility for the vowel dataset
│   │   ├── best_params_search.py - Utility for searching the best parameters
│   │   └── analysis_waveform.py - Analysis utility for the waveform dataset
│   ├── main.py - Main script for executing the project's primary functionality
│   └── test_performance.py - Script dedicated to performance testing of implemented algorithms
│
├── tests/
│   ├── DataPreprocessing.ipynb - Jupyter notebook for testing data preprocessing functions
│   └── FCM.py - Test script for the Fuzzy C-Means clustering algorithm
│
├── requirements.txt - List of project dependencies
└── environment.yml - Conda environment configuration file

```

### Setup the environment

These sections show how to create a virtual environment for our script and how to install dependencies.

1. Open the folder in terminal
```bash
cd <root_folder_of_project>/
```

2. Create a virtual environment
```bash
python3 -m venv venv/
```

3. Activate the virtual environment
```bash
source venv/bin/activate
```

4. Install the required dependencies
```bash
pip install -r requirements.txt
```
You can check if the dependencies were installed by running the next command; it should print a list of installed dependencies:
```bash
pip list
```

5. Deactivate the virtual environment
```bash
deactivate
```

## Execute scripts

1. Activate the virtual environment
```bash
source venv/bin/activate
```

2. Running the main script: main.py - Be sure to run the file from `src` directory
Arguments to choose:
```bash
- '-ds' or '--datasets': Specify the datasets you want to perform clustering on. Available options: 'iris', 'vowel', 'waveform', 'kr-vs-kp'.
- '-ags' or '--algorithms': Choose clustering algorithms to apply. Available options: 'kmeans', 'kmodes', 'kprot', 'fcm', 'dbscan', 'birch'.
- '-bp' or '--best_params': Set to 'True' if you want to search for the best algorithm parameters (default is 'True').
- '-dsm' or '--dataset_method': Specify the dataset type: 'numeric', 'categorical', or 'mixed' (default is 'numeric').
- '-ce' or '--cat_encoding': Choose categorical encoding: 'onehot' or 'ordinal' (default is 'onehot').
- '-r' or '--random_seed': Set an integer value for the random seed (default is '55').
```

For example, to run the script for the `iris` dataset using the `kprot` and `kmeans` algorithms:
```bash
python3 main.py -ds <dataset_name> -ags <algorithm_1> <algorithm_2> ...
```
```bash
python3 main.py -ds iris -ags kprot kmeans
```

3. When finish, deactivate the virtual environment
```bash
deactivate
```

## Analyzing the data
Inside the `src/utils` directory, there are three scripts tailored for in-depth analysis of the datasets. Each script provides unique insights and generates specific plots.
```bash
- analysis_kr-vs-kp.py
- analysis_vowel.py
- analysis_waveform.py
```
To execute them, just run:
```bash
python3 <analysis_script.py>
```

