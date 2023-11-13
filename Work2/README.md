# Introduction to Machine Learning
## Work 2 - Dimensionality reduction with PCA and TruncatedSVD and Visualization using PCA and ISOMAP

## Overview
"Work 2" is part of the Introduction to Machine Learning course for the Master in Artificial Intelligence at UPC. This phase focuses on dimensionality reduction techniques like PCA and truncatedSVD and to visualize visualize with PCA and ISOMAP algorithms several data sets from the UCI repository

## Directory Structure
```
work2/
│
├── data/
│   ├── raw/
│   │   ├── waveform.arff
│   │   ├── iris.arff
│   │   ├── vowel.arff
│   │   └── kr-vs-kp.arff
│
├── src/
│   ├── algorithms/
│   │   ├── BIRCH.py
│   │   ├── IsoMap.py
│   │   ├── kmeans.py
│   │   ├── PCA.py
│   │   ├── sklearnPCA.py
│   │   └── TruncatedSVD.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── utils/
│   │   ├── analysis_kr-vs-kp.py
│   │   ├── data_preprocessing.py
│   │   ├── analysis_vowel.py
│   │   ├── best_params_search.py
│   │   └── analysis_waveform.py
│   ├── main.py
│   └── test_performance.py
│
├── tests/
│   ├── DataPreprocessing.ipynb
│   └── FCM.py
│
├── requirements.txt
└── environment.yml
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
### Or Setup the Environment using Conda

1. **Open a Terminal**: Navigate to the 'Work2' folder in your terminal.
2. **Create the Conda Environment**: Execute the following command to create a Conda environment using the `environment.yml` file:
   ```
   conda env create -f environment.yml
   ```
3. **Activate the Environment**: Once the environment is created, activate it with the command:
   ```
   conda activate IML8
   ```

## Execute scripts

1. Activate the virtual environment
```bash
source venv/bin/activate
```

2. Running the main script: main.py - Be sure to run the file from `src` directory
Arguments to choose:
```bash
- '-ds' or '--dataset': Specify the dataset for the experiment. Available options: 'iris', 'vowel', 'waveform', 'kr-vs-kp'.
- '-exp' or '--experiment': Choose the type of experiment. Options: 'dr' (dimensionality reduction), 'fv' (feature visualization).
- '-alg' or '--clust_algorithm': Select the clustering algorithm. Options: 'Kmeans', 'Birch'.
- '-fr' or '--feature_reduction': Choose the method for feature reduction. Options: 'PCA', 'iPCA', 'OwnPCA', 'TSVD'.
- '-comp' or '--components': Set the number of components for feature reduction (default is 4).
- '-viz' or '--visualization': Select the visualization tool. Options: 'PCA', 'Isomap'.
- '-vcomp' or '--viz_components': Specify the number of components for visualization (default is 4).
- '-rs' or '--random_seed': Set an integer value for the random seed (default is '55').
```

For example

- **For Dimensionality Reduction Experiment**:
  ```
  python main.py -ds iris -exp dr -alg Kmeans -fr TSVD -comp 4
  ```
  This command runs the dimensionality reduction experiment on the Iris dataset using Kmeans for clustering, TSVD for feature reduction, and sets the number of components to 4.

- **For Feature Visualization Experiment**:
  ```
  python main.py -ds iris -exp fv -alg Kmeans -fr PCA -comp 4 -viz Isomap -vcomp 4
  ```
  This command is used to run the feature visualization experiment on the Iris dataset, employing Kmeans for clustering, PCA for feature reduction, Isomap for visualization, and setting both PCA and visualization components to 4.


### Notes
- Make sure to navigate to the 'src' directory within 'Work2' before running the `main.py` script.
- These instructions assume you have Conda installed and set up on your system. If not, please install Conda from [Anaconda's official website](https://www.anaconda.com/products/individual).