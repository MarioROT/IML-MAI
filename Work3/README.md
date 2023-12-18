
## Overview
"Work 3" is part of the Introduction to Machine Learning course for the Master in Artificial Intelligence at UPC, it focuses on Lazy learning.

## Directory Structure
```
Work3/
│
├── src/ main.py
├── utils/
│   ├── data_preprocessing.py
│   ├── StatTest.py
│   ├── best_params_search.py
│   └── ...
├── algorithms/
│   
└── data/
   
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

- "-bp" or  "--experiment", Options: "['BPS':BestParamsSearch, 'BFS':BestFeatureSelection,'BIS':'BestInstanceSelection']".
- "-ds"or "--datasets", nargs='+', Options: "['pen-based','vowel', 'kr-vs-kp']".
- "-k" or "--nearest_neighbors", nargs='+', Options: "[3, 5, 7]".
- "-vot" or "--voting", nargs='+', Options: "['MP':Modified_Plurality,'BC':Borda_Count']".
- "-ret" or "--retention", nargs='+', Options: "['NR':Never_Retain,'AR':Always_Retain,'DF':Different Class Ret,'DD':Degree disagreement]".
- "-fs" or "--feature_selection": Options: "['ones', 'CR':Correlation, 'IG':Information Gain,'C2S':Chi Square Stat, 'VT':Variance Treshold, 'MI':Mutual Inf.,'C2': ChiSq. SKL, 'RF': Relief]".
- "-kfs" or "--k_fs": Options:"['nonzero', 'n%' -> e.g. '80%']".
- "-is" or "--instance_selection", Options: "['None','MCNN':Modif. Cond NN, 'ENN':Edited NNR, 'IBL3']".
- "-sd" or "--sample_data": Options:"[2,3]".s

```

## Usage
To run the `main.py` script:
```
python main.py [OPTIONS]
```

### Example
```
python main.py -bp=BIS -k=7 --voting=BC -ret=DF -is=IB3
```

### Notes
- Make sure to navigate to the 'src' directory within 'Work3' before running the `main.py` script.
- These instructions assume you have Conda installed and set up on your system. If not, please install Conda from [Anaconda's official website](https://www.anaconda.com/products/individual).