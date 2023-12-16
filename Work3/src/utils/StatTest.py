import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scikit_posthocs as sp
        
def Friedman_Nem(Data=None, p=0.05):
    df=pd.DataFrame(Data)
    
    stat, p_table= stats.friedmanchisquare(*df.T.values.tolist())
    
    if p_table<p:
        print("----Null Hyp. rejected-----")
        print("----Performing Nemenyi-----")
        
        return sp.posthoc_nemenyi_friedman(df)
        
    else:
        print("----Null Hyp. confirmed-----")
        print("----Not need to perform Nemenyi-----")