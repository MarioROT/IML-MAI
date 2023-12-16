import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy.stats import rankdata
        
def Friedman_Nem(Data=None, p=0.05, title = ['','']):
    df=pd.DataFrame(Data)
    
    stat, p_table= stats.friedmanchisquare(*df.T.values.tolist())
    
    if p_table<p:
        print("----Null Hyp. rejected-----")
        print("----Performing Nemenyi-----")
        
        nem_fri = sp.posthoc_nemenyi_friedman(df)
        nem_fri.to_csv(title[0] + f'Fri_Nem_{title[1]}.csv', index=False)
        return nem_fri
        
    else:
        print("----Null Hyp. confirmed-----")
        print("----Not need to perform Nemenyi-----")
        
        return None
        
def process_results(results):
  combinations=[",".join(i.split(',')[1:]) for i in results['params'].unique()]
  groups=[]
  for combination in combinations:
    groups.append(results[results['params'].str.contains(combination)])
  accuracies=[ g['accuracies'].values for g in groups ]
  efficiencies=[ g['efficiencies'].values for g in groups ]
  times=[ g['total_times'].values for g in groups ]

  return accuracies,efficiencies,times



def avg_rank (Data, reverse=False, title=['','']):
  if reverse==True:
    samp=1-np.array(Data).T
  else:
    samp=np.array(Data).T

  ranks=[]
  
  for experiment in samp:
      rank=rankdata(experiment)
      ranks.append(rank)
      
  mean_ranks=[]
  for rank_v in np.array(ranks).T:
    mean_ranks.append(np.mean(rank_v))
  
  np.save(title[0] + f'Ranks_{title[1]}.npy', mean_ranks)
  return mean_ranks