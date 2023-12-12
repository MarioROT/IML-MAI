import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from collections import Counter
import time

class KIBL:
    def __init__(self, X=None, K=3, voting='MP', retention='NR'):
        print("-------Performing K-IBL-------")
        self.X=X
        self.K=K
        self.voting=voting  #MP: Modified Plurality , BC: Borda Count 
        self.retention=retention #NR: Never Retains, . Always retain (AR) Different Class retention (DF). Degree of Disagreement (DD).

    #Normalizing the train data
    def normalize(self):
      data_normalized=self.X.copy()
      
      for column in data_normalized.columns:
        if is_numeric_dtype(data_normalized[column]):
          values=data_normalized[column].values
          min_v=values.min()
          max_v=values.max()

          for i in range(len(data_normalized[column])):
            value=data_normalized.loc[i,column]
            new_v=(value-min_v)/(max_v-min_v)
            data_normalized.loc[i,column]=new_v

      return data_normalized

    #Calculating the euc distance between two instances
    def euc_distance(test_row,train_row):
      dist=0
      for i in range (len(test_row)):
          dist += np.sum((test_row[i]-train_row[i])**2)
      return np.sqrt(dist)

    #Getting the K nearest neighbors with all the data from the learning base
    def get_neighbors(self, test_row):
      distances = []
      for train_row in self.X.values:
          dist = euc_distance(test_row, train_row)
          distances.append((train_row, dist))
      distances.sort(key=lambda tup: tup[1])
      neighbors = []
      for i in range(self.K):
          neighbors.append(distances[i][0])
      return neighbors
    
    #Predict the type according to the majority of neighbors
    def predict(self, neighbors):
      neighbors_labels = [row[-1] for row in neighbors]
      neighbour_class, counts=np.unique(neighbors_labels,  return_counts=True)
      length=len(neighbors_labels)

      if self.voting =='MP':
        if np.count_nonzero(counts==counts.max())==1:
          prediction = max(set(neighbors_labels), key=neighbors_labels.count)

        else:
          while np.count_nonzero(counts==counts.max())>1:
            neighbors_labels=neighbors_labels[:-1]
            neighbour_class, counts=np.unique(neighbors_labels,  return_counts=True)
          prediction = max(set(neighbors_labels), key=neighbors_labels.count)

        return prediction
        
      elif self.voting == 'BC':  
        points=np.arange(length, 0,-1, dtype=int)
        values_array=np.array(neighbors_labels)
        scores={}

        for label in set(values_array):
          labels=np.where(values_array==label,1,0)
          score=np.dot(labels,points)
          scores[label]=score

        max_points = max(scores.values())
        max_point_class = [label for label, points in scores.items() if points == max_points]
    
        if len(max_point_class)==1:
          return max_point_class[0]
        
        else:
          return next(label for label in values if label in max_point_class)

    def evaluate_accuracy(predictions, true_labels):
        correct_count = sum(1 for pred, true_label in zip(predictions, true_labels) if pred == true_label)
        return correct_count / len(true_labels)

    def evaluate_efficiency(problem_solving_times):
        return np.mean(problem_solving_times)


    def kIBLAlgorithm(self, test_row):

      data_normalized=normalize(np.vstack([self.X, test_row]))
      train_normalized=data_normalized[:self.X.shape[0]]
      test_normalized=data_normalized[:-test_row.shape[0]]
      
      true_labels = test_normalized[:, -1]
      predictions=[]
      problem_solving_times = []

      for instance in test_normalized: 
        start_time = time.time()
        neighbors=get_neighbors(train_normalized, instance, self.K)
        predict=(self.voting, neighbors)
        predictions.append(predict)

        if self.retention == 'NR':
          retained_instance = None

        elif self.retention == 'AR':
          retained_instance = test_row

        elif self.retention == 'DF':
          if test_row[-1] != predicted_label:
            retained_instance = test_row

          else:
            return None

        elif self.retention == 'DD':
          return None
      
        if retained_instance is not None:
          self.X = np.vstack([self.X, retained_instance])
        
        end_time = time.time()
        problem_solving_times.append(end_time - start_time)

      accuracy = evaluate_accuracy(predictions, true_labels)
      efficiency = evaluate_efficiency(problem_solving_times)

      return accuracy, efficiency         
