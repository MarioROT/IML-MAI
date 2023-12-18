import sys
sys.path.append('../')
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from feature_selection import FeatureSelection

class KIBL:
    def __init__(self, 
                 X=None, 
                 K=3, 
                 voting='MP', 
                 retention='NR', 
                 feature_selection = 'ones', 
                 k_fs = 'nonzero',
                 normalize=False,
                 store_used_neighbors = False):
        print("-------Performing K-IBL-------")
        self.X=X
        self.K=K
        self.voting=voting  #MP: Modified Plurality , BC: Borda Count 
        self.retention=retention #NR: Never Retains, . Always retain (AR) Different Class retention (DF). Degree of Disagreement (DD).
        self.weights_m = feature_selection
        self.k_weights = k_fs
        self.normalize=normalize
        self.store_used_neighbors = store_used_neighbors
        
    #Normalizing the train data
    def normalize_fun(self,X):
        data_normalized=pd.DataFrame(X)
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
    def euc_distance(self, test_row, train_row):
        dist = np.sum(self.weights * ((test_row-train_row)**2))
        return np.sqrt(dist)

    #Getting the K nearest neighbors with all the data from the learning base
    def get_neighbors(self, train, test_row):
        distances = []
        for i, train_row in enumerate(train.values):
            x_train=train_row[:-1]
            dist = self.euc_distance(test_row[:-1], x_train)
            distances.append((train_row, dist, i))
        distances.sort(key=lambda tup: tup[1])
        neighbors = np.array(distances[:self.K],dtype=object)[:,0]
        if self.store_used_neighbors:
             self.used_neighbors.append(np.array(distances[:self.K], dtype=object)[:, 2])
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
            points=np.arange(length-1, -1,-1, dtype=int)
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
                return next(label for label in values_array if label in max_point_class)

    def compute_weights(self, data, method):
        features = data.loc[:, data.columns != 'y_true']
        labels = data.loc[:,'y_true']
        return FeatureSelection(features, labels, method, self.k_weights).compute_weights()
        

    def evaluate_accuracy(self, predictions, true_labels):
        correct_count = sum(1 for pred, true_label in zip(predictions, true_labels) if pred == true_label)
        return correct_count / len(true_labels)

    def evaluate_efficiency(self, problem_solving_times):
        return np.mean(problem_solving_times)


    def kIBLAlgorithm(self, test_data):
        self.used_neighbors = [] if self.store_used_neighbors else False

        if self.normalize==True:
            data_normalized=self.normalize_fun(np.vstack([self.X, test_data]))
            train_normalized=data_normalized[:self.X.shape[0]]
            test_normalized=data_normalized[-test_data.shape[0]:]
        else:
            train_normalized=self.X
            test_normalized=test_data
        
        print('----Data normalized----')

        self.weights = self.compute_weights(train_normalized, self.weights_m)

        true_labels = test_normalized.iloc[:,-1]
        predictions=[]
        problem_solving_times = []
        start_total_time=time.time()
        for i,instance in enumerate(test_normalized.values): 
            if i%1000==0:
                print(f'iteration:{i}')
            start_time = time.time()
            neighbors=self.get_neighbors(train_normalized, instance)
            predict=self.predict(neighbors)
            predictions.append(predict)

            if self.retention == 'NR':
                retained_instance = None
            elif self.retention == 'AR':
                retained_instance = instance
            elif self.retention == 'DF':
                if instance[-1] != predict:
                    retained_instance = instance
                else:
                    retained_instance = None
            elif self.retention == 'DD':
                neighbors_labels = [row[-1] for row in neighbors]
                neighbour_class, counts=np.unique(neighbors_labels,  return_counts=True)
                majority_cases=max(counts)
                dd=(self.K-majority_cases)/((len(neighbour_class)-1)*majority_cases)
    
                if dd>=0.5: 
                    retained_instance = instance
                else:
                    retained_instance = None
    
            if retained_instance is not None:
                self.X=pd.concat([self.X,pd.DataFrame(retained_instance.reshape(1,-1),columns=self.X.columns)],ignore_index=True)
                train_normalized=pd.concat([train_normalized,pd.DataFrame(retained_instance.reshape(1,-1),columns=train_normalized.columns)],ignore_index=True)
            
            end_time = time.time()
            problem_solving_times.append(end_time - start_time)

        self.predictions = predictions
        # self.used_neighbors = np.unique(self.used_neighbors)

        accuracy = self.evaluate_accuracy(predictions, true_labels)
        print('----Accuracy Completed----')

        efficiency = self.evaluate_efficiency(problem_solving_times)
        print('----Efficiency Completed----')
        end_total_time=time.time()
        total_time= end_total_time-start_total_time

        return accuracy, efficiency, total_time         