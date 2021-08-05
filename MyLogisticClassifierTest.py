# MyLogisticClassfierTest
"""
Created on Sat Jul 31 14:19:44 2021

@author: Suus ten Hage
"""

from LogisticClassifier.MyLogisticClassifier import MyLogisticClassifier
import pandas as pd
from sklearn.datasets import make_classification

#create mock data
X, y = make_classification(n_features = 2, n_redundant = 0, 
                           n_informative = 2, random_state = 1, 
                           n_clusters_per_class = 1)

y = y.reshape(y.shape[0],1) #make sure y is proper size (m, 1) and not (m,)

#apply logistic classifier to mock data
model =  MyLogisticClassifier(iterations = 20000, alpha = 0.1, normalize = False)
[theta, losses] = model.fit(X,y)

print(theta)
print(len(losses))

#load data
#file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/train.csv"
#data_train = pd.read_csv(file)
#file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/test.csv"
#data_test = pd.read_csv(file)

#Split data in X and y 


#apply logistic classifier to test survival of the Titanic tragedy
#urvival = MylogisticClassifier(data_train, data_test)



