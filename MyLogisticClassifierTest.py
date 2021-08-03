# MyLogisticClassfierTest
"""
Created on Sat Jul 31 14:19:44 2021

@author: Suus ten Hage
"""
from LogisticClassifier.MylogisticClassifier import MyLogisticClassifier as MLC
import pandas as pd

#load data
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/train.csv"
data_train = pd.read_csv(file)
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/test.csv"
data_test = pd.read_csv(file)

#Split data in X and y 


#apply logistic classifier to test survival of the Titanic tragedy
Survival = MylogisticClassifier(data_train, data_test)



#create mock data
#apply logistic classifier to mock data