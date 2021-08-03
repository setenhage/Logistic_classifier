# MyLogisticClassfierTest
"""
Created on Sat Jul 31 14:19:44 2021

@author: Suus ten Hage
"""
from LogisticClassifier.ObtainData import ObtainData
import pandas as pd

#load data
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/train.csv"
data_train = pd.read_csv(file)
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/test.csv"
data_test = pd.read_csv(file)

print(data_train.head())

#create mock data


#apply logistic classfier to loaded data
#apply logistic classifier to mock data