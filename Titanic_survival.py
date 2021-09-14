# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:02:02 2021

@author: Suzanne
"""

import pandas as pd

#load data
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/train.csv"
data_train = pd.read_csv(file)
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/test.csv"
data_test = pd.read_csv(file)

#Split data in X and y 


#apply logistic classifier to test survival of the Titanic tragedy
survival = MylogisticClassifier(data_train, data_test)