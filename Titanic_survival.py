# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:02:02 2021

@author: Suzanne
"""
from LogisticClassifier.MyLogisticClassifier import MyLogisticClassifier
import pandas as pd
import numpy as np

#load data (pandas)
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/train.csv"
data_train_df = pd.read_csv(file)
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/test.csv"
data_test_df = pd.read_csv(file)

# Change 'female' and 'male' to 0 and 1. 
data_train_df['Sex'].replace('female', 0, inplace=True)
data_train_df['Sex'].replace('male', 1, inplace=True)
data_test_df['Sex'].replace('female', 0, inplace=True)
data_test_df['Sex'].replace('male', 1, inplace=True)

#Split in X and y
X_train_df = data_train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y_train_df = data_train_df['Survived']
X_test_df = data_test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

#create numpy array with passenger IDs
passenger_Id_df = data_test_df['PassengerId']
passenger_Id = passenger_Id_df.values
passenger_Id = passenger_Id.reshape(passenger_Id.shape[0],1) #make it shape (..,1)


#Change from pandas to numpy arrays
X_train = X_train_df.values
y_train = y_train_df.values
y_train = y_train.reshape(y_train.shape[0],1) #make sure y is proper size (m, 1) and not (m,)
X_test = X_test_df.values

#apply logistic classifier to test survival of the Titanic tragedy
survival = MyLogisticClassifier(iterations = 20000, alpha = 0.01, Lambda = 0.5,
                              normalize = False, fit_intercept = True)

[theta, J] = survival.fit(X_train,y_train)

#make predictions
pred = survival.predict(X_test, theta)

#Create csv for Kaggle submission (passenger id, prediction)
submit = np.concatenate((passenger_Id, pred), axis = 1)
submit_df = pd.DataFrame(submit, columns = ['passenger_Id', 'Survived'])
submit_df.to_csv('C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/submission.csv')