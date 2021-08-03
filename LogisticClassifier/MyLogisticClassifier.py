# My Logistic Classifier
"""
Created on Tue Aug  3 12:06:44 2021

@author: Suus ten Hage
"""

class MyLogistiClassifier:
    '''Class to build a logistic regression binary classifier from scratch.
    
    Takes in two pandas dataframes, train and test and returns trained model 
    parameters as well as accuracy on the test set.'''
    
    def __init__(self, data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test
    
    def sigmoid(self, z):
        #Sigmoid hypothesis where z = Wx + b
        return 1.0 / (1 + np.exp(-z))
    
    def logistic_loss(y, y_est):
        loss = -np.mean(y * (np.log(y_est)) - (1 - y) * np.log(1 - y_est))
        return loss    
        
    
    
    pass