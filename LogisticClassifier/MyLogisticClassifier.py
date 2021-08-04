# My Logistic Classifier
"""
Created on Tue Aug  3 12:06:44 2021

@author: Suus ten Hage
"""

import numpy as np

class MyLogistiClassifier:
    '''Class to build a logistic regression binary classifier from scratch.
    
    Takes in features (X) and results (y) and returns trained model 
    parameters as well as accuracy on the test set.'''
    
    def __init__(self, it = 10000, alpha = 0.01, normalize = True):
        self.iterations = it #number of iterations for gradient descent
        self.alpha = alpha #learning rate
        self.normalize = normalize #True if data needs Z-score normalization
        self.theta = np.zeros((self.n,1)) #weight initialization
    
    def normalize(self, X):
        #Z-score normalization
        for i in range(self.n):
            X = (X - X.mean(axis = 0))/X.std(axis = 0)  
        
    def sigmoid(self, z):
        #Sigmoid hypothesis where z = Wx + b
        return 1 / (1 + np.exp(-z))
    
    def logistic_loss(self, h, y):
        loss = -np.mean(y * (np.log(h) - (1 - y) * np.log(1 - h)))
        return loss 
    
    def gradients(self, X, h, y):
        gradient = (1 / self.m) * np.dot(X.T, (h - y))
        return gradient

    def fit(self, X, y):
        self.m = X.shape[0] #number of training examples
        self.n = X.shape [1] #number of features
        
        #if data is not normalized yet, perform Z-score normalization
        if self.normalize == True:
             X = self.normalize(X)
            
        self.losses = [] #initialize losses list 
        for it in range(self.iterations):
            #calculate z and h with current weights 
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            
            #update weights
            self.theta -= self.alpha * self.gradient(X, h, y)
            
            #calculate z and h with new weights 
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)   
            
            #calculate and store loss
            losses[i] = self.loss(h, y)
            
            
    
    pass