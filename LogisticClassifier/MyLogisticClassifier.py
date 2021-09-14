# My Logistic Classifier
"""
Created on Tue Aug  3 12:06:44 2021

@author: Suus ten Hage
"""

import numpy as np

class MyLogisticClassifier:
    '''Class to build a logistic regression binary classifier from scratch.
    
    Takes in features (X) and results (y) and returns trained model 
    parameters as well as accuracy on the test set.'''
    
    def __init__(self, iterations = 10000, alpha = 0.01, normalize = True, 
                 fit_intercept = True):
        self.iterations = iterations #number of iterations for gradient descent
        self.alpha = alpha #learning rate
        self.normalize = normalize #True if data needs Z-score normalization
        self.fit_intercept = fit_intercept
    
    def Z_normalization(self, X):
        #Z-score normalization: (X - mean) / stdev 
        X = (X - X.mean(axis = 0))/X.std(axis = 0)  
        return X
    
    def add_intercept(self, X):
        #function to add a column of ones to X, so it is possible to fit the 
        #intercept during training. 
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis = 1)
        
    def sigmoid(self, z):
        sig = 1 / (1 + np.exp(-z))
        return sig
    
    def logistic_cost(self, h, y):
        cost = -np.mean(y * (np.log(h) - (1 - y) * np.log(1 - h)))
        return cost 
    
    def gradients(self, X, h, y):
        gradient = (1 / self.m) * np.dot(X.T, (h - y))
        return gradient

    def fit(self, X, y):

        #Add intercept if necessary
        if self.fit_intercept == True:
            X = self.add_intercept(X)

        self.m = X.shape[0] #number of training examples
        self.n = X.shape[1] #number of features
        #self.theta = np.zeros((self.n,1)) #weight initialization
        self.theta = np.random.random((self.n,1)) #weight initialization
        print(self.theta)
        
        #if data is not normalized yet, perform Z-score normalization
        if self.normalize == True:
             X = self.Z_normalization(X)
             
        #initialize losses list    
        costs = [None] * self.iterations 
        
        #Gradient descent
        for it in range(self.iterations):
            #calculate z and h with current weights
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            
            #update weights
            self.theta -= self.alpha * self.gradients(X, h, y)
            
            #calculate z and h with new weights 
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)   
            
            #calculate and store loss
            costs[it] = self.logistic_cost(h, y)
        
        return self.theta, costs
    
    def predict(self, X, theta):
        
        if self.fit_intercept == True:
            X = self.add_intercept(X)
            
        #if data is not normalized yet, perform Z-score normalization
        if self.normalize == True:
             X = self.Z_normalization(X)   
             
        #Calculate class probabilities
        probabilities = self.sigmoid(np.dot(X, theta))
        
        #predict classes
        predict_class = np.zeros((self.n,1))
        predict_class = np.array([1 if i > 0.5 else 0 for i in probabilities])
        predict_class = predict_class.reshape(predict_class.shape[0],1)
        
        return predict_class   
    
    pass