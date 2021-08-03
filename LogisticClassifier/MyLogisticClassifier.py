# My Logistic Classifier
"""
Created on Tue Aug  3 12:06:44 2021

@author: Suus ten Hage
"""

class MyLogistiClassifier:
    '''Class to build a logistic regression binary classifier from scratch.
    
    Takes in features (X) and results (y) and returns trained model 
    parameters as well as accuracy on the test set.'''
    
    def __init__(self, X, y, epochs = 1000, bs, alpha = 0.01,\
                 normalize = True):
        self.X = X
        self.y = y
        self.epochs = epochs #number of epochs
        self.bs = bs #batch size
        self.alpha = alpha #learning rate
        self.normalize = normalize #True if data needs Z-score normalization
        self.losses = []
        self.m = X.shape[0] #number of training examples
        self.n = X.shape [1] #number of features
        self.w = np.zeros((self.n,1)) #weight initialization
        self.b = 0 #bias initialization
        
    def sigmoid(self, z):
        #Sigmoid hypothesis where z = Wx + b
        return 1.0 / (1 + np.exp(-z))
    
    def logistic_loss(self, y, y_est):
        loss = -np.mean(y * (np.log(y_est)) - (1 - y) * np.log(1 - y_est)) \ 
            + lambda / (2 * self.m)
        return loss    
        
    def normalize(self)
        for i in range(self.n):
            X = (X - X.mean(axis = 0))/X.std(axis = 0)            
    
    def gradient_desc(self):
        
        if self.normalize = True:
                self.X = normalize(self.X)
        for epoch in range(epochs):
            
            #Calculate change in weights 
            y_est = self.sigmoid(np.dot(xb, self.w) + self.b)
            self.w = self.w - (self.alpha/ self.m) * \
                np.dot(self.X.T, (y_est - self.y))
    
    pass