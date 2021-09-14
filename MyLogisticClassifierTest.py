# MyLogisticClassfierTest
"""
Created on Sat Jul 31 14:19:44 2021

@author: Suus ten Hage
"""

from LogisticClassifier.MyLogisticClassifier import MyLogisticClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

#create mock data
X, y = make_classification(n_features = 2, n_redundant = 0, 
                           n_informative = 2, random_state = 1, 
                           n_clusters_per_class = 1)

y = y.reshape(y.shape[0],1) #make sure y is proper size (m, 1) and not (m,)

#split data in test and train
index = np.arange(100)
np.random.shuffle(index)
training, test = index[:80], index[80:]

#m_train = math.floor(0.8 * X.shape[0])
#index = random.sample(range(X.shape[0]), m_train)

X_train = X[training,:]
y_train = y[training]

X_test = X[test,:]
y_test = y[test]

#apply logistic classifier to mock data
model =  MyLogisticClassifier(iterations = 20000, alpha = 0.01, 
                              normalize = False, fit_intercept = True)
[theta, J] = model.fit(X_train,y_train)

plt.plot(J)
plt.ylabel('losses')
plt.show()


pred = model.predict(X_test, theta)

accuracy = (pred == y_test).mean()
print(accuracy)

#load data
#file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/train.csv"
#data_train = pd.read_csv(file)
#file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/test.csv"
#data_test = pd.read_csv(file)

#Split data in X and y 


#apply logistic classifier to test survival of the Titanic tragedy
#survival = MylogisticClassifier(data_train, data_test)



