# MyLogisticClassfierTest
"""
Created on Sat Jul 31 14:19:44 2021

@author: Suus ten Hage
"""
from LogisticClassifier.ObtainData import ObtainData


#load data
file = "C:/Users/Suzanne/Documents/Nanodegree_MLE/Log_reg_classifier/Data/titanic/train.csv"
data = ObtainData.load_from_dir_csv(file)
print(len(data))

#create mock data


#apply logistic classfier to loaded data
#apply logistic classifier to mock data