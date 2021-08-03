# MyLogisticClassifier
"""
Created on Sat Jul 31 13:53:00 2021

@author: Suus ten Hage
"""
import pandas as pd

class ObtainData:
    
    def __init__(self,filename):
        self.filename = filename
        
        
    def load_from_dir_csv(self):
        self.data = pd.read_csv(self.filename)
        return self.data
        
    #def create_mock_data(self):
        #Create data
        
        #Store data
    
    #def split_train_test(self):
        #split train and test data with provided split values. 
    
    pass