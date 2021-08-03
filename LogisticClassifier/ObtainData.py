# MyLogisticClassifier
"""
Created on Sat Jul 31 13:53:00 2021

@author: Suus ten Hage
"""
import pandas as pd

class ObtainData:
    
    def __init__(self, directory, filename):
        self.dir = directory
        self.filename = filename
        
    def load_from_dir_csv(self, file_dir):
        #load csv with filename from directory (including the filename)
        self.data = pd.read_csv(file_dir)
        
    #def create_mock_data(self):
        #Create data
        
        #Store data
    
    #def split_train_test(self):
        #split train and test data with provided split values. 
    
    pass