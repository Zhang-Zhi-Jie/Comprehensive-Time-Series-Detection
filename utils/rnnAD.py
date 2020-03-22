import pandas as pd
from data.preprocess import DataSets

class rnnAd():

    def __init__(self, data):
        self.data = data
        self.dataFrame, self.label = self.load_AdtkFormat_Data(self.data)
        
    
    def load_AdtkFormat_Data(self):
        dataFrame = pd.DataFrame(data=self.data)
        return dataFrame, label 
    