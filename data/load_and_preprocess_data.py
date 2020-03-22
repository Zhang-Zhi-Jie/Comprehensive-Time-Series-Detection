import os
import numpy as np 
import pandas as pd
import pickle


def standardization(seqData,mean,std):
    return (seqData-mean)/std

class DataSets():
    """
    load and preprocess data

    Parameters
    ----------
    data_type: "ecg", "gesture", "nyc_taxi", "power_demand", "respiration", "space_shuttle"

    filename: just as its name implies

    """
    def __init__(self, data_type, filename):
        self.data_type = data_type
        self.filename = filename
        self.loadData()


    """
    load data
    """
    def loadData(self):
        path = './data/'+ self.data_type+'/labeled'+'/whole/'+ self.filename
        with open(str(path), 'rb') as f:
            data = np.array(pickle.load(f))
            label = data[:,-1]
            data = data[:,:-1]
        label = list(label)
        for i in range(len(label)):
            if(label[i] == 0.0):
                label[i] = False
            else:
                label[i] = True
        self.mean = data.mean(axis=0)
        self.std= data.std(axis=0)
        self.data = data
        self.label = label
    """
    preprocess data by standardization

    return
    ------
    standardized data
    label
    """
    def preprocess(self):
        data = standardization(self.data,self.mean,self.std)
        return data, self.label
