import pandas as pd
import numpy as np
import pyod
from pyod.models.auto_encoder import AutoEncoder
from adtk.data import validate_series
from adtk.visualization import plot
import matplotlib.pyplot as plt 

class PyodAd():

    def __init__(self, data, label):
        self.X = data
        self.data, self.label = self.load_PyodFormat_Data(data, label)
    def load_PyodFormat_Data(self, data, label):
        rng = pd.date_range('3/21/2020', periods=data.shape[0], freq='S')
        label = pd.DataFrame(data = label, index = rng, columns=['label'])
        label = pd.Series(data = label['label'].values, index=label.index)
        dataFrame = pd.DataFrame(data = data, index = rng)
        data = validate_series(dataFrame)
        return data, label
    
    def aeAD(self, hidden_neurons, epochs):
        # train AutoEncoder detector
        clf_name = 'AutoEncoder'
        clf = AutoEncoder(hidden_neurons = hidden_neurons, epochs=epochs)
        clf.fit(self.X)
        # get the prediction labels and outlier scores of the training data
        y_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_scores = clf.decision_scores_  # raw outlier scores
        ad = pd.DataFrame(data = np.array([False] * self.data.shape[0]), index=self.label.index, columns=["y_pred"])
        for i in range(self.data.shape[0]):
            if y_pred[i] == 1:
                ad.loc[self.label.index[i], "y_pred"] = True
        self.anomalies = pd.Series(ad['y_pred'].values, index=ad.index)
        