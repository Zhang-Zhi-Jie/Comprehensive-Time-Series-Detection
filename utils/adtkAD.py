import adtk
import pandas as pd
import matplotlib.pyplot as plt
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import ThresholdAD
from adtk.detector import QuantileAD
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
from adtk.detector import InterQuartileRangeAD
from adtk.detector import GeneralizedESDTestAD
from adtk.detector import OutlierDetector 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from adtk.detector import RegressionAD
from sklearn.linear_model import LinearRegression
from adtk.detector import PcaAD


class AatkAd():

    def __init__(self, data, label):
        self.data, self.label = self.load_AdtkFormat_Data(data, label)
        
    
    def load_AdtkFormat_Data(self, data, label):
        rng = pd.date_range('3/21/2020', periods=data.shape[0], freq='S')
        label = pd.DataFrame(data = label, index = rng, columns=['label'])
        label = pd.Series(data = label['label'].values, index=label.index)
        dataFrame = pd.DataFrame(data = data, index = rng)
        data = validate_series(dataFrame)
        return data, label
    
    """
    unvariable time-series anomaly detection
    """
    def thresholdAd(self, high, low):
        threshold_ad = ThresholdAD(high=high, low=low)
        anomalies = threshold_ad.detect(self.data)
        self.anomalies = anomalies

    def quantileAd(self, high, low):
        quantile_ad = QuantileAD(high=high, low=low)
        anomalies = quantile_ad.fit_detect(self.data)
        self.anomalies = anomalies
    
    def interQuartileRangeAD(self, c):
        iqr_ad = InterQuartileRangeAD(c = c)
        anomalies = iqr_ad.fit_detect(self.data)
        self.anomalies = anomalies
    
    def generalizedESDTestAD(self, alpha):
        esd_ad = GeneralizedESDTestAD(alpha=alpha)
        anomalies = esd_ad.fit_detect(self.data)
        self.anomalies = anomalies

    """
    Multivariate time-series anomaly detection
    """
    def minClusterDetector(self,  n_clusters):
        min_cluster_detector = MinClusterDetector(KMeans(n_clusters=n_clusters))
        anomalies = min_cluster_detector.fit_detect(self.data)
        self.anomalies = anomalies
    
    def localOutlierFactorAD(self, c):
        outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=c))
        anomalies = outlier_detector.fit_detect(self.data)
        self.anomalies = anomalies
    
    def isolationForestAD(self, c):
        outlier_detector = OutlierDetector(IsolationForest(contamination=c))
        anomalies = outlier_detector.fit_detect(self.data)
        self.anomalies = anomalies

    def regressionAD(self, target, c):
        regression_ad = RegressionAD(regressor=LinearRegression(), target=target, c=c)
        anomalies = regression_ad.fit_detect(self.data)
        self.anomalies = anomalies

    def pcaAD(self, k):
        pca_ad = PcaAD(k=k)
        anomalies = pca_ad.fit_detect(self.data)
        self.anomalies = anomalies

