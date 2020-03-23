from data.load_and_preprocess_data import DataSets
from utils.adtkAD import AatkAd
from utils.pyodAD import PyodAd
from utils.adPlot import anomaly_plot, tsne_plot
data_type = "gesture"
filename = "ann_gun_CentroidA.pkl"

ds = DataSets(data_type = data_type, filename = filename)
data, label = ds.preprocess()

adtkAd = AatkAd(data = data, label = label)
adtkAd.isolationForestAD(0.02)
anomaly_plot(adtkAd.data, adtkAd.label, adtkAd.anomalies)
# pyodAd = PyodAd(data = data, label = label)
# pyodAd.aeAD(hidden_neurons=[2,2,2,2], epochs=200)
# pyodAd.pyodPlot()
