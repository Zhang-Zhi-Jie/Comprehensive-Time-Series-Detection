from utils.adtkAD import AatkAd
from data.load_and_preprocess_data import DataSets

data_type = "ecg"
filename = "chfdb_chf13_45590.pkl"

ds = DataSets(data_type = data_type, filename = filename)
data, label = ds.preprocess()

adtkAd = AatkAd(data = data, label = label)
# print(adtkAd.label)
adtkAd.pcaAD(1)
adtkAd.adtk_plot(anomaly_true = adtkAd.label, anomaly_pred=adtkAd.anomalies)

