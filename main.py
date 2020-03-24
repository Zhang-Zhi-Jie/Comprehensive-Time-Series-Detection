from data.load_and_preprocess_data import DataSets
from utils.adtkAD import AatkAd
from utils.pyodAD import PyodAd
from utils.adPlot import anomaly_plot, tsne_plot
import os

data_type = "ecg"
filename = "chfdb_chf13_45590.pkl"

'''
DATA_PATH = "./data"
data_dir_list = os.listdir(DATA_PATH)
for obj in data_dir_list:
    DATA_TYPE_PATH = os.path.join(DATA_PATH, obj)
    if os.path.isdir(DATA_TYPE_PATH) and obj != "nyc_taxi" \
        and obj != "__pycache__":
        data_type = obj
        DATA_FILE_PATH = os.path.join(DATA_TYPE_PATH, "labeled/whole")
        file_dir_list = os.listdir(DATA_FILE_PATH)
        for filename in file_dir_list:
            ds = DataSets(data_type = data_type, filename = filename)
            data, label = ds.preprocess()
            # adtkAd = AatkAd(data = data, label = label)
            # adtkAd.isolationForestAD(0.02)
'''
# anomaly_plot(adtkAd.data, adtkAd.label, adtkAd.anomalies)
ds = DataSets(data_type = data_type, filename = filename)
data, label = ds.preprocess()
pyodAd = PyodAd(data = data, label = label)
pyodAd.vaeAD(encoder_neurons=[3,2,1], decoder_neurons=[1,2,3],
             epochs=200, contamination=0.1)
