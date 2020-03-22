import matplotlib.pyplot as plt 
from adtk.visualization import plot

def plot(data, anomaly_true, anomaly_pred):
    plot(data, anomaly_true = anomaly_true, anomaly_pred = anomaly_pred, 
            ts_linewidth=1, ts_markersize=3, at_color='red', at_alpha=0.3, curve_group='all');
    plt.show()