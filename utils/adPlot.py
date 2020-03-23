import matplotlib.pyplot as plt 
from adtk.visualization import plot
from sklearn import manifold
import numpy as np

def anomaly_plot(data, anomaly_true, anomaly_pred):
    """
    Plot time series and/or anomalies.
    """
    plot(data, 
         anomaly={"anomaly_true":anomaly_true, 
                  "anomaly_pred":anomaly_pred},
         ts_linewidth=1, 
         ts_markersize=3, 
         anomaly_color={"anomaly_true":'blue',
                        "anomaly_pred":'red'}, 
         anomaly_alpha=0.3, 
         curve_group='all')
    plt.show()


def tsne_plot(data, label, anomalies):
    """
    Reduce the dimension(>=3) of data to 2D
    and plot a figure
    """
    anomalies_index = anomalies.reset_index(drop=True)
    y = list(anomalies_index[anomalies_index == True].index)
    label = [i for i,x in enumerate(label) if x == True]
    #plot(new_df, anomaly_pred=anomalies_KMeans, axes = ax[0], ts_linewidth=1, ts_markersize=3, ap_color='red', ap_alpha=0.3, curve_group='all')
    print("t-SNE....")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(data)
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)
    plt.title("T-SNE result", fontsize=20)
    for i in range(X_tsne.shape[0]):
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], marker = '*', color = '#000000')
    for i in range(len(y)):
        plt.scatter(X_tsne[y[i],0], X_tsne[y[i], 1], marker = '*', 
                    alpha = 0.5, color = 'r')
    for i in range(len(label)):
        plt.scatter(X_tsne[label[i],0], X_tsne[label[i], 1], marker = '*', 
                    alpha = 0.5, color = 'b')
    plt.xticks([])
    plt.yticks([])
    plt.show()