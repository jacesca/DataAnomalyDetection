"""
Clustering, in the context of anomaly detection, is a technique used to group
data points into clusters or groups based on their similarity or proximity.

How Clustering is Applied in Anomaly Detection
1. Data Representation:
    Before applying clustering, the data is usually transformed or represented
    in a suitable format.
    For instance, numerical features may need to be standardized or normalized,
    and categorical features may be one-hot encoded or otherwise prepared.
2. Clustering Algorithm:
    A clustering algorithm, such as K-Means, DBSCAN, hierarchical clustering,
    or Gaussian Mixture Models (GMM), is applied to the prepared data.
    These algorithms group similar data points together based on distance
    metrics or probabilistic models.
3. Cluster Formation:
    The algorithm partitions the data into clusters. Each cluster contains data
    points that are similar or closely related to each other in some way, such
    as in terms of distance or density.
4. Anomaly Detection:
    Anomalies or outliers are detected by assessing how well data points fit
    within their respective clusters.
    Data points that are significantly different from their cluster's
    characteristics are considered anomalies.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris


# Set environment
SEED = 42
np.random.seed(SEED)

plt.rcParams.update({'axes.labelsize': 6, 'xtick.labelsize': 6,
                     'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'font.size': 6, 'axes.titlesize': 8,
                     'figure.titlesize': 10})


class AnomaliesDetection():
    """Anomalies detection usig DBScan."""
    def __init__(self, data, eps=0.35, min_samples=6):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.anomalies = []

    def autlier_detection(self):
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(self.data)

        self.outliers_index = np.where(labels == -1)
        self.normal_index = np.where(labels == 1)

        self.anomalies = self.data.loc[self.outliers_index]
        return self.anomalies

    def plot_outliers(self):
        if len(self.anomalies) > 0:
            columns = self.data.columns
            plt.figure()
            plt.scatter(x=self.data.loc[self.normal_index, columns[0]],
                        y=self.data.loc[self.normal_index, columns[1]],
                        c='blue', label='Normal data', alpha=.6)
            plt.scatter(x=self.data.loc[self.outliers_index, columns[0]],
                        y=self.data.loc[self.outliers_index, columns[1]],
                        c='red', marker='x', label='Outliers', s=100)
            plt.title('DBSCAN Outlier Detection')
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.legend()
            plt.tight_layout()
            plt.show()
            plt.style.use('default')


if __name__ == '__main__':
    # Read the data
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris.data, iris.target],
        columns=iris.feature_names + ['target']
    )
    # print(df.head())

    # Select a specific feature to analize: MedInc
    data = df.iloc[:, :2]

    # Detect anomalies
    data_quality = AnomaliesDetection(data)
    outliers = data_quality.autlier_detection()
    print(f'\n\nAnomalies detection using DBSCAN ML: {len(outliers)} cases')
    print(outliers)
    data_quality.plot_outliers()
