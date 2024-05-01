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

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


# Set environment
SEED = 42
np.random.seed(SEED)


class AnomaliesDetection():
    """Anomalies detection usig KMeans."""
    def __init__(self, data, n_cluster, threshold_percentile=95):
        self.scaler = StandardScaler()

        self.data = data
        self.n_cluster = n_cluster
        self.threshold_percentile = threshold_percentile
        self.data_scaled = self.scaler.fit_transform(self.data)
        self.anomalies = []

    def autlier_detection(self):
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=SEED)
        kmeans.fit(self.data_scaled)

        labels = kmeans.predict(self.data_scaled)
        cluster_center = kmeans.cluster_centers_
        distances = np.linalg.norm(self.data_scaled - cluster_center[labels],
                                   axis=1)
        threshold = np.percentile(distances, self.threshold_percentile)
        self.anomalies = self.data[distances > threshold]
        return self.anomalies


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
    data_quality = AnomaliesDetection(data, 5, 95)
    outliers = data_quality.autlier_detection()
    print(f'\n\nAnomalies detection using KMeans ML: {len(outliers)} cases')
    print(outliers)
