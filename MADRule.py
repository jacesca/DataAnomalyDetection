"""
The MAD (Median Absolute Deviation) rule
    is a statistical outlier detectionmethod that uses the median and the
    median absolute deviation as robust estimators to identify outliers in
    a dataset.
    It is particularly useful when dealing with data that may not follow a
    normal distribution or when there are potential outliers that can
    significantly impact the mean and standard deviation.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


# Set environment
np.random.seed(42)


class AnomaliesDetection():
    """Anomalies detection usig MAD Rule."""
    def __init__(self, data):
        self.data = data
        self.median = np.median(self.data)
        self.abs_diff = np.abs(self.data - self.median)
        self.mad = np.median(self.abs_diff)
        self.threshold = None
        self.outlier_threshold = None
        self.anomalies = []

    def autlier_detection(self, threshold=3):
        self.threshold = threshold
        self.outlier_threshold = self.threshold * self.mad
        self.anomalies = self.data[self.abs_diff > self.outlier_threshold]  # noqa
        return self.anomalies


if __name__ == '__main__':
    # Read the data
    california_housing = fetch_california_housing()
    df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)  # noqa
    print(df.head())

    # Select a specific feature to analize: MedInc
    data = df['MedInc']

    # Detect anomalies
    print('\n\nAnomalies detection using MAD rule')
    data_quality = AnomaliesDetection(data)
    outliers = data_quality.autlier_detection(threshold=5)
    print(outliers)
