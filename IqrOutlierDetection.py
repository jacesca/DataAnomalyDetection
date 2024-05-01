import numpy as np


# Set environment
np.random.seed(42)


class AnomaliesDetection():
    """Anomalies detection usig IQR.
    It's based on the spread of data around the median."""
    def __init__(self, data):
        self.data = np.array(data)
        self.q1 = np.percentile(data, 25)
        self.q3 = np.percentile(data, 75)
        self.iqr = self.q3 - self.q1
        self.lower_threshold = self.q1 - 1.5 * self.iqr
        self.upper_threshold = self.q3 + 1.5 * self.iqr
        self.anomalies = []

    def autlier_detection(self):
        self.anomalies = data[(data < self.lower_threshold) | (data > self.upper_threshold)]  # noqa
        return self.anomalies


if __name__ == '__main__':
    # Generate synthetic data with some outliers
    data = np.concatenate([np.random.normal(0, 1, 100),
                           np.random.normal(15, 5, 10)])

    # Detect anomalies
    print('\n\nAnomalies detection using IQR rule')
    data_quality = AnomaliesDetection(data)
    outliers = data_quality.autlier_detection()
    print(outliers)
