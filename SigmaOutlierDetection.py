"""
Normal Distribution Assumption:
    The 3-sigma rule assumes that the data follows a normal distribution
    (Gaussian distribution). In a normal distribution, approximately 68%
    of the data falls within one standard deviation (sigma) of the mean,
    approximately 95% falls within two standard deviations, and about
    99.7% falls within three standard deviations.
Identification of Outliers:
    According to the 3-sigma rule, data points that fall more than three
    standard deviations away from the mean are considered potential
    outliers. These data points are significantly different from the
    majority of the data and are often flagged for further investigation.
"""
import numpy as np


# Set environment
np.random.seed(42)


class AnomaliesDetection():
    """Anomalies detection usig Sigma rule."""
    def __init__(self, data):
        self.data = np.array(data)
        self.std_dev = np.std(data)
        self.mean = np.mean(data)
        self.anomalies = []

    def autlier_detection(self):
        self.anomalies = data[np.abs(self.data - self.mean) > 3 * self.std_dev]  # noqa
        return self.anomalies


if __name__ == '__main__':
    # Generate synthetic data with some outliers
    data = np.concatenate([np.random.normal(0, 1, 100),
                           np.random.normal(15, 5, 10)])

    # Detect anomalies
    print('\n\nAnomalies detection using Sigma Rule')
    data_quality = AnomaliesDetection(data)
    outliers = data_quality.autlier_detection()
    print(outliers)
