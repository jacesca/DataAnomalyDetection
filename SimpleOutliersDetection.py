import numpy as np


# Set environment
np.random.seed(42)


class AnomaliesDetection():
    """Custom class where a threshold is defined."""
    def __init__(self, data):
        self.data = data
        self.mean = np.mean(data)
        self.std_dev = np.std(data)
        self.anomalies = []

    def rule_based_anomaly_detection(self, threshold=3):
        anomalies = []
        for i, value in enumerate(self.data):
            if abs(value - self.mean) > threshold * self.std_dev:
                anomalies.append((i, value))

        self.anomalies = anomalies
        self.print_anomalies()

    def euclidean_distance_anomaly_detection(self, threshold=3):
        anomalies = []
        for i, value in enumerate(self.data):
            euclidean_dist = np.sqrt((value - self.mean) ** 2)
            if euclidean_dist > threshold:
                anomalies.append((i, value))

        self.anomalies = anomalies
        self.print_anomalies()

    def print_anomalies(self):
        if len(self.anomalies) > 0:
            print('Detected anomalies:')
            for i, value in enumerate(self.anomalies):
                print(f'Index {i}: value: {value}')
        else:
            print('No anomalies detected.')


if __name__ == '__main__':
    threshold = 3

    # Generate synthetic data with some outliers
    data = np.concatenate([np.random.normal(0, 1, 100),
                           np.random.normal(15, 5, 10)])

    # Detect anomalies
    print('\n\nUsing Rule Based Method')
    data_quality = AnomaliesDetection(data)
    data_quality.rule_based_anomaly_detection(threshold)

    print('\n\nUsing Euclidean Based Method')
    data_quality.euclidean_distance_anomaly_detection(threshold)
