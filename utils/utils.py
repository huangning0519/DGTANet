import numpy as np

class StandardScaler:
    def __init__(self, data):
        self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        self.std = np.std(data, axis=(0, 1, 2), keepdims=True)

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, data):
        return (data * (self.std + 1e-6)) + self.mean
