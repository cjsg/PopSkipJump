import numpy as np
from distances import MSE


class Adversarial:
    def __init__(self, image, label, distance=MSE):
        self.unperturbed = image.astype(np.float32)
        self.true_label = label
        self.distance_metric = distance
        self.distance = distance(value=np.inf)

    def calculate_distance(self, x, bounds):
        return self.distance_metric(self.unperturbed, x, bounds=bounds)
