import numpy as np


class Adversarial:
    def __init__(self, image, label, distance='MSE'):
        self.unperturbed = image.astype(np.float32)
        self.true_label = label
        self.distance_metric = distance
        self.distance = np.inf
        self.perturbed = None

    def calculate_distance(self, x, bounds):
        if self.distance_metric == 'MSE':
            return calculate_l2_distance(self.unperturbed, x, bounds=bounds)
        else:
            return calculate_linf_distance(self.unperturbed, x, bounds=bounds)

    def set_starting_point(self, point, bounds):
        self.perturbed = point
        self.distance = self.calculate_distance(point, bounds=bounds)


def calculate_l2_distance(a, b, bounds=(0, 1)):
    min_, max_ = bounds
    n = a.size
    f = n * (max_ - min_) ** 2
    diff = a - b
    value = np.vdot(diff, diff) / f
    return value


def calculate_linf_distance(a, b, bounds=(0, 1)):
    min_, max_ = bounds
    diff = (b - a) / (max_ - min_)
    value = np.max(np.abs(diff)).astype(np.float64)
    return value
