# import numpy as np
import torch


class Adversarial:
    def __init__(self, image, label, device=None, distance='MSE'):
        self.unperturbed = torch.tensor(image).to(device)
        self.true_label = label
        self.distance_metric = distance
        self.distance = float('Inf')
        self.perturbed = None
        self.device = device

    def calculate_distance(self, x, bounds):
        if self.distance_metric == 'MSE':
            return calculate_l2_distance(self.unperturbed, x, bounds=bounds)
        else:
            return calculate_linf_distance(self.unperturbed, x, bounds=bounds)

    def set_starting_point(self, point, bounds):
        self.perturbed = torch.tensor(point).to(self.device)
        self.distance = self.calculate_distance(self.perturbed, bounds=bounds)


def calculate_l2_distance(a, b, bounds=(0, 1)):
    min_, max_ = bounds
    n = a.numel()
    f = n * (max_ - min_) ** 2
    diff = a - b
    value = torch.dot(diff.flatten(), diff.flatten()) / f
    return value


def calculate_linf_distance(a, b, bounds=(0, 1)):
    min_, max_ = bounds
    diff = (b - a) / (max_ - min_)
    value = torch.max(torch.abs(diff))
    return value
