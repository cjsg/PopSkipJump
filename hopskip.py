import time
import logging
import torch
import numpy as np
from tracker import DiaryPage, InfoMaxStats
from abstract_attack import Attack
from infomax_gpu import get_n_from_cos, get_cos_from_n


class HopSkipJump(Attack):

    def perform_bin_search(self, original, perturbed, page=None):
        perturbed, dist_post_update = self.binary_search_batch(original, perturbed[None])
        return perturbed, dist_post_update, None

    def perform_gradient_approximation(self, perturbed, num_evals_det, delta, average, dist_post_update, estimates,
                                       page):
        return self.approximate_gradient(perturbed, num_evals_det, delta, average)

    def perform_opposite_direction_movement(self, original, perturbed):
        # Do Nothing
        return perturbed