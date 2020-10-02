import time

import numpy as np
import torch

from abstract_attack import Attack
from infomax_gpu import get_n_from_cos, get_cos_from_n
from tracker import InfoMaxStats


class PopSkipJump(Attack):

    def perform_bin_search(self, original, perturbed, page=None):
        perturbed, dist_post_update, s_, (tmap, xx) = self.info_max_batch(
            original, perturbed[None], self.grid_size, self.a.true_label
        )
        if page is not None:
            page.info_max_stats = InfoMaxStats(s_, tmap, xx)
        return perturbed, dist_post_update, {'s': s_}

    def perform_gradient_approximation(self, perturbed, num_evals_det, delta, average, dist_post_update, estimates,
                                       page):
        target_cos = get_cos_from_n(num_evals_det, theta=self.theta_det, delta=delta / dist_post_update,
                                    d=self.d)
        num_evals_prob = int(get_n_from_cos(target_cos, s=estimates['s'], theta=(1 / self.grid_size),
                                            delta=(np.sqrt(self.d) / self.grid_size), d=self.d))
        page.num_eval_prob = num_evals_prob
        num_evals_prob = int(min(num_evals_prob, self.max_num_evals))
        page.time.num_evals = time.time()

        gradf = self.approximate_gradient(
            perturbed, num_evals_prob, dist_post_update * np.sqrt(self.d) / self.grid_size, average
        )
        return gradf

    def perform_opposite_direction_movement(self, original, perturbed):
        # Go in the opposite direction
        return torch.clamp(2 * perturbed - original, self.clip_min, self.clip_max)
