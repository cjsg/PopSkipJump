import time
import math
import torch

from abstract_attack import Attack
from infomax_gpu import get_n_from_cos, get_cos_from_n, bin_search
from tracker import InfoMaxStats


class PopSkipJump(Attack):

    def bin_search_step(self, original, perturbed, page=None):
        perturbed, dist_post_update, s_, (tmap, xx) = self.info_max_batch(
            original, perturbed[None], self.grid_size, self.a.true_label
        )
        if page is not None:
            page.info_max_stats = InfoMaxStats(s_, tmap, xx)
        return perturbed, dist_post_update, {'s': s_}

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        theta_prob = 1 / self.grid_size  # Theta for Info-max procedure
        delta_det_unit = delta / dist_post_update  # HSJA's delta in unit scale
        delta_prob_unit = theta_prob * math.sqrt(self.d)  # PSJA's delta in unit scale
        delta_prob = dist_post_update * delta_prob_unit  # PSJA's delta

        target_cos = get_cos_from_n(num_evals_det, theta=self.theta_det, delta=delta_det_unit, d=self.d)
        num_evals_prob = get_n_from_cos(target_cos, s=estimates['s'], theta=theta_prob, delta=delta_prob_unit, d=self.d)
        page.num_eval_prob = num_evals_prob
        num_evals_prob = int(min(num_evals_prob, self.max_num_evals))
        page.time.num_evals = time.time()
        return self._gradient_estimator(perturbed, num_evals_prob, delta_prob)

    def opposite_movement_step(self, original, perturbed):
        # Go in the opposite direction
        return torch.clamp(2 * perturbed - original, self.clip_min, self.clip_max)

    def info_max_batch(self, unperturbed, perturbed_inputs, grid_size, true_label):
        border_points = []
        dists = []
        smaps = []
        for perturbed_input in perturbed_inputs:
            output = bin_search(
                unperturbed, perturbed_input, self.model_interface, d=self.d,
                grid_size=grid_size, device=self.device,
                true_label=true_label, prev_t=self.prev_t, prev_s=self.prev_s,
                prior_frac=self.prior_frac, queries=self.queries)
            nn_tmap_est = output['nn_tmap_est']
            t_map, s_map = output['tts_max'][-1]
            border_point = (1 - t_map) * unperturbed + t_map * perturbed_input
            self.prev_t, self.prev_s = t_map, s_map
            dist = self.compute_distance(unperturbed, border_point)
            border_points.append(border_point)
            dists.append(dist)
            smaps.append(s_map)
        idx = int(torch.argmin(torch.tensor(dists)))
        dist = self.compute_distance(unperturbed, perturbed_inputs[idx])
        if dist == 0:
            print("Distance is zero in search")
        out = border_points[idx]
        dist_border = self.compute_distance(out, unperturbed)
        if dist_border == 0:
            print("Distance of border point is 0")
        if self.hsja:
            self.get_decision_in_batch(out[None], freq=1,
                                       remember=True)  # this is to make the model remember the sample
        else:
            self.get_decision_in_batch(out[None], freq=self.sampling_freq * 32,
                                       remember=True)  # this is to make the model remember the sample
        return out, dist, smaps[idx], (nn_tmap_est, output['xxj'])
