import time
import math
import torch

from abstract_attack import Attack
from infomax_gpu import get_n_from_cos, get_cos_from_n, bin_search
from tracker import InfoMaxStats


class PopSkipJump(Attack):

    def bin_search_step(self, original, perturbed, page=None):
        perturbed, dist_post_update, s_, e_, (tmap, xx) = self.info_max_batch(
            original, perturbed[None], self.a.true_label)
        if page is not None:
            page.info_max_stats = InfoMaxStats(s_, tmap, xx, e_)
        return perturbed, dist_post_update, {'s': s_, 'e': e_}

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        theta_prob = 1 / self.grid_size  # Theta for Info-max procedure
        delta_det_unit = delta / dist_post_update  # HSJA's delta in unit scale
        delta_prob_unit = theta_prob * math.sqrt(self.d)  # PSJA's delta in unit scale
        delta_prob = dist_post_update * delta_prob_unit  # PSJA's delta

        target_cos = get_cos_from_n(num_evals_det, theta=self.theta_det, delta=delta_det_unit, d=self.d)
        num_evals_prob = get_n_from_cos(
            target_cos, s=estimates['s'], theta=theta_prob,
            delta=delta_prob_unit, d=self.d, eps=estimates['e'])
        page.num_eval_prob = num_evals_prob
        num_evals_prob = int(min(num_evals_prob, self.max_num_evals))
        page.time.num_evals = time.time()
        return self._gradient_estimator(perturbed, num_evals_prob, delta_prob)

    def opposite_movement_step(self, original, perturbed):
        # Go in the opposite direction
        return torch.clamp(perturbed + 0.5 * (perturbed - original), self.clip_min, self.clip_max)

    def info_max_batch(self, unperturbed, perturbed_inputs, true_label):
        border_points = []
        dists = []
        smaps = []
        emaps = []
        for perturbed_input in perturbed_inputs:
            output = bin_search(
                unperturbed, perturbed_input, self.model_interface, d=self.d,
                grid_size=self.grid_size, device=self.device,
                true_label=true_label, prev_t=self.prev_t, prev_s=None,
                prev_e=None, prior_frac=self.prior_frac,
                queries=self.queries, plot=False)
            nn_tmap_est = output['nn_tmap_est']
            t_map, s_map, e_map = output['ttse_max'][-1]
            border_point = (1 - t_map) * unperturbed + t_map * perturbed_input
            self.prev_t, self.prev_s, self.prev_e = t_map, s_map, e_map
            dist = self.compute_distance(unperturbed, border_point)
            border_points.append(border_point)
            dists.append(dist)
            smaps.append(s_map)
            emaps.append(e_map)
        # print('e_map=', e_map)
        idx = int(torch.argmin(torch.tensor(dists)))
        dist = self.compute_distance(unperturbed, perturbed_inputs[idx])
        if dist == 0:
            print("Distance is zero in search")
        out = border_points[idx]
        dist_border = self.compute_distance(out, unperturbed)
        if dist_border == 0:
            print("Distance of border point is 0")
        return out, dist, smaps[idx], emaps[idx], (nn_tmap_est, output['xxj'])

    def geometric_progression_for_stepsize(self, x, update, dist, current_iteration, original=None):
        return dist / math.sqrt(current_iteration)
