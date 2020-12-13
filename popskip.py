import time
import math
import torch

from abstract_attack import Attack
from defaultparams import DefaultParams
from infomax import get_n_from_cos, get_cos_from_n, bin_search
from tracker import InfoMaxStats


class PopSkipJump(Attack):
    def __init__(self, model_interface, data_shape, device=None, params: DefaultParams = None):
        super().__init__(model_interface, data_shape, device, params)
        self.delta_det_unit = self.theta_det * math.sqrt(self.d)
        self.theta_prob = 1. / self.grid_size # Theta for Info-max procedure
        self.delta_prob_unit = math.sqrt(self.d) / self.grid_size  # PSJA's delta in unit scale
        self.stop_criteria = params.infomax_stop_criteria

    def bin_search_step(self, original, perturbed, page=None, estimates=None, step=None):
        perturbed, dist_post_update, s_, e_, t_, n_, (nn_tmap, xx) = self.info_max_batch(
            original, perturbed[None], self.a.true_label, estimates, step)
        if page is not None:
            page.info_max_stats = InfoMaxStats(s_, t_, xx, e_, n_)
        return perturbed, dist_post_update, {'s': s_, 'e': e_, 'n': n_, 't': t_}

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        delta_prob_unit = self.theta_prob * math.sqrt(self.d)  # PSJA's delta in unit scale
        delta_prob = dist_post_update * delta_prob_unit  # PSJA's delta

        num_evals_prob = estimates['n']
        page.num_eval_prob = num_evals_prob
        num_evals_prob = int(min(num_evals_prob, self.max_num_evals))
        page.time.num_evals = time.time()
        return self._gradient_estimator(perturbed, num_evals_prob, delta_prob)

    def opposite_movement_step(self, original, perturbed):
        # Go in the opposite direction
        return torch.clamp(perturbed + 0.5 * (perturbed - original), self.clip_min, self.clip_max)

    def get_theta_prob(self, target_cos, estimates=None):
        """
        Performs binary search for finding maximal theta_prob that does not affect estimated samples
        """
        # TODO: Replace Binary Search with a closed form solution (if it exists)
        if estimates is None:
            s, eps = 100., 1e-4
            n1 = get_n_from_cos(target_cos, s=s, theta=0, delta=self.delta_prob_unit, d=self.d, eps=eps)
        else:
            s, eps = estimates['s'], estimates['e']
            n1 = get_n_from_cos(target_cos, s=s, theta=0, delta=self.delta_prob_unit, d=self.d, eps=eps)
        low, high = 0, self.theta_det
        theta = self.theta_det
        n2 = get_n_from_cos(target_cos, s=s, theta=theta, delta=self.delta_prob_unit, d=self.d, eps=eps)
        while (n2-n1) < 1:
            theta *= 2
            n2 = get_n_from_cos(target_cos, s=s, theta=theta, delta=self.delta_prob_unit, d=self.d, eps=eps)
            low, high = theta/2, theta

        while high - low >= self.theta_det:
            mid = (low + high) / 2
            n2 = get_n_from_cos(target_cos, s=s, theta=mid, delta=self.delta_prob_unit, d=self.d, eps=eps)
            if (n2 - n1) < 1:
                low = mid
            else:
                high = mid
        return low

    def info_max_batch(self, unperturbed, perturbed_inputs, true_label, estimates, step):
        border_points = []
        dists = []
        smaps, tmaps, emaps, ns = [], [], [], []
        if estimates is None:
            target_cos = get_cos_from_n(self.initial_num_evals, theta=self.theta_det, delta=self.delta_det_unit, d=self.d)
        else:
            num_evals_det = int(min([self.initial_num_evals * math.sqrt(step+1), self.max_num_evals]))
            target_cos = get_cos_from_n(num_evals_det, theta=self.theta_det, delta=self.delta_det_unit, d=self.d)
        theta_prob_dynamic = self.get_theta_prob(target_cos, estimates)
        grid_size_dynamic = min(self.grid_size, int(1 / theta_prob_dynamic) + 1)
        for perturbed_input in perturbed_inputs:
            output, n = bin_search(
                unperturbed, perturbed_input, self.model_interface, d=self.d,
                grid_size=grid_size_dynamic, device=self.device, delta=self.delta_prob_unit,
                true_label=true_label, prev_t=self.prev_t, prev_s=self.prev_s,
                prev_e=self.prev_e, prior_frac=self.prior_frac, target_cos=target_cos,
                queries=self.queries, plot=False, stop_criteria=self.stop_criteria, dist_metric=self.constraint)
            nn_tmap_est = output['nn_tmap_est']
            t_map, s_map, e_map = output['ttse_max'][-1]
            border_point = (1 - t_map) * perturbed_input + t_map * unperturbed
            self.prev_t, self.prev_s, self.prev_e = t_map, s_map, e_map
            dist = self.compute_distance(unperturbed, border_point)
            border_points.append(border_point)
            dists.append(dist)
            smaps.append(s_map)
            tmaps.append(t_map)
            emaps.append(e_map)
            ns.append(n)
        idx = int(torch.argmin(torch.tensor(dists)))
        dist = self.compute_distance(unperturbed, perturbed_inputs[idx])
        if dist == 0:
            print("Distance is zero in search")
        out = border_points[idx]
        dist_border = self.compute_distance(out, unperturbed)
        if dist_border == 0:
            print("Distance of border point is 0")
        return out, dist, smaps[idx], emaps[idx], tmaps[idx], ns[idx], (nn_tmap_est, output['xxj'])

    def geometric_progression_for_stepsize(self, x, update, dist, current_iteration, original=None):
        return dist / math.sqrt(current_iteration)


class PopSkipJumpTrueLogits(PopSkipJump):
    def bin_search_step(self, original, perturbed, page=None, estimates=None, step=None):
        dists_post_update = self.compute_distance(original, perturbed)
        if self.constraint == "linf":
            high = dists_post_update
            # Stopping criteria.
            threshold = dists_post_update * self.theta_det
        else:
            high = torch.ones(1, device=self.device)
            threshold = self.theta_det
        low = torch.zeros(1, device=self.device)
        while high - low > threshold:
            mid = (high + low) / 2.0
            mid_input = self.project(original, perturbed, mid)
            prob = self.model_interface.decision_with_logits(mid_input, self.a.true_label)[0]
            if prob[self.a.true_label] < 0.5:
                high = mid
            else:
                low = mid
        out_input = self.project(original, perturbed, high)[0]
        return out_input, dists_post_update, None

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        delta_prob_unit = self.theta_prob * math.sqrt(self.d)  # PSJA's delta in unit scale
        delta_prob = dist_post_update * delta_prob_unit  # PSJA's delta
        # TODO: This will work well for deterministic classifier only. For prob classifier, we can query for probs.
        return self._gradient_estimator(perturbed, num_evals_det, delta_prob)