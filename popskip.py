import time
import math
import numpy as np
import torch

from abstract_attack import Attack
from infomax_gpu import get_n_from_cos, get_cos_from_n, bin_search
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
        idx = np.argmin(np.array(dists))
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

    def approximate_decisions(self, x, grad_queries):
        # returns 1 if adversarial
        probs = []
        num_batchs = int(math.ceil(len(x) * 1.0 / self.batch_size))
        for j in range(num_batchs):
            current_batch = x[self.batch_size * j: self.batch_size * (j + 1)]
            out = self.model_interface.get_probs_(images=current_batch)
            out = out[:, self.a.true_label]
            probs.append(out)
        probs = torch.cat(probs, dim=0)
        probs = probs.repeat(grad_queries)
        outs = self.model_interface.sample_bernoulli(1-probs)
        return outs

    def approximate_gradient(self, sample, num_evals, delta, average=False, grad_queries=1):
        """ Computes an approximation by querying every point `grad_queries` times"""
        # Generate random vectors.
        noise_shape = [int(num_evals/grad_queries)] + list(self.shape)
        perturbed, rv = self.generate_random_vectors(delta, noise_shape, sample)
        # query the model.
        outputs = self.approximate_decisions(perturbed, grad_queries)
        rv = rv.repeat(grad_queries, *([1]*(len(noise_shape)-1)))
        return self.calculate_grad(outputs, rv)

