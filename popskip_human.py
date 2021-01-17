import math

import matplotlib.pyplot as plt
import torch

from abstract_attack import Attack
from defaultparams import DefaultParams


class PopSkipJumpHuman(Attack):
    def __init__(self, model_interface, data_shape, device=None, params: DefaultParams = None):
        super().__init__(model_interface, data_shape, device, params)
        self.theta_prob = 1. / self.grid_size  # Theta for Info-max procedure
        if self.constraint == 'l2':
            self.delta_det_unit = self.theta_det * math.sqrt(self.d)
            self.delta_prob_unit = math.sqrt(self.d) / self.grid_size  # PSJA's delta in unit scale
        elif self.constraint == 'linf':
            self.delta_det_unit = self.theta_det * self.d
            self.delta_prob_unit = self.d / self.grid_size  # PSJA's delta in unit scale
        self.stop_criteria = params.infomax_stop_criteria

    def get_input_from_console(self, msg):
        try:
            res = int(input(msg).strip())
            assert res == 0 or res == 1
        except:
            res = self.get_input_from_console('Invalid Input! Enter again: ')
        return res

    """
        Asks human if x is closer to x_star or x_tilde?
        Returns 1 if human says x_tilde
        Returns 0 if human says x_star
    """

    def decision_bin(self, x, x_star, x_tilde):
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(x, cmap='gray')
        axarr[0].set_title('x')
        axarr[1].imshow(x_star, cmap='gray')
        axarr[1].set_title('x_star')
        axarr[2].imshow(x_tilde, cmap='gray')
        axarr[2].set_title('x_tilde')
        plt.suptitle('Which is more close to x? Press 0 for x_star and 1 for x_tilde')
        plt.show()
        res = self.get_input_from_console("Which is more close to x? Press 0 for x_star and 1 for x_tilde: ")
        return res

    """
        Asks human if x1 is closer to x_star or x2 is closer to x_star?
        Returns 1 if human says x1
        Returns 0 if human says x0
    """

    def decision_grad(self, x0, x1, x_star):
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(x0, cmap='gray')
        axarr[0].set_title('x0')
        axarr[1].imshow(x1, cmap='gray')
        axarr[1].set_title('x1')
        axarr[2].imshow(x_star, cmap='gray')
        axarr[2].set_title('x_star')
        plt.suptitle('Which is more close to x_star? Press 0 for x0 and 1 for x1')
        plt.show()
        res = self.get_input_from_console("Which is more close to x_star? Press 0 for x0 and 1 for x1: ")
        return res

    def binary_search(self, unperturbed, perturbed):
        # Compute distance between each of perturbed and unperturbed input.
        dists_post_update = self.compute_distance(unperturbed, perturbed)
        if self.constraint == "linf":
            high = dists_post_update
            # Stopping criteria.
            threshold = dists_post_update * self.theta_prob / 2
        else:
            high = 1.0
            threshold = self.theta_prob / 2
        low = 0.0

        while high - low > threshold:
            mid = (high + low) / 2.0
            mid_input = self.project(unperturbed, perturbed[None], [mid])[0]
            decision = self.decision_bin(mid_input, unperturbed, perturbed)
            if decision == 0:
                low = mid
            else:
                high = mid
        out = self.project(unperturbed, perturbed[None], [high])[0]
        return out, dists_post_update

    def bin_search_step(self, original, perturbed, page=None, estimates=None, step=None):
        perturbed, dist_post_update = self.binary_search(original, perturbed)
        return perturbed, dist_post_update, None

    def human_gradient_estimator(self, sample, num_evals, delta, x_star):
        num_evals = int(num_evals)
        rv = self.generate_random_vectors(num_evals)
        decisions = torch.zeros(num_evals, device=self.device)
        for i in range(num_evals):
            perturbed_1 = sample + delta * rv[i]
            perturbed_2 = sample - delta * rv[i]
            perturbed_1 = torch.clamp(perturbed_1, self.clip_min, self.clip_max)
            perturbed_2 = torch.clamp(perturbed_2, self.clip_min, self.clip_max)
            rv[i] = (perturbed_1 - sample) / delta
            decisions[i] = self.decision_grad(perturbed_1, perturbed_2, x_star)
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = 2 * decisions.view(decision_shape) - 1.0
        # Baseline subtraction (when fval differs)
        if torch.abs(torch.mean(fval)) == 1.0:
            vals = fval
        else:
            vals = fval - torch.mean(fval)
        sum_directions = torch.sum(vals * rv, dim=0)
        gradf = sum_directions / num_evals
        gradf = gradf / torch.norm(gradf)
        return gradf, sum_directions, num_evals

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        if self.constraint == "l2":
            delta_prob_unit = self.theta_prob * math.sqrt(self.d)  # PSJA's delta in unit scale
        elif self.constraint == "linf":
            delta_prob_unit = self.theta_prob * self.d  # PSJA's delta in unit scale
        else:
            raise RuntimeError
        delta_prob = dist_post_update * delta_prob_unit  # PSJA's delta
        # TODO: change num of queries later
        return self.human_gradient_estimator(perturbed, 10, delta_prob, self.diary.original)[0]

    def opposite_movement_step(self, original, perturbed):
        return perturbed

    def geometric_progression_for_stepsize(self, x, update, dist, current_iteration, original=None):
        return dist / math.sqrt(current_iteration)
