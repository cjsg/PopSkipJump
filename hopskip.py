import math
import torch
import logging

from abstract_attack import Attack
from defaultparams import DefaultParams


class HopSkipJump(Attack):
    """
        Implements Original HSJA.
        When repeat_queries=1, it is same as vanilla HSJA.
    """

    def __init__(self, model_interface, data_shape, device=None, params: DefaultParams = None):
        super().__init__(model_interface, data_shape, device, params)
        self.grad_queries = 1  # Original HSJA does not perform multiple queries
        self.repeat_queries = 1

    def bin_search_step(self, original, perturbed, page=None, estimates=None, step=None):
        perturbed, dist_post_update = self.binary_search_batch(original, perturbed[None])
        return perturbed, dist_post_update, None

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        return self._gradient_estimator(perturbed, num_evals_det, delta)

    def opposite_movement_step(self, original, perturbed):
        # Do Nothing
        return perturbed

    def binary_search_batch(self, unperturbed, perturbed_inputs, cosine=False):
        """ Binary search to approach the boundary. """

        # Compute distance between each of perturbed and unperturbed input.
        dists_post_update = torch.tensor([
            self.compute_distance(unperturbed, perturbed_x)
            for perturbed_x in perturbed_inputs
        ])

        # Choose upper thresholds in binary searchs based on constraint.
        if self.constraint == "linf":
            highs = dists_post_update
            # Stopping criteria.
            thresholds = dists_post_update * self.theta_det
        else:
            highs = torch.ones(len(perturbed_inputs), device=self.device)
            # thresholds = self.theta * 1000  # remove 1000 later
            thresholds = self.theta_det  # remove 1000 later
            if cosine:
                thresholds /= self.d

        lows = torch.zeros(len(perturbed_inputs), device=self.device)

        # Call recursive function.
        while torch.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs, mids)

            # Update highs and lows based on model decisions.
            decisions = self.decision_by_repetition(mid_inputs)
            lows = torch.where(decisions == 0, mids, lows)
            highs = torch.where(decisions == 1, mids, highs)

        out_inputs = self.project(unperturbed, perturbed_inputs, highs)

        # Compute distance of the output to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = torch.tensor([self.compute_distance(unperturbed, out) for out in out_inputs])
        idx = torch.argmin(dists)

        dist = dists_post_update[idx]
        out = out_inputs[idx]
        return out, dist

    def project(self, unperturbed, perturbed_inputs, alphas):
        """ Projection onto given l2 / linf balls in a batch. """
        if type(alphas) != torch.Tensor:
            alphas = torch.tensor(alphas)
        alphas_shape = [len(alphas)] + [1]
        alphas = alphas.view(alphas_shape)
        if self.constraint == "l2":
            projected = (1 - alphas) * unperturbed + alphas * perturbed_inputs
        elif self.constraint == "linf":
            projected = torch.clamp(
                perturbed_inputs, unperturbed - alphas, unperturbed + alphas
            )
        else:
            raise RuntimeError(f"Unknown constraint type: {self.constraint}")
        return projected

    def geometric_progression_for_stepsize(self, x, update, dist, current_iteration, original=None):
        """ Geometric progression to search for stepsize.
          Keep decreasing stepsize by half until reaching
          the desired side of the boundary.
        """
        epsilon = dist / math.sqrt(current_iteration)
        count = 1
        while True:
            if count % 200 == 0:
                logging.warning("Decreased epsilon {} times".format(count))
            # updated = torch.clamp(x + epsilon * update, self.clip_min, self.clip_max)
            updated = x + epsilon * update
            success = (self.decision_by_repetition(updated[None]))[0]
            if success:
                break
            else:
                epsilon = epsilon / 2.0  # pragma: no cover
                count += 1
        return epsilon

    def _gradient_estimator(self, sample, num_evals, delta):
        """ Computes an approximation by querying every point `grad_queries` times"""
        # Generate random vectors.
        num_rvs = int(num_evals)
        sum_directions = torch.zeros(self.d, device=self.device)
        num_batchs = int(math.ceil(num_rvs * 1.0 / self.batch_size))
        for j in range(num_batchs):
            batch_size = min(self.batch_size, num_rvs - j * self.batch_size)
            rv = self.generate_random_vectors(batch_size)
            perturbed = sample + delta * rv
            # perturbed = torch.clamp(perturbed, self.clip_min, self.clip_max)
            rv = (perturbed - sample) / delta
            decisions = self.decision_by_repetition(perturbed)
            decision_shape = [len(decisions)] + [1]
            # Map (0, 1) to (-1, +1)
            fval = 2 * decisions.view(decision_shape) - 1.0
            # Baseline subtraction (when fval differs)
            if torch.abs(torch.mean(fval)) == 1.0:
                vals = fval
            else:
                vals = fval - torch.mean(fval)
            sum_directions = sum_directions + torch.sum(vals * rv, dim=0)
        # Get the gradient direction.
        gradf = sum_directions / num_rvs
        gradf = gradf / torch.norm(gradf)
        return gradf

    def decision_by_repetition(self, perturbed):
        decisions = self.model_interface.decision(perturbed, self.a.true_label, self.repeat_queries)
        decisions = decisions.sum(dim=1) / self.repeat_queries
        decisions = (decisions > 0.5) * 1
        for i in range(perturbed.shape[0]):
            if decisions[i] == 1:
                distance = self.a.calculate_distance(perturbed[i], self.bounds)
                if self.a.distance > distance:
                    self.a.distance = distance
                    self.a.perturbed = perturbed[i]
        return decisions


class HopSkipJumpRepeated(HopSkipJump):
    """
        Implements Original HSJA with repeated queries at every point.
        When repeat_queries=1, it is same as vanilla HSJA.
    """

    def __init__(self, model_interface, data_shape, device=None, params: DefaultParams = None):
        super().__init__(model_interface, data_shape, device, params)
        self.repeat_queries = params.hsja_repeat_queries


class HopSkipJumpRepeatedWithPSJDelta(HopSkipJump):
    """
        Implements Original HSJA with repeated queries at every point.
        Uses theta and delta from PSJA
    """

    def __init__(self, model_interface, data_shape, device=None, params: DefaultParams = None):
        super().__init__(model_interface, data_shape, device, params)
        if params.theta_fac is -1:
            tf = 1.5 * self.d * math.sqrt(self.d) / self.grid_size
        else:
            tf = params.theta_fac
        self.theta_det = self.theta_det * tf

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        # delta = dist_post_update * math.sqrt(self.d) / self.grid_size
        return self._gradient_estimator(perturbed, num_evals_det, delta)