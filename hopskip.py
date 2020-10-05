import torch

from abstract_attack import Attack


class HopSkipJump(Attack):

    def bin_search_step(self, original, perturbed, page=None):
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
        dists_post_update = torch.tensor(
            [
                self.compute_distance(unperturbed, perturbed_x)
                for perturbed_x in perturbed_inputs
            ]
        )

        # Choose upper thresholds in binary searchs based on constraint.
        if self.constraint == "linf":
            highs = dists_post_update
            # Stopping criteria.
            thresholds = dists_post_update * self.theta_det
        else:
            highs = torch.ones(len(perturbed_inputs))
            # thresholds = self.theta * 1000  # remove 1000 later
            thresholds = self.theta_det  # remove 1000 later
            if cosine:
                thresholds /= self.d

        lows = torch.zeros(len(perturbed_inputs))

        # Call recursive function.
        while torch.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs, mids)

            # Update highs and lows based on model decisions.
            # decisions = decision_function(mid_inputs, self.sampling_freq, remember=not cosine)
            decisions = self.get_decision_in_batch(mid_inputs, self.sampling_freq, remember=self.remember)
            decisions[decisions == -1] = 0
            lows = torch.where(decisions == 0, mids, lows)
            highs = torch.where(decisions == 1, mids, highs)

        out_inputs = self.project(unperturbed, perturbed_inputs, highs)

        # Compute distance of the output to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = torch.tensor([self.compute_distance(unperturbed, out) for out in out_inputs])
        idx = torch.argmin(dists)

        dist = dists_post_update[idx]
        out = out_inputs[idx]
        if self.hsja:
            self.get_decision_in_batch(out[None], freq=1,
                                       remember=True)  # this is to make the model remember the sample
        else:
            self.get_decision_in_batch(out[None], freq=self.sampling_freq * 32,
                                       remember=True)  # this is to make the model remember the sample
        return out, dist

    def project(self, unperturbed, perturbed_inputs, alphas):
        """ Projection onto given l2 / linf balls in a batch. """
        if type(alphas) != torch.Tensor:
            alphas = torch.tensor(alphas)
        alphas_shape = [len(alphas)] + [1] * len(self.shape)
        alphas = alphas.view(alphas_shape)
        if self.constraint == "l2":
            projected = (1 - alphas) * unperturbed + alphas * perturbed_inputs
        elif self.constraint == "linf":
            projected = torch.clamp(
                perturbed_inputs, unperturbed - alphas, unperturbed + alphas
            )
        return projected
