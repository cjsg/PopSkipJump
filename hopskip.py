import math
import numpy as np
import logging


class HopSkipJumpAttack:
    def __init__(self, model_interface, a, initial_num_evals=100, max_num_evals=10000, distance="MSE",
                 stepsize_search="geometric_progression", gamma=1.0, batch_size=256,
                 internal_dtype=np.float32, bounds=(0, 1), experiment='default', dataset='mnist'):
        self.model_interface = model_interface
        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.batch_size = batch_size
        self.internal_dtype = internal_dtype
        self.clip_min, self.clip_max = bounds
        self.experiment = experiment
        self.dataset = dataset

        # Set constraint based on the distance.
        if distance == 'MSE':
            self.constraint = "l2"
        elif distance == 'Linf':
            self.constraint = "linf"

        # Set binary search threshold.
        self.shape = a.unperturbed.shape
        self.d = np.prod(self.shape)
        if self.constraint == "l2":
            self.theta = self.gamma / (np.sqrt(self.d) * self.d)
        else:
            self.theta = self.gamma / (self.d * self.d)

    def attack(self, a, iterations=64):
        if a.perturbed is None:
            logging.info('Initializing Starting Point...')
            self.initialize_starting_point(a)
            logging.info('Model Calls till now: %d' % self.model_interface.model_calls)
        original = a.unperturbed.astype(self.internal_dtype)
        perturbed = a.perturbed.astype(self.internal_dtype)
        additional = {'iterations': list(), 'initial': perturbed}

        def decision_function(x):
            outs = []
            num_batchs = int(math.ceil(len(x) * 1.0 / self.batch_size))
            for j in range(num_batchs):
                current_batch = x[self.batch_size * j: self.batch_size * (j + 1)]
                out = self.model_interface.forward(current_batch.astype(self.internal_dtype), a)
                outs.append(out)
            outs = np.concatenate(outs, axis=0)
            return outs

        logging.info('Binary Search to project to boundary...')
        perturbed, dist_post_update = self.binary_search_batch(
            original, np.expand_dims(perturbed, 0), decision_function
        )
        logging.info('Model Calls till now: %d' % self.model_interface.model_calls)
        dist = self.compute_distance(perturbed, original)
        distance = a.distance
        for step in range(1, iterations + 1):
            logging.info('Step %d...' % step)
            # Choose delta.
            delta = self.select_delta(dist_post_update, step)

            # Choose number of evaluations.
            num_evals = int(
                min([self.initial_num_evals * np.sqrt(step), self.max_num_evals])
            )
            num_evals = int(num_evals)
            logging.info('Approximating grad with %d evaluation...' % num_evals)
            while True:
                gradf = self.approximate_gradient(
                    decision_function, perturbed, num_evals, delta
                )
                if gradf is None:
                    delta *= 2
                else:
                    break

            if self.constraint == "linf":
                update = np.sign(gradf)
            else:
                update = gradf

            logging.info('Binary Search back to the boundary')
            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilon = self.geometric_progression_for_stepsize(
                    perturbed, update, dist, decision_function, step
                )

                # Update the sample.
                perturbed = np.clip(
                    perturbed + epsilon * update, self.clip_min, self.clip_max
                )

                # Binary search to return to the boundary.
                perturbed, dist_post_update = self.binary_search_batch(
                    original, perturbed[None], decision_function
                )

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = np.clip(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = decision_function(perturbeds)

                if (idx_perturbed == 1).any():
                    # Select the perturbation that yields the minimum
                    # distance after binary search.
                    perturbed, dist_post_update = self.binary_search_batch(
                        original, perturbeds[idx_perturbed], decision_function
                    )

            # compute new distance.
            dist = self.compute_distance(perturbed, original)

            if self.constraint == "l2":
                distance = dist ** 2 / self.d / (self.clip_max - self.clip_min) ** 2
            elif self.constraint == "linf":
                distance = dist / (self.clip_max - self.clip_min)
            logging.info('Model Calls till now: %d' % self.model_interface.model_calls)
            logging.info('distance of adversarial = %f', distance)
            additional['iterations'].append(a.perturbed)
        return additional

    def initialize_starting_point(self, a):
        success = 0
        num_evals = 0

        while True:
            random_noise = np.random.uniform(
                self.clip_min, self.clip_max, size=self.shape
            ).astype(self.internal_dtype)

            success = self.model_interface.forward_one(random_noise, a)
            # when model is confused, it is not adversarial
            num_evals += 1
            if success == 1:
                break
            if num_evals > 1e4:
                return

        # Binary search to minimize l2 distance to the original input.
        low = 0.0
        high = 1.0
        # while high - low > 0.001:
        while high - low > 0.2:
            mid = (high + low) / 2.0
            blended = (1 - mid) * a.unperturbed + mid * random_noise
            success = self.model_interface.forward_one(blended, a)
            # when model is confused, it is not adversarial
            logging.info(a.distance)
            if success == 1:
                high = mid
            else:
                low = mid

    def binary_search_batch(self, unperturbed, perturbed_inputs, decision_function):
        """ Binary search to approach the boundary. """

        # Compute distance between each of perturbed and unperturbed input.
        dists_post_update = np.array(
            [
                self.compute_distance(unperturbed, perturbed_x)
                for perturbed_x in perturbed_inputs
            ]
        )

        # Choose upper thresholds in binary searchs based on constraint.
        if self.constraint == "linf":
            highs = dists_post_update
            # Stopping criteria.
            thresholds = dists_post_update * self.theta
        else:
            highs = np.ones(len(perturbed_inputs))
            thresholds = self.theta * 1000  # remove 1000 later

        lows = np.zeros(len(perturbed_inputs))

        # Call recursive function.
        while np.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs, mids)

            # Update highs and lows based on model decisions.
            decisions = decision_function(mid_inputs)
            decisions[decisions == -1] = 0
            lows = np.where(decisions == 0, mids, lows)
            highs = np.where(decisions == 1, mids, highs)

        out_inputs = self.project(unperturbed, perturbed_inputs, highs)

        # Compute distance of the output to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = np.array(
            [self.compute_distance(unperturbed, out) for out in out_inputs]
        )
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out = out_inputs[idx]
        return out, dist

    def select_delta(self, dist_post_update, current_iteration):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.
        """
        if current_iteration == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.constraint == "l2":
                delta = np.sqrt(self.d) * self.theta * dist_post_update
            elif self.constraint == "linf":
                delta = self.d * self.theta * dist_post_update

        return delta * 10

    def approximate_gradient(self, decision_function, sample, num_evals, delta):
        """ Gradient direction estimation """
        # Generate random vectors.
        noise_shape = [num_evals] + list(self.shape)
        if self.constraint == "l2":
            rv = np.random.randn(*noise_shape)
        elif self.constraint == "linf":
            rv = np.random.uniform(low=-1, high=1, size=noise_shape)

        axis = tuple(range(1, 1 + len(self.shape)))
        rv = rv / np.sqrt(np.sum(rv ** 2, axis=axis, keepdims=True))
        perturbed = sample + delta * rv
        perturbed = np.clip(perturbed, self.clip_min, self.clip_max)
        rv = (perturbed - sample) / delta

        # query the model.
        outputs = decision_function(perturbed)
        decisions = outputs[outputs != -1]
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = 2 * decisions.astype(self.internal_dtype).reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        vals = fval if abs(np.mean(fval)) == 1.0 else fval - np.mean(fval)
        rv = rv[outputs != -1]
        gradf = np.mean(vals * rv, axis=0)

        # Get the gradient direction.
        if np.sum(outputs != -1) == 0:
            return None
        gradf = gradf / np.linalg.norm(gradf)

        return gradf

    def geometric_progression_for_stepsize(
            self, x, update, dist, decision_function, current_iteration
    ):
        """ Geometric progression to search for stepsize.
          Keep decreasing stepsize by half until reaching
          the desired side of the boundary.
        """
        epsilon = dist / np.sqrt(current_iteration)
        while True:
            updated = np.clip(x + epsilon * update, self.clip_min, self.clip_max)
            success = (decision_function(updated[None]))[0]
            if success:
                break
            else:
                epsilon = epsilon / 2.0  # pragma: no cover

        return epsilon

    def compute_distance(self, x1, x2):
        if self.constraint == "l2":
            return np.linalg.norm(x1 - x2)
        elif self.constraint == "linf":
            return np.max(abs(x1 - x2))

    def project(self, unperturbed, perturbed_inputs, alphas):
        """ Projection onto given l2 / linf balls in a batch. """
        alphas_shape = [len(alphas)] + [1] * len(self.shape)
        alphas = alphas.reshape(alphas_shape)
        if self.constraint == "l2":
            projected = (1 - alphas) * unperturbed + alphas * perturbed_inputs
        elif self.constraint == "linf":
            projected = np.clip(
                perturbed_inputs, unperturbed - alphas, unperturbed + alphas
            )
        return projected.astype(self.internal_dtype)
