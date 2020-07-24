import math
import numpy as np
import logging
from adversarial import Adversarial
from numpy import dot
from numpy.linalg import norm

class HopSkipJumpAttack:
    def __init__(self, model_interface, data_shape, initial_num_evals=100, max_num_evals=10000, distance="MSE",
                 stepsize_search="geometric_progression", gamma=1.0, batch_size=256,
                 internal_dtype=np.float32, bounds=(0, 1), experiment='default', dataset='mnist',
                 sampling_freq=10, grad_sampling_freq=10):
        self.model_interface = model_interface
        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.batch_size = batch_size
        self.internal_dtype = internal_dtype
        self.bounds = bounds
        self.clip_min, self.clip_max = bounds
        self.experiment = experiment
        self.dataset = dataset
        self.sampling_freq = sampling_freq
        self.grad_sampling_freq = grad_sampling_freq

        # Set constraint based on the distance.
        if distance == 'MSE':
            self.constraint = "l2"
        elif distance == 'Linf':
            self.constraint = "linf"

        # Set binary search threshold.
        self.shape = data_shape
        self.d = np.prod(self.shape)
        if self.constraint == "l2":
            self.theta = self.gamma / (np.sqrt(self.d) * self.d)
            # self.theta = self.gamma / (np.sqrt(self.d))  # Based on CJ experiment
        else:
            self.theta = self.gamma / (self.d * self.d)

    def attack(self, images, labels, starts=None, iterations=64, average=False):
        raw_results = []
        distances = []
        for i, (image, label) in enumerate(zip(images, labels)):
            logging.warning("Attacking Image: {}".format(i))
            a = Adversarial(image=image, label=label)
            if starts is not None:
                a.set_starting_point(starts[i], self.bounds)
            results = self.attack_one(a, iterations, average)
            if results is None:
                results = {}
                logging.error("Skipping image: Model Prediction of input does not match label")
            else:
                distances.append(a.distance)
            results['true_label'] = label
            results['original'] = image
            raw_results.append(results)
        median = np.median(np.array(distances))
        return median, raw_results

    def attack_one(self, a, iterations=64, average=False):
        if self.model_interface.forward_one(a.unperturbed, a, self.sampling_freq) == 1:
            return None
        if a.perturbed is None:
            logging.info('Initializing Starting Point...')
            self.initialize_starting_point(a)
            logging.info('Model Calls till now: %d' % self.model_interface.model_calls)
        original = a.unperturbed.astype(self.internal_dtype)
        perturbed = a.perturbed.astype(self.internal_dtype)
        additional = {'iterations': list(), 'initial': perturbed, 'manifold': list(), 'cosine_details': list()}

        def decision_function(x, freq, average=False):
            outs = []
            num_batchs = int(math.ceil(len(x) * 1.0 / self.batch_size))
            for j in range(num_batchs):
                current_batch = x[self.batch_size * j: self.batch_size * (j + 1)]
                out = self.model_interface.forward(current_batch.astype(self.internal_dtype), a, freq, average)
                outs.append(out)
            outs = np.concatenate(outs, axis=0)
            return outs

        # ss = self.screenshot_manifold(perturbed, original)
        # additional['manifold_init'] = ss
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
                    decision_function, perturbed, num_evals, delta, average
                )
                if gradf is None:
                    # delta *= 2
                    raise RuntimeError
                else:
                    break

            if self.constraint == "linf":
                update = np.sign(gradf)
            else:
                update = gradf

            cos_details = self.capture_cosines(perturbed, original, gradf, a.true_label)
            additional['cosine_details'].append(cos_details)

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

                # ss = self.screenshot_manifold(perturbed, original)
                # additional['manifold'].append(ss)

                # Binary search to return to the boundary.
                perturbed, dist_post_update = self.binary_search_batch(
                    original, perturbed[None], decision_function
                )
                # perturbed, dist_post_update = self.info_max_batch(
                #     original, perturbed[None], decision_function
                # )
                # _check = decision_function(perturbed[None], self.sampling_freq)[0]
                # assert _check == 1

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = np.clip(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = decision_function(perturbeds, self.sampling_freq)

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
            additional['iterations'].append({'perturbed': a.perturbed, 'distance': a.distance})
        return additional

    def screenshot_manifold(self, perturbed, original):
        alphas = np.linspace(0, 1, 201)
        screenshot = {}
        for alpha in alphas:
            point = (2*alpha-1) * original + (2-2*alpha) * perturbed
            probs = self.model_interface.get_probs(point)
            screenshot[alpha] = probs.flatten()
        return screenshot

    def capture_cosines(self, perturbed, original, gradf, true_label):
        grad_st_line = perturbed - original
        grad_true = self.model_interface.get_grads(perturbed[None], true_label)
        _g_true = grad_true.flatten()
        _g_line = grad_st_line.flatten()
        _g_estm = gradf.flatten()
        cos_true_vs_line = dot(_g_true, _g_line) / (norm(_g_true) * norm(_g_line))
        cos_true_vs_estm = dot(_g_true, _g_estm) / (norm(_g_true) * norm(_g_estm))
        return {'true_vs_line': abs(cos_true_vs_line), 'true_vs_estm': abs(cos_true_vs_estm)}

    def initialize_starting_point(self, a):
        success = 0
        num_evals = 0

        while True:
            random_noise = np.random.uniform(
                self.clip_min, self.clip_max, size=self.shape
            ).astype(self.internal_dtype)

            success = self.model_interface.forward_one(random_noise, a, self.sampling_freq)
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
            success = self.model_interface.forward_one(blended, a, self.sampling_freq)
            # when model is confused, it is not adversarial
            logging.info(a.distance)
            if success == 1:
                high = mid
            else:
                low = mid

    def info_max_batch(self, unperturbed, perturbed_inputs, decision_function):
        border_points = []
        dists = []
        for perturbed_input in perturbed_inputs:
            a, b = 0, 1  # interval [a, b]
            Nx = Nt = 101  # discretization number
            kmax = 5000  # number of sample points

            # discretize parameter (search) space
            ts = np.linspace(a, b, Nt)
            xs = np.linspace(a, b, Nx)
            ys = [0, 1]
            ss = 10 ** np.linspace(-2, 0, 11)  # s \in [.01, 1.]
            epss = np.linspace(0., .1, 2)

            def f_py_xts(y, t, x, s, eps):
                # y : int in {0,1}
                # t : float in [a,b]
                # x : float in [0,1]
                # s : positive float
                # eps: float in [0., .5]

                sigmoid = lambda x: .5 * np.tanh(x) + .5
                p = eps + (1 - 2 * eps) * sigmoid((x - t) / s)
                return y * p + (1 - y) * (1 - p)
            f_py_xts = np.vectorize(f_py_xts)

            Y, T, X, S, E = np.meshgrid(ys, ts, xs, ss, epss, indexing='ij')
            py_txse = f_py_xts(Y, T, X, S, E)  # [y, x, t, s, eps] axis always in this order
            py_tx = py_txse.sum(axis=(3, 4))  # marginalizing out s and eps
            pt_x = np.ones((1, Nt, 1)) / Nt  # prior on t

            xjs = []
            yjs = []
            t_map, t_max = -1, -1
            for k in range(kmax):  # k = query index; kmax = total nbr of queries
                _this_t_map = (pt_x.flatten() * ts).sum()  # Mean a posteriori (or prior mean)
                _this_t_max = ts[np.argmax(pt_x.flatten())]  # Maximum a posteriori (or prior max)
                if abs(t_map - _this_t_map) < 1e-4 and abs(t_max - _this_t_max) < 1e-4:
                    break
                else:
                    t_map, t_max = _this_t_map, _this_t_max

                # Compute mutual information I(y, t | (x1,y1), (x2, y2) ... (xj, yj))
                pyt_x = py_tx * pt_x  # pt_xs = pt_x
                py_x = pyt_x.sum(axis=1, keepdims=True)
                I_x = (pyt_x * np.log(pyt_x / (py_x * pt_x + 1e-8) + 1e-8)).sum(axis=(0, 1))

                # Maximize mutual info and sample
                imax = np.argmax(I_x)
                xj = xs[imax]
                projection = (1-xj)*unperturbed+xj*perturbed_input
                yj = int(decision_function(projection[None], freq=1))
                xjs.append(xj)
                yjs.append(yj)

                pyj_txj = py_tx[yj:(yj + 1), :, imax:(imax + 1)]
                pyj_xj = py_x[yj:(yj + 1), :, imax:(imax + 1)]
                pt_x = pyj_txj * pt_x / pyj_xj

            border_point = (1-t_map)*unperturbed + t_map*perturbed_input
            dist = self.compute_distance(unperturbed, border_point)
            border_points.append(border_point)
            dists.append(dist)
        idx = np.argmin(np.array(dists))
        dist = self.compute_distance(unperturbed, perturbed_inputs[idx])
        out = border_points[idx]
        return out, dist

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
            # thresholds = self.theta * 1000  # remove 1000 later
            thresholds = self.theta  # remove 1000 later

        lows = np.zeros(len(perturbed_inputs))

        # Call recursive function.
        while np.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs, mids)

            # Update highs and lows based on model decisions.
            decisions = decision_function(mid_inputs, self.sampling_freq)
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

        # return delta * 10  # Did it for noisy models
        return delta

    def approximate_gradient(self, decision_function, sample, num_evals, delta, average=False):
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
        outputs = decision_function(perturbed, self.grad_sampling_freq, average)
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
            success = (decision_function(updated[None], self.sampling_freq))[0]
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
