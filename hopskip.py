import math
import numpy as np
import time
import logging
from adversarial import Adversarial
from numpy import dot
import torch
from infomax_gpu import bin_search, get_n_from_cos, get_cos_from_n
from numpy.linalg import norm
from img_utils import get_device


class HopSkipJumpAttack:
    def __init__(self, model_interface, data_shape, initial_num_evals=100, max_num_evals=50000, distance="MSE",
                 stepsize_search="geometric_progression", batch_size=256, internal_dtype=np.float32, bounds=(0, 1),
                 experiment='default', device=None, params=None):
        self.model_interface = model_interface
        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.gamma = params.gamma
        self.batch_size = batch_size
        self.internal_dtype = internal_dtype
        self.bounds = bounds
        self.clip_min, self.clip_max = bounds
        self.experiment = experiment
        self.sampling_freq = params.sampling_freq_binsearch
        self.grad_sampling_freq = params.sampling_freq_approxgrad
        self.search = params.search
        self.hsja = params.hopskipjumpattack
        self.remember = params.remember_all
        self.device = device

        # Set constraint based on the distance.
        if distance == 'MSE':
            self.constraint = "l2"
        elif distance == 'Linf':
            self.constraint = "linf"

        # Set binary search threshold.
        self.shape = data_shape
        self.d = np.prod(self.shape)
        self.grid_size = params.grid_size
        self.theta_prob = 1.0/self.grid_size
        if self.constraint == "l2":
            self.theta_det = self.gamma / (np.sqrt(self.d) * self.d)
            # self.theta = self.gamma / (np.sqrt(self.d))  # Based on CJ experiment
        else:
            self.theta_det = self.gamma / (self.d * self.d)

    def attack(self, images, labels, starts=None, iterations=64, average=False, flags=None):
        raw_results = []
        distances = []
        for i, (image, label) in enumerate(zip(images, labels)):
            logging.warning("Attacking Image: {}".format(i))
            a = Adversarial(image=image, label=label, device=self.device)
            if starts is not None:
                a.set_starting_point(starts[i], self.bounds)
            results = self.attack_one(a, iterations, average, flags=flags)
            if len(results) > 0:
                distances.append(a.distance)
            results['true_label'] = label
            results['original'] = image
            raw_results.append(results)
        median = torch.median(torch.tensor(distances))
        return median, raw_results

    def attack_one(self, a, iterations=64, average=False, flags=None):
        self.model_interface.model_calls = 0
        print(type(a.unperturbed))
        # if self.model_interface.forward_one(a.unperturbed, a, self.sampling_freq) == 1:
        #     logging.error("Skipping image: Model Prediction of input does not match label")
        #     return dict()
        if a.perturbed is None:
            logging.info('Initializing Starting Point...')
            self.initialize_starting_point(a)
        original = a.unperturbed
        perturbed = a.perturbed
        additional = {'iterations': list(),  # Perturbed images and L2 distance for every iteration
                      'initial': perturbed,  # Starting point of the attack
                      'model_calls': {'initialization': self.model_interface.model_calls,
                                      'projection': -1,
                                      'iters': list()},  # cumulative model calls
                      'manifold': list(),  # Captures probability manifold
                      'progression': list(),
                      'timing': {'initial': time.time(), 'iters': list()},
                      'grad_num_evals': list(), # track number of evals in approximate gradient
                      'cosine_details': list()  # Details of grad cosines (true vs line vs estimate)
                      }

        def decision_function(x, freq, average=False, remember=True):
            # returns 1 if adversarial
            outs = []
            num_batchs = int(math.ceil(len(x) * 1.0 / self.batch_size))
            for j in range(num_batchs):
                current_batch = x[self.batch_size * j: self.batch_size * (j + 1)]
                out = self.model_interface.forward(images=current_batch,
                                                   a=a, freq=freq, average=average, remember=remember)
                outs.append(out)
            outs = torch.cat(outs, dim=0)
            return outs

        if flags['stats_manifold']:
            ss = self.screenshot_manifold(perturbed, original)
            additional['manifold_init'] = ss

        logging.info('Binary Search to project to boundary...')

        if self.search == "binary" or self.hsja:
            perturbed, dist_post_update = self.binary_search_batch(
                original, perturbed[None], decision_function
            )
        else:
            perturbed, dist_post_update, s_, _ = self.info_max_batch2(
                original, perturbed[None], decision_function, self.grid_size, a.true_label
            )
        additional['timing']['init_search'] = time.time()
        additional['progression'].append({'binary': perturbed, 'approx_grad':additional['initial']})
        additional['model_calls']['projection'] = self.model_interface.model_calls
        dist = self.compute_distance(perturbed, original)
        distance = a.distance
        for step in range(1, iterations + 1):
            additional['timing']['iters'].append({'start': time.time()})
            additional['progression'].append(dict())
            additional['model_calls']['iters'].append(dict())
            logging.info('Step %d...' % step)
            # Choose delta.
            delta = self.select_delta(dist_post_update, step)

            # Choose number of evaluations.
            num_evals_det = int(min([self.initial_num_evals * np.sqrt(step), self.max_num_evals]))
            # self.grad_sampling_freq = self.sampling_freq

            if self.hsja:
                gradf = self.approximate_gradient(
                    decision_function, perturbed, num_evals_det, delta, average
                )
            else:
                target_cos = get_cos_from_n(num_evals_det, theta=self.theta_det, delta=delta / dist_post_update, d=self.d)
                num_evals_prob = int(get_n_from_cos(target_cos, s=s_, theta=(1/100), delta=(np.sqrt(self.d)/100), d=self.d))
                additional['grad_num_evals'].append(num_evals_prob)
                num_evals_prob = int(min(num_evals_prob, self.max_num_evals))
                logging.info('Approximating grad with %d evaluation...' % num_evals_det)
                additional['timing']['iters'][-1]['num_evals'] = time.time()

                gradf = self.approximate_gradient(
                    decision_function, perturbed, num_evals_prob, dist_post_update*np.sqrt(self.d)/100, average
                )
            additional['timing']['iters'][-1]['approx_grad'] = time.time()
            additional['model_calls']['iters'][-1]['approx_grad'] = self.model_interface.model_calls

            if self.constraint == "linf":
                update = np.sign(gradf)
            else:
                update = gradf

            if flags['stats_cosines']:
                cos_details = self.capture_cosines(perturbed, original, gradf, a.true_label, decision_function)
                additional['cosine_details'].append(cos_details)

            logging.info('Binary Search back to the boundary')
            if self.stepsize_search == "geometric_progression":
                # find step size.
                epsilon = self.geometric_progression_for_stepsize(
                    perturbed, update, dist, decision_function, step, original
                )
                additional['model_calls']['iters'][-1]['step_search'] = self.model_interface.model_calls
                additional['timing']['iters'][-1]['step_search'] = time.time()

                # Update the sample.
                perturbed = torch.clamp(perturbed + epsilon * update, self.clip_min, self.clip_max)
                additional['progression'][-1]['approx_grad'] = perturbed
                if flags['stats_manifold']:
                    ss = self.screenshot_manifold(perturbed, original)
                    additional['manifold'].append(ss)

                # Go in the opposite direction
                if not self.hsja:
                    perturbed = torch.clamp(2*perturbed - original, self.clip_min, self.clip_max)
                additional['progression'][-1]['opposite'] = perturbed

                # Binary search to return to the boundary.
                if self.search == "binary" or self.hsja:
                    perturbed, dist_post_update = self.binary_search_batch(
                        original, perturbed[None], decision_function
                    )
                else:
                    perturbed, dist_post_update, s_, (tmap, xx) = self.info_max_batch2(
                        original, perturbed[None], decision_function, self.grid_size, a.true_label
                    )
                    additional['progression'][-1]['tmap'] = tmap
                    additional['progression'][-1]['samples'] = xx
                    additional['timing']['iters'][-1]['bin_search'] = time.time()

                    # _check = decision_function(perturbed[None], self.sampling_freq)[0]
                    # assert _check == 1
                additional['progression'][-1]['binary'] = perturbed
                additional['model_calls']['iters'][-1]['binary'] = self.model_interface.model_calls

            elif self.stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
                perturbeds = np.clip(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = decision_function(perturbeds, self.sampling_freq)

                if (idx_perturbed == 1).any():
                    # Select the perturbation that yields the minimum distance after binary search.
                    perturbed, dist_post_update = self.binary_search_batch(
                        original, perturbeds[idx_perturbed], decision_function
                    )
            additional['timing']['iters'][-1]['end'] = time.time()

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
            point = (2 * alpha - 1) * original + (2 - 2 * alpha) * perturbed
            probs = self.model_interface.get_probs(point)
            screenshot[alpha] = probs.flatten()
        return screenshot

    def capture_cosines(self, perturbed, original, gradf, true_label, decision_function):
        grad_st_line = perturbed - original
        # boundary, _ = self.binary_search_batch(original, perturbed[None], decision_function, cosine=True)
        # grad_true = self.model_interface.get_grads(boundary[None], true_label)
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

        # # Binary search to minimize l2 distance to the original input.
        # low = 0.0
        # high = 1.0
        # # while high - low > 0.001:
        # while high - low > 0.2:  # Kept it high so as to keep infomax in check
        #     mid = (high + low) / 2.0
        #     blended = (1 - mid) * a.unperturbed + mid * random_noise
        #     success = self.model_interface.forward_one(blended, a, self.sampling_freq)
        #     # when model is confused, it is not adversarial
        #     logging.info(a.distance)
        #     if success == 1:
        #         high = mid
        #     else:
        #         low = mid

    def info_max_batch2(self, unperturbed, perturbed_inputs, decision_function, grid_size, true_label):
        border_points = []
        dists = []
        smaps = []
        for perturbed_input in perturbed_inputs:
            output = bin_search(unperturbed, perturbed_input, self.model_interface, d=self.d,
                                grid_size=grid_size, device=get_device(), true_label=true_label)
            nn_tmap_est = output['nn_tmap_est']
            t_map, s_map = output['tts_max'][-1]
            # t_map, s_map = t_map.numpy(), s_map.numpy()
            border_point = (1 - t_map) * unperturbed + t_map * perturbed_input
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
            decision_function(out[None], freq=1, remember=True)  # this is to make the model remember the sample
        else:
            decision_function(out[None], freq=self.sampling_freq*32, remember=True)  # this is to make the model remember the sample
        return out, dist, smaps[idx], (nn_tmap_est, output['xxj'])

    # This function is deprecated now
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
                projection = (1 - xj) * unperturbed + xj * perturbed_input
                projection = (1 - xj) * unperturbed + xj * perturbed_input
                yj = int(decision_function(projection[None], freq=1))
                xjs.append(xj)
                yjs.append(yj)

                pyj_txj = py_tx[yj:(yj + 1), :, imax:(imax + 1)]
                pyj_xj = py_x[yj:(yj + 1), :, imax:(imax + 1)]
                pt_x = pyj_txj * pt_x / pyj_xj

            border_point = (1 - t_map) * unperturbed + t_map * perturbed_input
            dist = self.compute_distance(unperturbed, border_point)
            border_points.append(border_point)
            dists.append(dist)
        idx = np.argmin(np.array(dists))
        dist = self.compute_distance(unperturbed, perturbed_inputs[idx])
        out = border_points[idx]
        return out, dist

    def binary_search_batch(self, unperturbed, perturbed_inputs, decision_function, cosine=False):
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
            thresholds = dists_post_update * self.theta_det
        else:
            highs = np.ones(len(perturbed_inputs))
            # thresholds = self.theta * 1000  # remove 1000 later
            thresholds = self.theta_det  # remove 1000 later
            if cosine:
                thresholds /= self.d

        lows = np.zeros(len(perturbed_inputs))

        # Call recursive function.
        while np.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs, mids)

            # Update highs and lows based on model decisions.
            # decisions = decision_function(mid_inputs, self.sampling_freq, remember=not cosine)
            decisions = decision_function(mid_inputs, self.sampling_freq, remember=self.remember)
            decisions[decisions == -1] = 0
            lows = np.where(decisions == 0, mids, lows)
            highs = np.where(decisions == 1, mids, highs)

        out_inputs = self.project(unperturbed, perturbed_inputs, highs)

        # Compute distance of the output to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = np.array([self.compute_distance(unperturbed, out) for out in out_inputs])
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out = out_inputs[idx]
        if self.hsja:
            decision_function(out[None], freq=1, remember=True)  # this is to make the model remember the sample
        else:
            decision_function(out[None], freq=self.sampling_freq*32, remember=True)  # this is to make the model remember the sample
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
                delta = np.sqrt(self.d) * self.theta_det * dist_post_update
            elif self.constraint == "linf":
                delta = self.d * self.theta_det * dist_post_update
            else:
                raise RuntimeError("Unknown constraint metric: {}".format(self.constraint))
        return delta

    def approximate_gradient(self, decision_function, sample, num_evals, delta, average=False):
        """ Gradient direction estimation """
        # Generate random vectors.
        noise_shape = [num_evals] + list(self.shape)
        if self.constraint == "l2":
            if torch.cuda.is_available():
                rv = torch.cuda.FloatTensor(*noise_shape).normal_()
            else:
                rv = torch.FloatTensor(*noise_shape).normal_()
        elif self.constraint == "linf":
            rv = np.random.uniform(low=-1, high=1, size=noise_shape)
        else:
            raise RuntimeError("Unknown constraint metric: {}".format(self.constraint))

        axis = tuple(range(1, 1 + len(self.shape)))
        rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=axis, keepdim=True))
        perturbed = sample + delta * rv
        perturbed = torch.clamp(perturbed, self.clip_min, self.clip_max)
        rv = (perturbed - sample) / delta

        # query the model.
        outputs = decision_function(perturbed, self.grad_sampling_freq, average, remember=self.remember)
        decisions = outputs[outputs != -1]
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = 2 * decisions.view(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        vals = fval if torch.abs(torch.mean(fval)) == 1.0 else fval - torch.mean(fval)
        rv = rv[outputs != -1]
        gradf = torch.mean(vals * rv, dim=0)

        # Get the gradient direction.
        if torch.sum(outputs != -1) == 0:
            return None
        gradf = gradf / torch.norm(gradf)

        return gradf

    def geometric_progression_for_stepsize(
            self, x, update, dist, decision_function, current_iteration, original=None
    ):
        """ Geometric progression to search for stepsize.
          Keep decreasing stepsize by half until reaching
          the desired side of the boundary.
        """
        epsilon = dist / np.sqrt(current_iteration)
        if self.hsja:
            count = 1
            while True:
                if count % 200 == 0:
                    logging.warning("Decreased epsilon {} times".format(count))
                updated = np.clip(x + epsilon * update, self.clip_min, self.clip_max)
                success = (decision_function(updated[None], self.sampling_freq, remember=self.remember))[0]
                if success:
                    break
                else:
                    epsilon = epsilon / 2.0  # pragma: no cover
                count += 1
        return epsilon

    def compute_distance(self, x1, x2):
        if self.constraint == "l2":
            return torch.norm(x1 - x2)
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
