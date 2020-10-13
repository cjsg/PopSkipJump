import logging
import torch
import math
import time
from tracker import Diary, DiaryPage
from defaultparams import DefaultParams
from adversarial import Adversarial
from model_interface import ModelInterface


class Attack:
    def __init__(self, model_interface, data_shape, device=None, params: DefaultParams = None):
        self.model_interface: ModelInterface = model_interface
        self.initial_num_evals = params.initial_num_evals
        self.max_num_evals = params.max_num_evals
        self.gamma = params.gamma
        self.batch_size = params.batch_size
        self.internal_dtype = params.internal_dtype
        self.bounds = params.bounds
        self.clip_min, self.clip_max = params.bounds
        self.sampling_freq = params.sampling_freq_binsearch
        self.device = device
        self.prev_t = None
        self.prev_s = None
        self.prev_e = None
        self.prior_frac = params.prior_frac
        self.queries = params.queries
        self.grad_queries = params.grad_queries

        # Set constraint based on the distance.
        if params.distance == 'MSE':
            self.constraint = "l2"
        elif params.distance == 'Linf':
            self.constraint = "linf"

        # Set binary search threshold.
        self.shape = data_shape
        self.d = int(torch.prod(torch.tensor(self.shape)))
        self.grid_size = params.grid_size
        if self.constraint == "l2":
            self.theta_det = self.gamma / (math.sqrt(self.d) * self.d)
            # self.theta = self.gamma / (np.sqrt(self.d))  # Based on CJ experiment
        else:
            self.theta_det = self.gamma / (self.d * self.d)

    def attack(self, images, labels, starts=None, iterations=64):
        raw_results = []
        distances = []
        for i, (image, label) in enumerate(zip(images, labels)):
            logging.warning("Attacking Image: {}".format(i))
            a = Adversarial(image=image, label=label, device=self.device)
            if starts is not None:
                a.set_starting_point(starts[i], self.bounds)
            self.reset_variables(a)
            self.attack_one(iterations)
            if len(self.diary.iterations) > 0:
                distances.append(a.distance)
            raw_results.append(self.diary)
        median = torch.median(torch.tensor(distances))
        return median, raw_results

    def perform_initialization(self):
        if self.a.perturbed is None:
            logging.info('Initializing Starting Point...')
            self.initialize_starting_point(self.a)

    def bin_search_step(self, original, perturbed, page=None):
        """
        Performs Binary Search between original and perturbed to find the closest point to the adversarial boundary
        :param original: x_star (The original image)
        :param perturbed: \tilde{x}_t (Result of Initialization or Opposite Movement step)
        :param page: Data Structure for remembering statistics for plotting later
        :return: x_t (Closest image to boundary)
        """
        raise NotImplementedError

    def gradient_approximation_step(self, perturbed, num_evals_det, delta, dist_post_update, estimates, page):
        """
        Estimates the direction of gradient by doing random sampling in the sphere and then observing model decisions
        :param perturbed: Point at which the gradient direction is being estimated
        :param num_evals_det: HSJA's estimate of num of samples to use in gradient estimation
        :param delta: HSJA's estimate of the radius of the sphere
        :param dist_post_update: Distance between perturbed and original image
        :param estimates: Statistics regarding the sigmoid approximation of probability manifold
        :param page: Book-keeping data structure
        :return:
        """
        raise NotImplementedError

    def opposite_movement_step(self, original, perturbed):
        """
        This is step performed by PSJA where we move the current point in the direction opposite to original image
        :param original: x_star
        :param perturbed: \tilde{x_t}
        :return:
        """
        raise NotImplementedError

    def attack_one(self, iterations=64):
        self.diary.epoch_start = time.time()

        self.perform_initialization()
        original, perturbed = self.a.unperturbed, self.a.perturbed

        self.diary.initial_image = self.a.perturbed
        self.diary.initialization_calls = self.model_interface.model_calls
        self.diary.epoch_initialization = time.time()

        perturbed, dist_post_update, estimates = self.bin_search_step(original, perturbed)
        self.diary.epoch_initial_bin_search = time.time()
        self.diary.initial_projection = perturbed
        self.diary.calls_initial_bin_search = self.model_interface.model_calls

        dist = self.compute_distance(perturbed, original)
        distance = self.a.distance
        for step in range(1, iterations + 1):
            page = DiaryPage()
            page.time.start = time.time()
            page.calls.start = self.model_interface.model_calls

            delta = self.select_delta(dist_post_update, step)
            num_evals_det = int(min([self.initial_num_evals * math.sqrt(step), self.max_num_evals]))
            gradf = self.gradient_approximation_step(perturbed, num_evals_det, delta, dist_post_update,
                                                     estimates, page)
            page.num_eval_det = num_evals_det
            page.time.approx_grad = time.time()
            page.calls.approx_grad = self.model_interface.model_calls

            update = gradf if self.constraint == 'l2' else torch.sign(gradf)

            # find step size.
            epsilon = self.geometric_progression_for_stepsize(perturbed, update, dist, step, original)
            page.time.step_search = time.time()
            page.calls.step_search = self.model_interface.model_calls

            # Update the sample.
            perturbed = torch.clamp(perturbed + epsilon * update, self.clip_min, self.clip_max)
            page.approx_grad = perturbed

            perturbed = self.opposite_movement_step(original, perturbed)
            page.opposite = perturbed

            # Binary search to return to the boundary.
            perturbed, dist_post_update, estimates = self.bin_search_step(original, perturbed, page)
            page.time.bin_search = time.time()
            page.calls.bin_search = self.model_interface.model_calls
            page.bin_search = perturbed

            # compute new distance.
            dist = self.compute_distance(perturbed, original)
            if self.constraint == "l2":
                distance = dist ** 2 / self.d / (self.clip_max - self.clip_min) ** 2
            elif self.constraint == "linf":
                distance = dist / (self.clip_max - self.clip_min)
            logging.info('distance of adversarial = %f', distance)

            page.time.end = time.time()
            page.calls.end = self.model_interface.model_calls
            page.perturbed = self.a.perturbed
            page.distance = self.a.distance
            self.diary.iterations.append(page)
        return self.diary

    def reset_variables(self, a):
        self.model_interface.model_calls = 0
        self.a: Adversarial = a
        self.prev_t = None
        self.prev_s = None
        self.prev_e = None
        self.diary = Diary(a.unperturbed, a.true_label)

    def initialize_starting_point(self, a):
        num_evals = 0
        while True:
            random_noise = torch.rand(size=self.shape) * (self.clip_max - self.clip_min) + self.clip_min
            success = self.model_interface.forward(random_noise[None], a, self.sampling_freq)
            # when model is confused, it is not adversarial
            num_evals += 1
            if success == 1:
                break
            if num_evals > 1e4:
                return

    def generate_random_vectors(self, batch_size):
        noise_shape = [int(batch_size)] + list(self.shape)
        if self.constraint == "l2":
            rv = torch.randn(size=noise_shape, device=self.device)
            # if torch.cuda.is_available():
            #     rv = torch.cuda.FloatTensor(*noise_shape).normal_()
            # else:
            #     rv = torch.FloatTensor(*noise_shape).normal_()
        elif self.constraint == "linf":
            rv = 2 * torch.rand(size=noise_shape) - 1  # random vector between -1 and +1
        else:
            raise RuntimeError("Unknown constraint metric: {}".format(self.constraint))
        axis = tuple(range(1, 1 + len(self.shape)))
        rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=axis, keepdim=True))
        return rv

    def _gradient_estimator(self, sample, num_evals, delta):
        """ Computes an approximation by querying every point `grad_queries` times"""
        # Generate random vectors.
        num_rvs = int(num_evals/self.grad_queries)
        sum_directions = torch.zeros(self.shape, device=self.device)
        num_batchs = int(math.ceil(num_rvs * 1.0 / self.batch_size))
        for j in range(num_batchs):
            batch_size = min(self.batch_size, num_rvs - j*self.batch_size)
            rv = self.generate_random_vectors(batch_size)
            perturbed = sample + delta * rv
            perturbed = torch.clamp(perturbed, self.clip_min, self.clip_max)
            rv = (perturbed - sample) / delta
            decisions = self.model_interface.decision(perturbed, self.a.true_label, self.grad_queries)
            decisions = decisions.sum(dim=1)
            decision_shape = [len(decisions)] + [1] * len(self.shape)
            # Map (0, 1) -> (-1, +1)
            fval = 2 * decisions.view(decision_shape) - self.grad_queries
            # Baseline subtraction (when fval differs)
            if torch.abs(torch.mean(fval/self.grad_queries)) == 1.0:
                vals = fval
            else:
                vals = fval - torch.mean(fval)
            sum_directions = sum_directions + torch.sum(vals * rv, dim=0)
        # Get the gradient direction.
        gradf = sum_directions / (num_rvs*self.grad_queries)
        gradf = gradf / torch.norm(gradf)
        return gradf

    def geometric_progression_for_stepsize(self, x, update, dist, current_iteration, original=None):
        """
            Decides the appropriate step-size in the estimated gradient direction
        """
        raise NotImplementedError

    def compute_distance(self, x1, x2):
        if self.constraint == "l2":
            return torch.norm(x1 - x2)
        elif self.constraint == "linf":
            return torch.max(torch.abs(x1 - x2))

    def select_delta(self, dist_post_update, current_iteration):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.
        """
        if current_iteration == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.constraint == "l2":
                delta = math.sqrt(self.d) * self.theta_det * dist_post_update
            elif self.constraint == "linf":
                delta = self.d * self.theta_det * dist_post_update
            else:
                raise RuntimeError("Unknown constraint metric: {}".format(self.constraint))
        return delta

    def calculate_grad(self, decisions, rv):
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = 2 * decisions.view(decision_shape) - 1.0
        # Baseline subtraction (when fval differs)
        vals = fval if torch.abs(torch.mean(fval)) == 1.0 else fval - torch.mean(fval)
        gradf = torch.mean(vals * rv, dim=0)
        # Get the gradient direction.
        gradf = gradf / torch.norm(gradf)
        return gradf
