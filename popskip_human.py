import math

import matplotlib.pyplot as plt
import torch

from abstract_attack import Attack
from defaultparams import DefaultParams
from tracker import InfoMaxStats
from infomax import bin_search, get_cos_from_n


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
        # perturbed, dist_post_update = self.binary_search(original, perturbed)
        # return perturbed, dist_post_update, None
        perturbed, dist_post_update, s_, e_, t_, n_, (nn_tmap, xx) = self.info_max_batch(
            original, perturbed[None], self.a.true_label, estimates, step)
        if page is not None:
            page.info_max_stats = InfoMaxStats(s_, t_, xx, e_, n_)
        return perturbed, dist_post_update, {'s': s_, 'e': e_, 'n': n_, 't': t_}

    def info_max_batch(self, unperturbed, perturbed_inputs, label, estimates, step):
        if self.prior_frac == 0:
            if step is None or step <= 1:
                prior_frac = 1
            elif step <= 4:
                prior_frac = 0.5
            elif step <= 10:
                prior_frac = 0.2
            else:
                prior_frac = 0.1
        else:
            prior_frac = self.prior_frac
        border_points = []
        dists = []
        smaps, tmaps, emaps, ns = [], [], [], []
        if estimates is None:
            target_cos = get_cos_from_n(self.initial_num_evals, theta=self.theta_det, delta=self.delta_det_unit, d=self.d)
        else:
            num_evals_det = int(min([self.initial_num_evals * math.sqrt(step+1), self.max_num_evals]))
            target_cos = get_cos_from_n(num_evals_det, theta=self.theta_det, delta=self.delta_det_unit, d=self.d)
        grid_size_dynamic = self.grid_size
        for perturbed_input in perturbed_inputs:
            output, n = bin_search(
                unperturbed, perturbed_input, self.model_interface, d=self.d,
                grid_size=grid_size_dynamic, device=self.device, delta=self.delta_prob_unit,
                label=label, targeted=self.targeted, prev_t=self.prev_t, prev_s=self.prev_s,
                prev_e=self.prev_e, prior_frac=prior_frac, target_cos=target_cos,
                queries=self.queries, plot=False, stop_criteria=self.stop_criteria, dist_metric=self.constraint,
                human_interface=self.decision_bin)
            nn_tmap_est = output['nn_tmap_est']
            t_map, s_map, e_map = output['ttse_max'][-1]
            num_retries = 0
            while t_map == 1 and num_retries < 5:
                num_retries += 1
                print(f'Got t_map == 1, Retrying {num_retries}...')
                output, n = bin_search(
                    unperturbed, perturbed_input, self.model_interface, d=self.d,
                    grid_size=grid_size_dynamic, device=self.device, delta=self.delta_prob_unit,
                    label=label, targeted=self.targeted, prev_t=self.prev_t, prev_s=self.prev_s,
                    prev_e=self.prev_e, prior_frac=prior_frac, target_cos=target_cos,
                    queries=self.queries, plot=False, stop_criteria=self.stop_criteria, dist_metric=self.constraint,
                    human_interface=self.decision_bin)
                nn_tmap_est = output['nn_tmap_est']
                t_map, s_map, e_map = output['ttse_max'][-1]
            if t_map == 1:
                print('Prob of label (unperturbed):', self.model_interface.get_probs(unperturbed)[0, label])
                print('Prob of label (perturbed):', self.model_interface.get_probs(perturbed_input)[0, label])
                space = [(1 - tt) * perturbed_input + tt * unperturbed for tt in torch.linspace(0, 1, 21)]
                print([self.model_interface.get_probs(x)[0, label] for x in space])
                print('delta:', self.delta_prob_unit)
                print('label:', label)
                print('prev_t,s,e:', self.prev_t, self.prev_s, self.prev_e)
                print('prior_frac:', prior_frac)
                print('target_cos', target_cos)
                torch.save(unperturbed, open('dumps/unperturbed.pkl', 'wb'))
                torch.save(perturbed_input, open('dumps/perturbed.pkl', 'wb'))
                t_map = 1.0 - 0.5 / self.grid_size
                # torch.save(self.model_interface, open('dumps/model_interface.pkl', 'wb'))

            if self.constraint == 'l2':
                border_point = (1 - t_map) * perturbed_input + t_map * unperturbed
            elif self.constraint == 'linf':
                dist_linf = self.compute_distance(unperturbed, perturbed_input)
                alphas = (1 - t_map) * dist_linf
                border_point = self.project(unperturbed, perturbed_input, alphas[None])[0]
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
